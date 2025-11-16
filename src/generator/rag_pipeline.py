from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import time
from typing import Optional

from utils.rag_config import RAGConfig
from retriever.rag_retriever import RAGRetriever


class RAGPipeline:
    """RAG 파이프라인 - 검색 기반 답변 생성"""

    def __init__(self, config: RAGConfig = None, model: str = None, top_k: int = None):
        """
        초기화
        
        Args:
            config: RAG 설정 객체
            model: LLM 모델명 (None이면 config 기본값)
            top_k: 검색할 문서 수 (None이면 config 기본값)
        """
        self.config = config or RAGConfig()
        self.model = model or self.config.LLM_MODEL_NAME
        self.top_k = top_k or self.config.DEFAULT_TOP_K

        # OpenAI 클라이언트 초기화 (timeout 설정)
        self.client = OpenAI(
            api_key=self.config.OPENAI_API_KEY,
            timeout=60.0  # 60초 timeout
        )

        # Retriever 초기화
        self.retriever = RAGRetriever(config=self.config)
        
        # Temperature 지원 여부 (첫 호출에서 자동 감지)
        self._supports_temperature = None

        # 프롬프트 템플릿 정의
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공공기관 정보시스템 구축 사업 전문가입니다.
검색된 문서를 참고하여 정확하게 답변해주세요.

중요한 규칙:
1. 반드시 제공된 문서의 내용만을 기반으로 답변하세요
2. 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요
3. 답변 시 출처를 명시해주세요
4. 간결하고 명확하게 답변하세요"""),

            ("user", """검색된 문서:
{context}

사용자 질문: {query}""")
        ])

        print(f"RAG 파이프라인 초기화 완료 (모델: {self.model})")

    def _format_context(self, retrieved_docs: list) -> str:
        """검색된 문서를 컨텍스트 문자열로 변환"""
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[문서 {i}]\n"
                f"출처: {doc['filename']}\n"
                f"발주기관: {doc['organization']}\n"
                f"관련도: {doc['relevance_score']:.3f}\n"
                f"내용: {doc['content']}\n"
            )

        return "\n".join(context_parts)

    def _build_prompt(self, query: str, retrieved_docs: list):
        """프롬프트 구성 및 OpenAI API 형식으로 변환"""
        # context 생성
        context = self._format_context(retrieved_docs)

        # 메시지 생성
        messages = self.chat_prompt.format_messages(
            context=context,
            query=query
        )

        # OpenAI API 형식으로 변환
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})

        return api_messages

    def _format_sources(self, retrieved_docs: list) -> list:
        """검색된 문서를 sources 형식으로 변환"""
        sources = []
        
        for doc in retrieved_docs:
            sources.append({
                'content': doc['content'],
                'metadata': doc['metadata'],
                'score': doc['relevance_score'],
                'filename': doc['filename'],
                'organization': doc['organization']
            })
        
        return sources

    def _call_openai_api(self, messages: list, use_temperature: bool = True):
        """
        OpenAI API 호출 (파라미터 조합 처리)
        
        Args:
            messages: 메시지 리스트
            use_temperature: temperature 사용 여부
            
        Returns:
            response 객체
        """
        # 기본 파라미터
        params = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.config.DEFAULT_MAX_TOKENS
        }
        
        # Temperature 추가 (지원하는 경우)
        if use_temperature and self._supports_temperature is not False:
            params["temperature"] = 0.7
        
        try:
            response = self.client.chat.completions.create(**params)
            
            # 처음으로 성공하면 temperature 지원 확인
            if self._supports_temperature is None and use_temperature:
                self._supports_temperature = True
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Temperature 관련 에러 감지
            if 'temperature' in error_msg and 'unsupported' in error_msg:
                if self._supports_temperature is None:
                    print(f"   ℹ️ 이 모델은 temperature를 지원하지 않습니다. 기본값 사용.")
                    self._supports_temperature = False
                
                # Temperature 없이 재시도
                if use_temperature:
                    return self._call_openai_api(messages, use_temperature=False)
            
            # 다른 에러는 전파
            raise

    def _call_openai_with_retry(
        self, 
        messages: list, 
        max_retries: int = 3
    ) -> Optional[dict]:
        """
        재시도 로직이 있는 OpenAI API 호출
        
        Returns:
            성공시 response 객체, 실패시 None
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # API 호출 (temperature 자동 처리)
                response = self._call_openai_api(messages)
                
                # 응답 검증
                if not response or not response.choices:
                    raise ValueError("빈 응답 반환")
                
                if not response.choices[0].message.content:
                    raise ValueError("빈 메시지 내용")
                
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                # Rate limit 감지
                if 'rate limit' in error_msg or 'rate_limit' in error_msg:
                    wait_time = (2 ** attempt) * 5  # 5, 10, 20초
                    print(f"   ⚠️ Rate Limit 감지. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                
                # Timeout 감지
                elif 'timeout' in error_msg:
                    wait_time = (2 ** attempt) * 3  # 3, 6, 12초
                    print(f"   ⚠️ Timeout 발생. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                
                # 기타 에러
                else:
                    wait_time = 2 ** attempt  # 1, 2, 4초
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ API 호출 실패 (시도 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ 최종 실패: {str(e)[:100]}")
        
        # 모든 재시도 실패
        print(f"   ❌ OpenAI API 호출 최종 실패: {last_error}")
        return None

    def generate_answer(
        self, 
        query: str, 
        top_k: int = None,
        temperature: float = None,
        max_retries: int = 3
    ) -> dict:
        """
        답변 생성
        
        Args:
            query: 사용자 질문
            top_k: 검색할 문서 수
            temperature: LLM temperature (사용 안 함)
            max_retries: 최대 재시도 횟수
            
        Returns:
            dict: {'answer': str, 'sources': list, 'usage': dict}
        """
        try:
            k = top_k if top_k is not None else self.config.DEFAULT_TOP_K
            
            # 1. 문서 검색
            retrieved_docs = self.retriever.search(query, top_k=k)
            
            if not retrieved_docs:
                return {
                    'answer': "관련 문서를 찾을 수 없습니다.",
                    'sources': [],
                    'usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
                }
            
            # 2. 프롬프트 생성
            messages = self._build_prompt(query, retrieved_docs)
            
            # 3. OpenAI API 호출 (재시도 포함)
            response = self._call_openai_with_retry(messages, max_retries=max_retries)
            
            # 4. 응답 검증
            if response is None:
                # API 호출 실패 - 명확한 에러 표시
                raise RuntimeError("OpenAI API 호출 실패 (모든 재시도 소진)")
            
            answer = response.choices[0].message.content
            
            if not answer or answer.strip() == "":
                raise ValueError("빈 답변 생성됨")
            
            # 5. 결과 반환
            return {
                'answer': answer,
                'sources': self._format_sources(retrieved_docs),
                'usage': {
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            }
        
        except Exception as e:
            # 에러 로깅
            print(f"❌ 답변 생성 최종 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 전파 (평가 코드에서 감지 가능하도록)
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e

    def print_result(self, result: dict):
        """결과 출력"""
        print("\n" + "="*60)
        print(f"질문: {result.get('query', 'N/A')}")
        print("="*60)

        print(f"\n💬 답변:\n{result['answer']}")

        print(f"\n📚 참고 문서 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['filename']}")
            print(f"      발주기관: {source['organization']}")
            print(f"      관련도: {source['score']:.3f}")

        print(f"\n📊 사용량:")
        print(f"  모델: {self.model}")
        print(f"  토큰: {result['usage']['total_tokens']} "
              f"(입력: {result['usage']['prompt_tokens']}, "
              f"출력: {result['usage']['completion_tokens']})")

        print("="*60)