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
        """초기화"""
        self.config = config or RAGConfig()
        self.model = model or self.config.LLM_MODEL_NAME
        self.top_k = top_k or self.config.DEFAULT_TOP_K

        self.client = OpenAI(
            api_key=self.config.OPENAI_API_KEY,
            timeout=60.0
        )

        self.retriever = RAGRetriever(config=self.config)

        # 🔥 간결하고 직접적인 프롬프트
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공공기관 정보시스템 구축 사업 전문가입니다.
검색된 문서를 기반으로 질문에 직접적이고 정확하게 답변하세요.

답변 원칙:
1. 핵심 정보만 간결하게 제시하세요
2. 질문이 묻는 것에만 정확히 답하세요
3. 불필요한 서론이나 부연설명은 최소화하세요
4. 출처 표기는 하지 마세요
5. 문서에서 정보를 찾았다면 자연스럽게 답변하세요

답변 스타일:
- "~입니다", "~해야 합니다" 같은 단정적 표현 사용
- 구체적인 숫자, 기준, 조건 포함
- 한 문장 또는 짧은 문단으로 답변"""),

            ("user", """검색된 문서:
{context}

질문: {query}

위 문서의 정보를 바탕으로 질문에 간결하고 정확하게 답변하세요.""")
        ])

        print(f"RAG 파이프라인 초기화 완료 (모델: {self.model})")

    def _format_context(self, retrieved_docs: list) -> str:
        """검색된 문서를 컨텍스트로 변환"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[문서 {i}]\n{doc['content']}\n")
        return "\n".join(context_parts)

    def _build_prompt(self, query: str, retrieved_docs: list):
        """프롬프트 구성"""
        context = self._format_context(retrieved_docs)
        messages = self.chat_prompt.format_messages(context=context, query=query)
        
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

    def _call_openai_with_retry(self, messages: list, max_retries: int = 3) -> Optional[dict]:
        """재시도 로직이 있는 OpenAI API 호출"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.config.DEFAULT_MAX_TOKENS
                )
                
                if not response or not response.choices:
                    raise ValueError("빈 응답")
                
                if not response.choices[0].message.content:
                    raise ValueError("빈 메시지")
                
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                
                if 'rate limit' in error_msg:
                    wait_time = (2 ** attempt) * 5
                    print(f"   ⚠️ Rate Limit. {wait_time}초 대기...")
                    time.sleep(wait_time)
                elif 'timeout' in error_msg:
                    wait_time = (2 ** attempt) * 3
                    print(f"   ⚠️ Timeout. {wait_time}초 대기...")
                    time.sleep(wait_time)
                else:
                    wait_time = 2 ** attempt
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ 실패 (시도 {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
        
        print(f"   ❌ API 호출 최종 실패: {last_error}")
        return None

    def generate_answer(
        self, 
        query: str, 
        top_k: int = None,
        temperature: float = None,
        max_retries: int = 3
    ) -> dict:
        """답변 생성"""
        try:
            k = top_k if top_k is not None else self.config.DEFAULT_TOP_K
            
            # 문서 검색
            retrieved_docs = self.retriever.search(query, top_k=k)
            
            if not retrieved_docs:
                return {
                    'answer': "관련 문서를 찾을 수 없습니다.",
                    'sources': [],
                    'usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
                }
            
            # 프롬프트 생성
            messages = self._build_prompt(query, retrieved_docs)
            
            # API 호출
            response = self._call_openai_with_retry(messages, max_retries=max_retries)
            
            if response is None:
                raise RuntimeError("API 호출 실패")
            
            answer = response.choices[0].message.content
            
            if not answer or answer.strip() == "":
                raise ValueError("빈 답변")
            
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
            print(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"      관련도: {source['score']:.3f}")
        print(f"\n📊 토큰: {result['usage']['total_tokens']}")
        print("="*60)