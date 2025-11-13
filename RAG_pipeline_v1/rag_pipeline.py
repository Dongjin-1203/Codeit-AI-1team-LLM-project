from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from rag_config import RAGConfig
from rag_retriever import RAGRetriever


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

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

        # Retriever 초기화
        self.retriever = RAGRetriever(config=self.config)

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

    def generate_answer(
        self,
        query: str,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        질문에 대한 답변 생성
        
        Args:
            query: 사용자 질문
            temperature: LLM temperature (None이면 config 기본값)
            max_tokens: 최대 토큰 수 (None이면 config 기본값)
            
        Returns:
            답변 및 메타데이터 딕셔너리
        """
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE
        if max_tokens is None:
            max_tokens = self.config.DEFAULT_MAX_TOKENS

        # 1. 검색
        retrieved_docs = self.retriever.search(query, top_k=self.top_k)

        # 2. 프롬프트 구성
        api_messages = self._build_prompt(query, retrieved_docs)

        # 3. LLM 호출
        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 답변 추출
        answer = response.choices[0].message.content

        # 4. 결과 구조화
        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'filename': doc['filename'],
                    'organization': doc['organization'],
                    'relevance_score': doc['relevance_score'],
                    'content_preview': doc['content'][:100] + "..."
                }
                for doc in retrieved_docs
            ],
            'model': self.model,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }

        return result

    def print_result(self, result: dict):
        """결과 출력"""
        print("\n" + "="*60)
        print(f"질문: {result['query']}")
        print("="*60)

        print(f"\n💬 답변:\n{result['answer']}")

        print(f"\n📚 참고 문서 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source['filename']}")
            print(f"      발주기관: {source['organization']}")
            print(f"      관련도: {source['relevance_score']:.3f}")

        print(f"\n📊 사용량:")
        print(f"  모델: {result['model']}")
        print(f"  토큰: {result['usage']['total_tokens']} "
              f"(입력: {result['usage']['prompt_tokens']}, "
              f"출력: {result['usage']['completion_tokens']})")

        print("="*60)