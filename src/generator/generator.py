from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
import time
from typing import List, Dict

from src.utils.config import RAGConfig
from src.retriever.retriever import RAGRetriever
from src.router.query_router import QueryRouter


class RAGPipeline:
    """대화형 RAG 파이프라인 - LangChain Chain 기반"""

    def __init__(self, config: RAGConfig = None, model: str = None, top_k: int = None):
        """초기화"""
        self.config = config or RAGConfig()
        self.model = model or self.config.LLM_MODEL_NAME
        self.top_k = top_k or self.config.DEFAULT_TOP_K
        
        # 검색 설정
        self.search_mode = self.config.DEFAULT_SEARCH_MODE
        self.alpha = self.config.DEFAULT_ALPHA

        # LLM 초기화 (LangChain ChatOpenAI)
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=self.config.OPENAI_API_KEY,
            timeout=60.0,
            max_retries=3
        )

        # Retriever 및 라우터 초기화
        self.retriever = RAGRetriever(config=self.config)
        self.router = QueryRouter()
        self._direct_responses = {
            'greeting': "안녕하세요! 공공입찰 RFP 관련 궁금한 사항을 알려주시면 자료를 찾아 드릴게요.",
            'thanks': "도움이 되었다니 다행입니다. 추가로 궁금한 점이 있으면 언제든지 말씀해 주세요!",
            'out_of_scope': "해당 질문은 현재 보유한 입찰·사업 문서에서 다루지 않습니다. 다른 질문을 시도해 주세요."
        }
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        # 마지막 검색 결과 저장 (sources 반환용)
        self._last_retrieved_docs = []

        # 프롬프트 템플릿 (대화 히스토리 포함)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공공입찰 RFP를 분석하는 입찰메이트 사내 분석가입니다. 제공된 컨텍스트만으로 요구사항·예산·대상 기관·제출 방식 등을 구조화해 의사결정을 지원하세요.

            # 규칙
            - 답변은 한국어로 작성합니다.
            - 컨텍스트 밖 내용을 추측하지 않습니다.
            - 컨텍스트가 비어있거나 질문과 직접 관련된 사실이 없으면 "문서에서 해당 정보를 찾을 수 없습니다." 한 문장으로만 답합니다.
            - 여러 문서를 비교할 때는 문서별 차이를 표 또는 목록으로 정리합니다.
            - 숫자에는 가능한 단위를 포함합니다.
            - 직전 대화 맥락을 반영하되, 확인되지 않은 내용을 추론해 추가하지 않습니다.

            # 답변 형식
            1. 한 줄 요약: 질문 핵심을 한두 문장으로 작성합니다.
            2. 상세 답변: [요구사항], [대상 기관], [예산], [제출 형식/방법], [평가 기준] 등 문서에서 확인된 항목만 정리합니다.
            3. 근거 정보: 위 답변의 근거가 된 문장이나 문단을 요약합니다.
            4. 부족한 정보: 문서에서 찾을 수 없는 항목은 "문서에서 확인 불가"로 표기합니다."""),
                        
                        # 대화 히스토리
                        MessagesPlaceholder(variable_name="chat_history"),
                        
                        # 현재 질문과 컨텍스트
                        ("user", """# 컨텍스트
            {context}

            # 질문
            {question}

            위 규칙에 따라 답변하세요.""")
        ])

        # Chain 구성
        self.chain = (
            {
                "context": RunnableLambda(self._retrieve_and_format),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda x: self._get_chat_history())
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print(f"✅ RAG 파이프라인 초기화 완료")
        print(f"   - 모델: {self.model}")
        print(f"   - 기본 top_k: {self.top_k}")
        print(f"   - 검색 모드: {self.search_mode}")

    def _get_chat_history(self) -> List:
        """대화 히스토리를 LangChain 메시지 형식으로 변환"""
        messages = []
        for msg in self.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def _retrieve_and_format(self, query: str) -> str:
        """검색 수행 및 컨텍스트 포맷팅"""
        # 검색 모드에 따라 문서 검색
        if self.search_mode == "embedding":
            docs = self.retriever.search(query, top_k=self.top_k)
        elif self.search_mode == "hybrid":
            docs = self.retriever.hybrid_search(query, top_k=self.top_k, alpha=self.alpha)
        elif self.search_mode == "hybrid_rerank":
            docs = self.retriever.hybrid_search_with_rerank(
                query, top_k=self.top_k, alpha=self.alpha
            )
        else:
            docs = self.retriever.search(query, top_k=self.top_k)
        
        # 마지막 검색 결과 저장
        self._last_retrieved_docs = docs
        
        # 컨텍스트 포맷팅
        return self._format_context(docs)

    def _format_context(self, retrieved_docs: list) -> str:
        """검색된 문서를 컨텍스트로 변환"""
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[문서 {i}]\n{doc['content']}\n")
        return "\n".join(context_parts)

    def _format_sources(self, retrieved_docs: list) -> list:
        """검색된 문서를 sources 형식으로 변환"""
        sources = []
        for doc in retrieved_docs:
            source_info = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'filename': doc.get('filename', 'N/A'),
                'organization': doc.get('organization', 'N/A')
            }
            
            # 검색 모드에 따라 점수 필드가 다름
            if 'rerank_score' in doc:
                source_info['score'] = doc['rerank_score']
                source_info['score_type'] = 'rerank'
            elif 'hybrid_score' in doc:
                source_info['score'] = doc['hybrid_score']
                source_info['score_type'] = 'hybrid'
            elif 'relevance_score' in doc:
                source_info['score'] = doc['relevance_score']
                source_info['score_type'] = 'embedding'
            else:
                source_info['score'] = 0
                source_info['score_type'] = 'unknown'
            
            sources.append(source_info)
        return sources

    @traceable(
        name="RAG_Generate_Answer",
        metadata={"component": "generator", "version": "2.0"}
    )
    def generate_answer(
        self, 
        query: str, 
        top_k: int = None,
        search_mode: str = None,
        alpha: float = None
    ) -> dict:
        """
        답변 생성 (Chain 기반)
        
        Args:
            query: 질문
            top_k: 검색할 문서 수
            search_mode: 검색 모드 ("embedding", "hybrid", "hybrid_rerank")
            alpha: 임베딩 가중치 (0~1)
        
        Returns:
            dict: answer, sources, search_mode, usage
        """
        try:
            start_time = time.time()

            classification = self.router.classify(query)
            query_type = classification.get('type', 'document')

            # 비문서 질의는 즉시 응답
            if query_type != 'document':
                print(f"⏭️  라우터: 검색 생략 ({query_type})")
                answer = self._direct_responses.get(
                    query_type,
                    self._direct_responses['out_of_scope']
                )
                elapsed_time = time.time() - start_time
                self._last_retrieved_docs = []

                self.chat_history.append({"role": "user", "content": query})
                self.chat_history.append({"role": "assistant", "content": answer})

                return {
                    'answer': answer,
                    'sources': [],
                    'search_mode': 'none',
                    'elapsed_time': elapsed_time,
                    'usage': {
                        'total_tokens': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0
                    },
                    'routing': classification
                }

            # 파라미터 설정
            if top_k is not None:
                self.top_k = top_k
            if search_mode is not None:
                self.search_mode = search_mode
            if alpha is not None:
                self.alpha = alpha
            
            # Chain 실행
            answer = self.chain.invoke(query)
            
            # 검색 결과가 없으면 안전 응답으로 대체
            if not self._last_retrieved_docs:
                answer = "문서에서 관련 정보를 찾을 수 없습니다. 다른 질문을 입력해 주세요."
                print("⚠️  검색 결과 없음 - 안전 응답 반환")
            
            elapsed_time = time.time() - start_time
            
            # 대화 히스토리에 추가
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # 토큰 사용량 추정 (LangChain에서는 직접 접근 어려움)
            estimated_tokens = len(query.split()) + len(answer.split()) * 2
            
            return {
                'answer': answer,
                'sources': self._format_sources(self._last_retrieved_docs),
                'search_mode': self.search_mode,
                'elapsed_time': elapsed_time,
                'usage': {
                    'total_tokens': estimated_tokens,
                    'prompt_tokens': 0,
                    'completion_tokens': 0
                },
                'routing': classification
            }
        
        except Exception as e:
            print(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e

    def chat(self, query: str) -> str:
        """
        간단한 대화 인터페이스
        
        Args:
            query: 질문
        
        Returns:
            str: 답변 텍스트만 반환
        """
        result = self.generate_answer(query)
        return result['answer']

    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        print("🗑️ 대화 히스토리가 초기화되었습니다.")

    def get_history(self) -> List[Dict]:
        """대화 히스토리 반환"""
        return self.chat_history.copy()

    def set_search_config(self, search_mode: str = None, top_k: int = None, alpha: float = None):
        """검색 설정 변경"""
        if search_mode is not None:
            self.search_mode = search_mode
        if top_k is not None:
            self.top_k = top_k
        if alpha is not None:
            self.alpha = alpha
        
        print(f"🔧 검색 설정 변경: mode={self.search_mode}, top_k={self.top_k}, alpha={self.alpha}")

    def print_result(self, result: dict, query: str = None):
        """결과 출력"""
        print("\n" + "="*60)
        if query:
            print(f"질문: {query}")
        print(f"검색 모드: {result.get('search_mode', 'N/A')}")
        if 'elapsed_time' in result:
            print(f"소요 시간: {result['elapsed_time']:.2f}초")
        print("="*60)
        print(f"\n💬 답변:\n{result['answer']}")
        print(f"\n📚 참고 문서 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            score = source.get('score', 0)
            score_type = source.get('score_type', '')
            print(f"  [{i}] {source['filename']}")
            print(f"      점수: {score:.3f} ({score_type})")
        print("="*60)


# 대화형 실행
def interactive_mode():
    """대화형 모드 실행"""
    print("=" * 60)
    print("대화형 RAG 시스템 초기화 중...")
    print("=" * 60)
    
    config = RAGConfig()
    pipeline = RAGPipeline(config=config)
    
    print("\n" + "=" * 60)
    print("대화형 모드 시작")
    print("명령어: 'quit' (종료), 'clear' (히스토리 초기화), 'mode' (검색모드 변경)")
    print("=" * 60)
    
    while True:
        user_query = input("\n질문: ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() in ['quit', 'exit', '종료', 'q']:
            print("시스템을 종료합니다.")
            break
        
        if user_query.lower() == 'clear':
            pipeline.clear_history()
            continue
        
        if user_query.lower() == 'mode':
            print("\n검색 모드 선택:")
            print("1. embedding - 임베딩 검색")
            print("2. hybrid - BM25 + 임베딩")
            print("3. hybrid_rerank - Hybrid + Re-ranker (권장)")
            choice = input("선택 (1/2/3): ").strip()
            modes = {'1': 'embedding', '2': 'hybrid', '3': 'hybrid_rerank'}
            if choice in modes:
                pipeline.set_search_config(search_mode=modes[choice])
            continue
        
        try:
            result = pipeline.generate_answer(query=user_query)
            pipeline.print_result(result, user_query)
            
            # 소스 출력 여부
            show_source = input("\n참조 문서 상세 보기? (y/n): ").strip().lower()
            if show_source == 'y':
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{'='*40}")
                    print(f"[문서 {i}] {source['filename']}")
                    print(f"발주기관: {source['organization']}")
                    print(f"내용:\n{source['content'][:500]}...")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


# 사용 예시
if __name__ == "__main__":
    interactive_mode()
