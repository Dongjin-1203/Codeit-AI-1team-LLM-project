from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
import time
from typing import List, Dict

from src.utils.config import RAGConfig
from src.retriever.rag_retriever import RAGRetriever


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

        # Retriever 초기화
        self.retriever = RAGRetriever(config=self.config)
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        # 마지막 검색 결과 저장 (sources 반환용)
        self._last_retrieved_docs = []

        # 프롬프트 템플릿 (대화 히스토리 포함)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 공공입찰 RFP(제안요청서)를 분석하는 
B2G 입찰 컨설팅 스타트업 '입찰메이트'의 사내 분석가입니다.

당신의 목표는, 제공된 RFP 문서(context)만을 사용하여
컨설턴트가 빠르게 의사결정을 내릴 수 있도록
핵심 요구사항·예산·대상 기관·제출 방식 등을 정확하고 구조적으로 정리하는 것입니다.

# 답변 시 일반 규칙
1. 반드시 **한국어**로 답변합니다.
2. 외부 지식이나 일반 상식으로 추측하지 말고,
   참조 문서 컨텍스트에 명시된 내용만 사용합니다.
3. 문서에 정보가 없거나 불충분하면,
   반드시 "문서에서 해당 정보를 찾을 수 없습니다."라고 명시하고
   추측이나 가정을 덧붙이지 않습니다.
4. 질문이 여러 문서를 비교하거나 여러 기관의 사례를 묻는 경우,
   문서별로 차이점을 표나 목록 형태로 정리합니다.
5. 숫자(예산, 인원, 기간 등)가 등장하면 
   가능한 한 단위(억 원, 명, 개월 등)까지 함께 명시합니다.
6. 이전 대화 내용을 참고하여 맥락에 맞는 답변을 제공합니다.

# 답변 형식
다음 형식을 최대한 지켜서 답변하세요.

1. 한 줄 요약
- 사용자의 질문에 대한 핵심 답을 한 두 문장으로 요약합니다.

2. 상세 답변
- 질문에 직접적으로 답이 되는 내용을 항목별로 정리합니다.
- [요구사항], [대상 기관], [예산], [제출 형식/방법], [평가 기준] 등
    문서에서 확인되는 항목을 중심으로 적절히 나누어 설명합니다.

3. 근거 정보
- 위에서 작성한 답변이 어떤 문장/문단에서 나왔는지 요약해서 서술합니다.

4. 부족한 정보
- 질문에서 요구한 정보 중, 문서에서 찾을 수 없는 것이 있다면
    항목별로 "문서에서 확인 불가"라고 명시합니다."""),
            
            # 대화 히스토리
            MessagesPlaceholder(variable_name="chat_history"),
            
            # 현재 질문과 컨텍스트
            ("user", """# 참조 문서 컨텍스트
아래 내용은 검색된 RFP 문서 조각들입니다.
여기에 포함된 내용만 사용해서 답변해야 합니다.

{context}

# 질문
{question}

이제 위 규칙을 모두 고려하여, 질문에 대한 최선의 답변을 작성하세요.""")
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
            
            # 파라미터 설정
            if top_k is not None:
                self.top_k = top_k
            if search_mode is not None:
                self.search_mode = search_mode
            if alpha is not None:
                self.alpha = alpha
            
            # Chain 실행
            answer = self.chain.invoke(query)
            
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
                }
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