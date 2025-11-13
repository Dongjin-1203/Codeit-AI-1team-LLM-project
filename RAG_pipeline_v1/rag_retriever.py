from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os

from rag_config import RAGConfig


class RAGRetriever:
    """RAG 검색 시스템"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vectorstore = None
        self.embeddings = None

        self._initialize_embeddings()
        self._create_vectorstore()

    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY

        self.embeddings = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL_NAME
        )

    def _create_vectorstore(self):
        """기존 벡터스토어 로드"""
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.config.DB_DIRECTORY,
            collection_name=self.config.COLLECTION_NAME
        )

    def search(self, query: str, top_k: int = None, filter_metadata: dict = None):
        """
        유사 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수 (None이면 config 기본값)
            filter_metadata: 메타데이터 필터
            
        Returns:
            검색 결과 리스트
        """
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K

        # 검색 수행
        if filter_metadata:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=top_k
            )

        # 결과 포맷팅
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'distance': score,
                'relevance_score': 1 - score,
                'filename': doc.metadata.get('파일명', 'N/A'),
                'organization': doc.metadata.get('발주 기관', 'N/A')
            })

        return formatted_results

    def search_by_organization(self, query: str, organization: str, top_k: int = None):
        """특정 발주기관만 검색"""
        return self.search(
            query,
            top_k=top_k,
            filter_metadata={'발주 기관': organization}
        )

    def get_retriever(self):
        """LangChain 체인용 Retriever 반환"""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.DEFAULT_TOP_K}
        )