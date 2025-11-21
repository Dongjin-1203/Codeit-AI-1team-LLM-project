from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith import traceable
import time
import os
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

from src.utils.config import RAGConfig


class RAGRetriever:
    """RAG 검색 시스템 (Hybrid Search + Re-ranker)"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vectorstore = None
        self.embeddings = None

        self._initialize_embeddings()
        self._create_vectorstore()
        self._initialize_bm25()
        self._initialize_reranker()

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

    def _initialize_bm25(self):
        """BM25 인덱스 생성"""
        all_docs = self.vectorstore.get()
        
        self.doc_texts = all_docs['documents']
        self.doc_ids = all_docs['ids']
        self.doc_metadatas = all_docs['metadatas']
        
        self.content_to_id = {text: doc_id for text, doc_id in zip(self.doc_texts, self.doc_ids)}
        
        tokenized_docs = [doc.split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✅ BM25 인덱스 생성 완료: {len(self.doc_texts)}개 문서")

    def _initialize_reranker(self):
        """Re-ranker 초기화"""
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        print("✅ Re-ranker 초기화 완료 (bge-reranker-base)")

    @staticmethod
    def _min_max_normalize(scores):
        """0~1 범위로 정규화"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.full_like(scores, 0.5, dtype=float)
        
        return (scores - min_score) / (max_score - min_score)

    def _find_doc_id_by_content(self, content):
        """문서 content로 ID 찾기"""
        return self.content_to_id.get(content, None)

    def _rerank(self, query, documents, top_k):
        """
        검색 결과 재정렬
        
        Args:
            query: 검색 쿼리
            documents: hybrid_search 결과 리스트
            top_k: 최종 반환할 문서 수
        
        Returns:
            재정렬된 상위 k개 문서
        """
        if len(documents) == 0:
            return []
        
        # 1. (query, document) 쌍 생성
        pairs = [[query, doc['content']] for doc in documents]
        
        # 2. CrossEncoder로 점수 계산
        scores = self.reranker.predict(pairs)
        
        # 3. 점수를 문서에 추가
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # 4. 정렬 및 반환
        sorted_docs = sorted(documents, 
                            key=lambda x: x['rerank_score'], 
                            reverse=True)
        
        return sorted_docs[:top_k]

    @traceable(
        name="RAG_Hybrid_Search",
        metadata={"component": "retriever", "version": "2.0"}
    )
    def hybrid_search(self, query, top_k=None, alpha=0.5):
        """
        Hybrid Search: BM25 + 임베딩 결합
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            alpha: 임베딩 가중치 (0~1)
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        # 1. BM25 검색
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_normalized = self._min_max_normalize(bm25_scores)
        
        # 2. 임베딩 검색
        embedding_results = self.vectorstore.similarity_search_with_score(
            query, k=min(top_k * 3, len(self.doc_texts))
        )
        
        # 3. 임베딩 점수 정규화
        embedding_scores_raw = {}
        for doc, distance in embedding_results:
            doc_id = self._find_doc_id_by_content(doc.page_content)
            if doc_id:
                embedding_scores_raw[doc_id] = 1 / (1 + distance)
        
        if embedding_scores_raw:
            embed_values = np.array(list(embedding_scores_raw.values()))
            embed_normalized = self._min_max_normalize(embed_values)
            embedding_scores = dict(zip(embedding_scores_raw.keys(), embed_normalized))
        else:
            embedding_scores = {}
        
        # 4. 하이브리드 점수 계산
        hybrid_scores = {}
        for i, doc_id in enumerate(self.doc_ids):
            bm25_score = bm25_normalized[i]
            embed_score = embedding_scores.get(doc_id, 0)
            hybrid_scores[doc_id] = (1 - alpha) * bm25_score + alpha * embed_score
        
        # 5. 정렬 및 상위 k개 선택
        sorted_ids = sorted(hybrid_scores.keys(), 
                           key=lambda x: hybrid_scores[x], 
                           reverse=True)
        top_ids = sorted_ids[:top_k]
        
        # 6. 결과 포맷팅
        formatted_results = []
        for doc_id in top_ids:
            idx = self.doc_ids.index(doc_id)
            formatted_results.append({
                'content': self.doc_texts[idx],
                'metadata': self.doc_metadatas[idx],
                'hybrid_score': hybrid_scores[doc_id],
                'bm25_score': float(bm25_normalized[idx]),
                'embed_score': embedding_scores.get(doc_id, 0),
                'filename': self.doc_metadatas[idx].get('파일명', 'N/A'),
                'organization': self.doc_metadatas[idx].get('발주 기관', 'N/A')
            })
        
        end_time = time.time()
        print(f"🔍 Hybrid 검색 완료: {len(formatted_results)}개 (alpha={alpha}, {end_time-start_time:.3f}초)")
        return formatted_results

    @traceable(
        name="RAG_Hybrid_Search_Rerank",
        metadata={"component": "retriever", "version": "3.0"}
    )
    def hybrid_search_with_rerank(self, query, top_k=None, alpha=0.5, rerank_candidates=None):
        """
        Hybrid Search + Re-ranking
        
        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 문서 수
            alpha: BM25/임베딩 가중치
            rerank_candidates: Re-rank할 후보 수 (None이면 top_k * 3)
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        if rerank_candidates is None:
            rerank_candidates = top_k * 3
        
        # 1. Hybrid Search로 후보 문서 가져오기
        candidates = self.hybrid_search(query, top_k=rerank_candidates, alpha=alpha)
        
        # 2. Re-ranking
        if len(candidates) > 0:
            results = self._rerank(query, candidates, top_k)
        else:
            results = []
        
        end_time = time.time()
        print(f"🔄 Re-ranking 완료: {len(candidates)}개 → {len(results)}개 ({end_time-start_time:.3f}초)")
        
        return results

    def search_with_mode(self, query, top_k=None, mode="hybrid_rerank", alpha=0.5):
        """검색 모드 선택"""
        if mode == "embedding":
            return self.search(query, top_k)
        elif mode == "bm25":
            return self.hybrid_search(query, top_k, alpha=0.0)
        elif mode == "hybrid":
            return self.hybrid_search(query, top_k, alpha=alpha)
        elif mode == "hybrid_rerank":
            return self.hybrid_search_with_rerank(query, top_k, alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @traceable(
        name="RAG_Retriever_Search",
        metadata={"component": "retriever", "version": "1.0"}
    )
    def search(self, query: str, top_k: int = None, filter_metadata: dict = None):
        """
        유사 문서 검색 (임베딩 기반)
        """
        start_time = time.time()
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K

        if filter_metadata:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k, filter=filter_metadata
            )
        else:
            results = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )

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

        end_time = time.time()
        print(f"🔍 검색 완료: {len(results)}개 ({end_time-start_time:.3f}초)")
        return formatted_results

    def search_with_rerank(self, query, top_k=None, rerank_candidates=None):
        """
        임베딩 검색 + Re-ranking
        
        Args:
            query: 검색 쿼리
            top_k: 최종 반환할 문서 수
            rerank_candidates: Re-rank할 후보 수
        
        Returns:
            재정렬된 문서 리스트
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = self.config.DEFAULT_TOP_K
        
        if rerank_candidates is None:
            rerank_candidates = top_k * 3
        
        # 1. 임베딩 검색으로 후보 가져오기
        candidates = self.search(query, top_k=rerank_candidates)
        
        # 2. Re-ranking
        if len(candidates) > 0:
            results = self._rerank(query, candidates, top_k)
        else:
            results = []
        
        end_time = time.time()
        print(f"🔄 Embedding + Re-ranking 완료: {len(candidates)}개 → {len(results)}개 ({end_time-start_time:.3f}초)")
        
        return results

    def search_by_organization(self, query: str, organization: str, top_k: int = None):
        """특정 발주기관만 검색"""
        return self.search(
            query, top_k=top_k, filter_metadata={'발주 기관': organization}
        )

    def get_retriever(self):
        """LangChain 체인용 Retriever 반환"""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.DEFAULT_TOP_K}
        )