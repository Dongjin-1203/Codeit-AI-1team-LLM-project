import pandas as pd
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from tqdm import tqdm
import time

from rag_config import RAGConfig


class DataValidator:
    """데이터 검증 및 정제"""

    def __init__(self, config: RAGConfig):
        self.config = config

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 검증 및 정제 파이프라인"""
        df = self._check_required_columns(df)
        df = self._remove_duplicates(df)
        df = self._remove_nan(df)
        df = self._filter_by_length(df)
        df = self._clean_metadata(df)

        return df

    def _check_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """필수 컬럼 확인"""
        required = ['chunk_content', 'chunk_id']
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """중복 ID 제거"""
        return df.drop_duplicates(subset=['chunk_id'], keep='first')

    def _remove_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN 값 제거"""
        return df.dropna(subset=['chunk_content', 'chunk_id'])

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """길이 기준 필터링"""
        df['_temp_length'] = df['chunk_content'].str.len()

        df = df[
            (df['_temp_length'] >= self.config.MIN_CHUNK_LENGTH) &
            (df['_temp_length'] <= self.config.MAX_CHUNK_LENGTH)
        ]

        return df.drop(columns=['_temp_length'])

    def _clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """메타데이터 정제"""
        # NaN을 빈 문자열로 변환
        df = df.fillna('')

        # 메타데이터 컬럼의 타입을 문자열로 변환
        metadata_cols = [col for col in df.columns
                        if col not in ['chunk_content', 'chunk_id']]

        for col in metadata_cols:
            df[col] = df[col].astype(str)

        return df


class ChromaDBBuilder:
    """ChromaDB 벡터 데이터베이스 구축"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vectorstore = None
        self.embeddings = None

        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY

        self.embeddings = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL_NAME
        )

    def build_from_dataframe(self, df: pd.DataFrame):
        """DataFrame으로부터 벡터 DB 구축"""
        documents, ids, metadatas = self._prepare_data(df)
        self._validate_data_consistency(documents, ids, metadatas)
        self._create_vectorstore()
        self._add_documents_in_batches(documents, ids, metadatas)

        return self.vectorstore

    def _prepare_data(self, df: pd.DataFrame):
        """ChromaDB용 데이터 준비"""
        documents = df['chunk_content'].tolist()
        ids = df['chunk_id'].tolist()

        # 메타데이터 추출
        metadata_cols = [col for col in df.columns
                        if col not in ['chunk_content', 'chunk_id']]

        metadatas = []
        for _, row in df.iterrows():
            metadata = {
                col: row[col]
                for col in metadata_cols
                if row[col] and row[col] != 'nan' and row[col] != ''
            }
            metadatas.append(metadata)

        return documents, ids, metadatas

    def _validate_data_consistency(self, documents, ids, metadatas):
        """데이터 일관성 검증"""
        if not (len(documents) == len(ids) == len(metadatas)):
            raise ValueError("데이터 길이 불일치")

    def _create_vectorstore(self):
        """빈 벡터스토어 생성"""
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.config.DB_DIRECTORY,
            collection_name=self.config.COLLECTION_NAME
        )

    def _add_documents_in_batches(self, documents, ids, metadatas):
        """배치 처리로 문서 추가"""
        batch_size = self.config.BATCH_SIZE
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(documents), batch_size),
                     desc="임베딩 및 저장",
                     total=total_batches):

            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]

            self._add_batch_with_retry(batch_docs, batch_ids, batch_metas)
            time.sleep(1)

    def _add_batch_with_retry(self, docs, ids, metas):
        """배치 추가 (실패 시 재시도)"""
        batch_tokens = sum(len(doc) for doc in docs) / 4

        if batch_tokens > self.config.MAX_TOKENS_PER_BATCH:
            smaller_size = len(docs) // 2
            for j in range(0, len(docs), smaller_size):
                self.vectorstore.add_texts(
                    texts=docs[j:j + smaller_size],
                    metadatas=metas[j:j + smaller_size],
                    ids=ids[j:j + smaller_size]
                )
                time.sleep(0.5)
        else:
            try:
                self.vectorstore.add_texts(
                    texts=docs,
                    metadatas=metas,
                    ids=ids
                )
            except Exception as e:
                for j in range(0, len(docs), 10):
                    self.vectorstore.add_texts(
                        texts=docs[j:j + 10],
                        metadatas=metas[j:j + 10],
                        ids=ids[j:j + 10]
                    )
                    time.sleep(0.5)

    def get_collection_count(self):
        """저장된 문서 수 반환"""
        if self.vectorstore:
            return self.vectorstore._collection.count()
        return 0

    def search(self, query: str, k: int = 5):
        """검색 수행"""
        if not self.vectorstore:
            raise ValueError("벡터스토어가 초기화되지 않았습니다")

        return self.vectorstore.similarity_search_with_score(query, k=k)


class RAGVectorDBPipeline:
    """전체 RAG Vector DB 구축 파이프라인"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.validator = DataValidator(self.config)
        self.builder = ChromaDBBuilder(self.config)

    def build(self):
        """전체 파이프라인 실행"""
        # 데이터 로드
        df = pd.read_csv(self.config.RAG_INPUT_PATH)
        print(f"원본 데이터: {len(df)}개 청크")

        # 데이터 검증 및 정제
        df_cleaned = self.validator.validate_and_clean(df)
        print(f"정제 후 데이터: {len(df_cleaned)}개 청크")

        # 벡터 DB 구축
        vectorstore = self.builder.build_from_dataframe(df_cleaned)

        # 결과 확인
        count = self.builder.get_collection_count()
        print(f"✅ ChromaDB 저장 완료: {count}개 문서")
        print(f"저장 위치: {self.config.DB_DIRECTORY}")

        return vectorstore

    def test_search(self, query: str = "학사 정보 시스템", k: int = 3):
        """검색 테스트"""
        results = self.builder.search(query, k=k)

        print(f"\n테스트 쿼리: '{query}'")
        print(f"검색 결과: {len(results)}개\n")

        for i, (doc, score) in enumerate(results, 1):
            print(f"[{i}] 거리: {score:.4f}")
            print(f"내용: {doc.page_content[:100]}...")
            print(f"메타데이터: {doc.metadata}\n")

        return results