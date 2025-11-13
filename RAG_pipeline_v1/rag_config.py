import os
# from google.colab import userdata   # Colab 환경에서 사용자 데이터 접근


class RAGConfig:
    """RAG 설정 관리 클래스"""

    def __init__(self):
        # ===== 경로 설정 =====
        self.RAG_INPUT_PATH = "./data/rag_chunks_final.csv"
        self.DB_DIRECTORY = "./chroma_db"

        # ===== 모델 설정 =====
        self.EMBEDDING_MODEL_NAME = "text-embedding-3-small"
        self.LLM_MODEL_NAME = "gpt-4o-mini"

        # ===== API 키 =====
        self.OPENAI_API_KEY = self._get_api_key()

        # ===== 데이터 검증 기준 =====
        self.MIN_CHUNK_LENGTH = 10
        self.MAX_CHUNK_LENGTH = 10000

        # ===== 배치 설정 =====
        self.BATCH_SIZE = 50
        self.MAX_TOKENS_PER_BATCH = 250000

        # ===== 벡터 DB 설정 =====
        self.COLLECTION_NAME = "rag_documents"

        # ===== 검색 설정 =====
        self.DEFAULT_TOP_K = 5

        # ===== LLM 설정 =====
        self.DEFAULT_TEMPERATURE = 0.0
        self.DEFAULT_MAX_TOKENS = 1000

    # 코랩용 코드
    # def _get_api_key(self) -> str:
    #     """환경에 따라 API 키 자동 선택"""
    #     try:
    #         return userdata.get('codeit_openai_api_key')  # Colab
    #     except:
    #         return os.getenv("OPENAI_API_KEY")  # Local

    def _get_api_key(self) -> str:
        """환경에 따라 API 키 자동 선택"""
        try:
            return os.getenv("OPENAI_API_KEY")  # Local
        except:
            return print("OPENAI_API_KEY not found in environment variables.")

    def validate(self):
        """설정 유효성 검사"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
        
        if not os.path.exists(self.RAG_INPUT_PATH):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {self.RAG_INPUT_PATH}")
        
        return True

    def __repr__(self):
        """설정 정보 출력"""
        return f"""
RAGConfig:
  - 입력 파일: {self.RAG_INPUT_PATH}
  - DB 경로: {self.DB_DIRECTORY}
  - 임베딩 모델: {self.EMBEDDING_MODEL_NAME}
  - LLM 모델: {self.LLM_MODEL_NAME}
  - Top-K: {self.DEFAULT_TOP_K}
"""