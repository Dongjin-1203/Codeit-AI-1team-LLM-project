import os
from dotenv import load_dotenv


class Config:
    """RAG 시스템 통합 설정 클래스"""

    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # ===== API 키 =====
        self.OPENAI_API_KEY = self._get_api_key()
        
        # ===== 경로 설정 =====
        # 전처리
        self.META_CSV_PATH = "./data/data_list.csv"
        self.BASE_FOLDER_PATH = "./data/files/"
        self.OUTPUT_CHUNKS_PATH = "./data/rag_chunks_final.csv"
        
        # RAG
        self.RAG_INPUT_PATH = "./data/rag_chunks_final.csv"
        self.DB_DIRECTORY = "./chroma_db"
        
        # ===== 전처리 설정 =====
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.SEPARATORS = ["\n\n", "\n", " ", ""]
        self.MIN_TEXT_LENGTH = 100  # 최소 텍스트 길이
        
        # ===== 임베딩 설정 =====
        self.EMBEDDING_MODEL_NAME = "text-embedding-3-small"
        self.BATCH_SIZE = 50
        self.MAX_TOKENS_PER_BATCH = 250000
        
        # 청크 검증 기준
        self.MIN_CHUNK_LENGTH = 10
        self.MAX_CHUNK_LENGTH = 10000
        
        # ===== 벡터 DB 설정 =====
        self.COLLECTION_NAME = "rag_documents"
        
        # ===== 검색 설정 =====
        self.DEFAULT_TOP_K = 10
        self.DEFAULT_ALPHA = 0.5  # Hybrid Search 가중치
        self.DEFAULT_SEARCH_MODE = "hybrid_rerank"
        
        # ===== LLM 설정 =====
        self.LLM_MODEL_NAME = "gpt-5-mini"
        self.DEFAULT_TEMPERATURE = 0.0
        self.DEFAULT_MAX_TOKENS = 1000

    def _get_api_key(self) -> str:
        """환경변수에서 API 키 로드"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "프로젝트 루트에 .env 파일을 만들고 OPENAI_API_KEY=your-key 를 추가하세요."
            )
        
        return api_key

    def validate_preprocess(self):
        """전처리 설정 유효성 검사"""
        if not os.path.exists(self.META_CSV_PATH):
            raise FileNotFoundError(
                f"메타 CSV 파일을 찾을 수 없습니다: {self.META_CSV_PATH}"
            )
        
        if not os.path.exists(self.BASE_FOLDER_PATH):
            raise FileNotFoundError(
                f"파일 폴더를 찾을 수 없습니다: {self.BASE_FOLDER_PATH}"
            )
        
        # 출력 폴더 생성
        output_dir = os.path.dirname(self.OUTPUT_CHUNKS_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        return True

    def validate_rag(self):
        """RAG 설정 유효성 검사"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
        
        if not os.path.exists(self.RAG_INPUT_PATH):
            raise FileNotFoundError(
                f"입력 파일을 찾을 수 없습니다: {self.RAG_INPUT_PATH}"
            )
        
        return True

    def validate_all(self):
        """전체 설정 유효성 검사"""
        self.validate_preprocess()
        self.validate_rag()
        return True

    def validate(self):
        """설정 유효성 검사 (하위 호환성)"""
        return self.validate_preprocess()

    def __repr__(self):
        """설정 정보 출력"""
        return f"""
Config 설정:
  [경로]
  - 메타 CSV: {self.META_CSV_PATH}
  - 파일 폴더: {self.BASE_FOLDER_PATH}
  - 청크 출력: {self.OUTPUT_CHUNKS_PATH}
  - DB 경로: {self.DB_DIRECTORY}
  
  [전처리]
  - 청크 크기: {self.CHUNK_SIZE}
  - 청크 오버랩: {self.CHUNK_OVERLAP}
  
  [모델]
  - 임베딩: {self.EMBEDDING_MODEL_NAME}
  - LLM: {self.LLM_MODEL_NAME}
  
  [검색]
  - Top-K: {self.DEFAULT_TOP_K}
  - Alpha: {self.DEFAULT_ALPHA}
  - 모드: {self.DEFAULT_SEARCH_MODE}
"""


# 하위 호환성을 위한 별칭
PreprocessConfig = Config
RAGConfig = Config


# 테스트용
if __name__ == "__main__":
    config = Config()
    print(config)