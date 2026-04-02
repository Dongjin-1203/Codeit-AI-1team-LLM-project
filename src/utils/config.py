import os
from dotenv import load_dotenv


class Config:
    """RAG 시스템 통합 설정 클래스"""

    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # ===== API 키 =====
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        
        # ===== 경로 설정 =====
        # 전처리
        self.META_CSV_PATH = "./data/data_list.csv"
        self.BASE_FOLDER_PATH = "./data/files/"
        self.OUTPUT_CHUNKS_PATH = "./data/rag_chunks_final.csv"
        
        # RAG - 환경변수 우선, 없으면 기본값
        self.RAG_INPUT_PATH = "./data/rag_chunks_final.csv"
        self.DB_DIRECTORY = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # ===== 전처리 설정 =====
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.SEPARATORS = ["\n\n", "\n", " ", ""]
        self.MIN_TEXT_LENGTH = 100
        
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
        self.DEFAULT_ALPHA = 0.5
        self.DEFAULT_SEARCH_MODE = "hybrid_rerank"
        
        # ===== LLM 설정 =====
        self.LLM_MODEL_NAME = "gpt-4o-mini"
        self.DEFAULT_TEMPERATURE = 0.0
        self.DEFAULT_MAX_TOKENS = 1000
        
        # 시스템 프롬프트
        self.SYSTEM_PROMPT = "당신은 RFP(제안요청서) 분석 및 요약 전문가입니다."
        
        # ===== GGUF 로컬 모델 설정 =====
        # Model Hub 사용 여부 (환경변수 우선)
        self.USE_MODEL_HUB = os.getenv("USE_MODEL_HUB", "true").lower() == "true"
        
        # Hugging Face Model Hub 설정
        # Llama-3-Open-Ko-8B 한국어 GGUF 모델 사용
        self.MODEL_HUB_REPO = os.getenv(
            "MODEL_HUB_REPO", 
            "Dongjin1203/RFP_Documents_chatbot"
        )
        self.MODEL_HUB_FILENAME = os.getenv(
            "MODEL_HUB_FILENAME", 
            "Llama-3-Open-Ko-8B.Q4_K_M.gguf"
        )
        self.MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", ".cache/models")
        
        # 로컬 경로 (USE_MODEL_HUB=false인 경우)
        self.GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", ".cache/models/Llama-3-Open-Ko-8B.Q4_K_M.gguf")
        
        # GGUF GPU 설정 (T4 Medium 최적화 - 8B 모델용)
        self.GGUF_N_GPU_LAYERS = int(os.getenv("GGUF_N_GPU_LAYERS", "35"))  # T4에서 8B 모델 전체를 GPU에 로드
        self.GGUF_N_CTX = int(os.getenv("GGUF_N_CTX", "2048"))              # 컨텍스트 길이
        self.GGUF_N_THREADS = int(os.getenv("GGUF_N_THREADS", "4"))         # CPU 스레드 (GPU 사용 시 낮게)
        self.GGUF_MAX_NEW_TOKENS = int(os.getenv("GGUF_MAX_NEW_TOKENS", "512"))  # 최대 생성 토큰
        self.GGUF_TEMPERATURE = float(os.getenv("GGUF_TEMPERATURE", "0.7"))       # 생성 다양성
        self.GGUF_TOP_P = float(os.getenv("GGUF_TOP_P", "0.9"))                   # Nucleus sampling

    def validate_for_rag(self):
        """RAG 서빙 전 API 키 검증 — 임베딩·생성 단계 진입 전 명시적으로 호출"""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다.\n"
                "프로젝트 루트에 .env 파일을 만들고 OPENAI_API_KEY=your-key 를 추가하세요."
            )
        return True

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
        
        output_dir = os.path.dirname(self.OUTPUT_CHUNKS_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        return True

    def validate_rag(self):
        """RAG 설정 유효성 검사"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다")
        
        return True
    
    def validate_gguf(self):
        """GGUF 설정 유효성 검사"""
        if not self.USE_MODEL_HUB:
            # 로컬 파일 사용 시 경로 확인
            if not os.path.exists(self.GGUF_MODEL_PATH):
                print(f"⚠️ 경고: GGUF 모델 파일이 없습니다: {self.GGUF_MODEL_PATH}")
                print(f"   USE_MODEL_HUB=true로 설정하여 자동 다운로드하거나 모델 파일을 준비하세요.")
        
        # GPU 레이어 설정 확인
        if self.GGUF_N_GPU_LAYERS > 0:
            print(f"✅ GPU 가속 활성화: {self.GGUF_N_GPU_LAYERS}개 레이어")
        else:
            print(f"⚠️ CPU 전용 모드 (n_gpu_layers=0)")
        
        return True

    def validate_all(self):
        """전체 설정 유효성 검사"""
        self.validate_preprocess()
        self.validate_rag()
        self.validate_gguf()
        return True

    def validate(self):
        """설정 유효성 검사 (하위 호환성)"""
        return self.validate_preprocess()
    
    def print_gguf_config(self):
        """GGUF 설정 출력 (디버깅용)"""
        print("\n" + "="*50)
        print("GGUF 모델 설정")
        print("="*50)
        print(f"Model Hub 사용: {self.USE_MODEL_HUB}")
        if self.USE_MODEL_HUB:
            print(f"Hub Repo: {self.MODEL_HUB_REPO}")
            print(f"Hub 파일명: {self.MODEL_HUB_FILENAME}")
            print(f"캐시 디렉토리: {self.MODEL_CACHE_DIR}")
        else:
            print(f"로컬 경로: {self.GGUF_MODEL_PATH}")
        print(f"\nGPU 설정:")
        print(f"  - GPU 레이어: {self.GGUF_N_GPU_LAYERS}")
        print(f"  - 컨텍스트: {self.GGUF_N_CTX}")
        print(f"  - 스레드: {self.GGUF_N_THREADS}")
        print(f"\n생성 설정:")
        print(f"  - Max Tokens: {self.GGUF_MAX_NEW_TOKENS}")
        print(f"  - Temperature: {self.GGUF_TEMPERATURE}")
        print(f"  - Top-P: {self.GGUF_TOP_P}")
        print("="*50 + "\n")


# 하위 호환성을 위한 별칭
PreprocessConfig = Config
RAGConfig = Config