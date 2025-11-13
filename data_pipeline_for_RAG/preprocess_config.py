"""
RAG 전처리 설정 관리
경로, 청킹 파라미터 등 설정 중앙 관리
"""

import os


class PreprocessConfig:
    """RAG 전처리 설정 클래스"""
    
    def __init__(self):
        # ===== 경로 설정 =====
        self.META_CSV_PATH = "./data/data_list.csv"
        self.BASE_FOLDER_PATH = "./data/files/"
        self.OUTPUT_CHUNKS_PATH = "./data/rag_chunks_final.csv"
        
        # ===== 청킹 설정 =====
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.SEPARATORS = ["\n\n", "\n", " ", ""]
        
        # ===== 검증 기준 =====
        self.MIN_TEXT_LENGTH = 100  # 최소 텍스트 길이
    
    def validate(self):
        """설정 유효성 검사"""
        if not os.path.exists(self.META_CSV_PATH):
            raise FileNotFoundError(
                f"메타 CSV 파일을 찾을 수 없습니다: {self.META_CSV_PATH}"
            )
        
        if not os.path.exists(self.BASE_FOLDER_PATH):
            raise FileNotFoundError(
                f"파일 폴더를 찾을 수 없습니다: {self.BASE_FOLDER_PATH}"
            )
        
        # 출력 폴더 생성
        os.makedirs(os.path.dirname(self.OUTPUT_CHUNKS_PATH), exist_ok=True)
        
        return True
    
    def __repr__(self):
        """설정 정보 출력"""
        return f"""
PreprocessConfig:
  - 메타 CSV: {self.META_CSV_PATH}
  - 파일 폴더: {self.BASE_FOLDER_PATH}
  - 출력 경로: {self.OUTPUT_CHUNKS_PATH}
  - 청크 크기: {self.CHUNK_SIZE}
  - 청크 오버랩: {self.CHUNK_OVERLAP}
"""