"""
RAG 데이터 전처리 전체 파이프라인
텍스트 추출 → 정제 → 청킹 → 저장
"""

import os
import pandas as pd
from tqdm import tqdm

from preprocess_config import PreprocessConfig
from text_extractor import TextExtractor
from text_cleaner import TextCleaner
from document_chunker import DocumentChunker


class RAGPreprocessPipeline:
    """RAG 데이터 전처리 전체 파이프라인"""
    
    def __init__(self, config: PreprocessConfig = None):
        """
        초기화
        
        Args:
            config: 전처리 설정 (None이면 기본값)
        """
        self.config = config or PreprocessConfig()
        self.extractor = TextExtractor()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker(self.config)
        
        # 통계 정보
        self.stats = {
            'total_files': 0,
            'success_files': 0,
            'failed_files': 0,
            'total_chunks': 0
        }
    
    def extract_from_files(self) -> pd.DataFrame:
        """
        1단계: 파일에서 텍스트 추출
        
        Returns:
            텍스트가 추출된 DataFrame
        """
        print("\n" + "="*60)
        print("1단계: 텍스트 추출")
        print("="*60)
        
        # 메타데이터 로드
        df = pd.read_csv(self.config.META_CSV_PATH)
        self.stats['total_files'] = len(df)
        print(f"파일 로드 완료: {len(df)}개")
        
        extracted_data = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="텍스트 추출"):
            filepath = os.path.join(self.config.BASE_FOLDER_PATH, row['파일명'])
            file_format = row['파일형식']
            
            # 텍스트 추출
            raw_text = self.extractor.extract(filepath, file_format)
            
            # 정제
            cleaned_text = self.cleaner.clean(raw_text)
            
            # HWP 특수 처리 (텍스트가 너무 짧으면 실패로 간주)
            if file_format == 'hwp' and len(cleaned_text) < self.config.MIN_TEXT_LENGTH:
                if "[추출 실패" not in cleaned_text:
                    cleaned_text = "[추출 실패: HWP 텍스트 너무 짧음]"
            
            # 통계 업데이트
            if self.cleaner.validate(cleaned_text):
                self.stats['success_files'] += 1
            else:
                self.stats['failed_files'] += 1
            
            # 결과 저장
            new_row = row.to_dict()
            new_row['full_text'] = cleaned_text
            
            # 불필요한 컬럼 제거
            if '텍스트' in new_row:
                del new_row['텍스트']
            
            extracted_data.append(new_row)
        
        result_df = pd.DataFrame(extracted_data)
        
        print(f"\n텍스트 추출 완료:")
        print(f"  - 성공: {self.stats['success_files']}개")
        print(f"  - 실패: {self.stats['failed_files']}개")
        
        return result_df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        2단계: DataFrame 정제
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            정제된 DataFrame
        """
        print("\n" + "="*60)
        print("2단계: 텍스트 정제")
        print("="*60)
        
        # 컬럼명 변경
        df['text_content'] = df['full_text']
        df = df.drop(columns=['full_text'])
        
        # 결측치 처리
        df['text_content'] = df['text_content'].fillna('')
        
        # 통계 정보 추가
        df['text_length'] = df['text_content'].apply(len)
        
        print(f"텍스트 정제 완료")
        print(f"  - 평균 길이: {df['text_length'].mean():.0f} 문자")
        print(f"  - 최소 길이: {df['text_length'].min()} 문자")
        print(f"  - 최대 길이: {df['text_length'].max()} 문자")
        
        return df
    
    def create_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        3단계: 청킹
        
        Args:
            df: 정제된 DataFrame
            
        Returns:
            청크 DataFrame
        """
        print("\n" + "="*60)
        print("3단계: 청킹")
        print("="*60)
        
        df_chunks = self.chunker.chunk_dataframe(df)
        self.stats['total_chunks'] = len(df_chunks)
        
        return df_chunks
    
    def save_chunks(self, df_chunks: pd.DataFrame):
        """
        4단계: 결과 저장
        
        Args:
            df_chunks: 청크 DataFrame
        """
        print("\n" + "="*60)
        print("4단계: 결과 저장")
        print("="*60)
        
        df_chunks.to_csv(
            self.config.OUTPUT_CHUNKS_PATH, 
            index=False, 
            encoding='utf-8-sig'
        )
        
        print(f"최종 청크 저장 완료: {self.config.OUTPUT_CHUNKS_PATH}")
        print(f"총 청크 수: {len(df_chunks)}")
    
    def run(self) -> pd.DataFrame:
        """
        전체 파이프라인 실행
        
        Returns:
            최종 청크 DataFrame
        """
        print("="*60)
        print("RAG 전처리 파이프라인 시작")
        print("="*60)
        
        # 설정 검증
        self.config.validate()
        print(self.config)
        
        # 1. 텍스트 추출
        df_extracted = self.extract_from_files()
        
        # 2. 텍스트 정제
        df_cleaned = self.clean_dataframe(df_extracted)
        
        # 3. 청킹
        df_chunks = self.create_chunks(df_cleaned)
        
        # 4. 저장
        self.save_chunks(df_chunks)
        
        # 최종 통계
        self._print_final_stats()
        
        print("\n" + "="*60)
        print("✅ RAG 전처리 파이프라인 완료")
        print("="*60)
        
        return df_chunks
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "="*60)
        print("📊 최종 통계")
        print("="*60)
        print(f"총 파일 수: {self.stats['total_files']}")
        
        if self.stats['total_files'] > 0:
            success_rate = self.stats['success_files'] / self.stats['total_files'] * 100
            fail_rate = self.stats['failed_files'] / self.stats['total_files'] * 100
            
            print(f"  - 추출 성공: {self.stats['success_files']} ({success_rate:.1f}%)")
            print(f"  - 추출 실패: {self.stats['failed_files']} ({fail_rate:.1f}%)")
        
        print(f"총 청크 수: {self.stats['total_chunks']}")
        
        if self.stats['success_files'] > 0:
            avg_chunks = self.stats['total_chunks'] / self.stats['success_files']
            print(f"파일당 평균 청크: {avg_chunks:.1f}개")