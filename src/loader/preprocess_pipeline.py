"""
RAG 데이터 전처리 전체 파이프라인
텍스트 추출 → 정제 → 청킹 → 저장

모든 전처리 클래스를 하나의 파일로 통합
"""

import os
import re
import zlib
import struct
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
import olefile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import PreprocessConfig


# ============================================================
# 텍스트 추출 클래스
# ============================================================

class TextExtractor:
    """PDF 및 HWP 파일에서 텍스트 추출"""
    
    @staticmethod
    def extract_pdf(filepath: str) -> str:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            filepath: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            reader = PdfReader(filepath)
            page_texts = [
                page.extract_text() 
                for page in reader.pages 
                if page.extract_text()
            ]
            return "\n\n".join(page_texts)
        except Exception as e:
            return f"[PDF 추출 실패: {e}]"
    
    @staticmethod
    def extract_hwp(filepath: str) -> str:
        """
        HWP 파일에서 텍스트 추출
        
        Args:
            filepath: HWP 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            f = olefile.OleFileIO(filepath)
            dirs = f.listdir()
            
            # HWP 5.0 파일 검증
            if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
                return "[HWP 추출 실패: 유효한 HWP 5.0 파일이 아님]"
            
            # 압축 여부 확인
            header = f.openstream("FileHeader")
            header_data = header.read()
            is_compressed = (header_data[36] & 1) == 1
            
            # 섹션 번호 정렬
            nums = [
                int(d[1][len("Section"):]) 
                for d in dirs 
                if d[0] == "BodyText"
            ]
            sections = [f"BodyText/Section{x}" for x in sorted(nums)]
            
            # 텍스트 추출
            text = ""
            for section in sections:
                bodytext = f.openstream(section)
                data = bodytext.read()
                
                # 압축 해제
                if is_compressed:
                    unpacked_data = zlib.decompress(data, -15)
                else:
                    unpacked_data = data
                
                # 레코드 파싱
                i = 0
                size = len(unpacked_data)
                while i < size:
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    rec_len = (header >> 20) & 0xfff
                    
                    # 텍스트 레코드 (타입 67)
                    if rec_type == 67:
                        rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                        text += rec_data.decode('utf-16', errors='ignore')
                    
                    i += 4 + rec_len
            
            f.close()
            return text
            
        except Exception as e:
            return f"[HWP 추출 실패: {e}]"
    
    @staticmethod
    def extract(filepath: str, file_format: str) -> str:
        """
        파일 형식에 따라 텍스트 추출
        
        Args:
            filepath: 파일 경로
            file_format: 파일 형식 ('pdf' 또는 'hwp')
            
        Returns:
            추출된 텍스트
        """
        if not os.path.exists(filepath):
            return "[추출 실패: 파일 없음]"
        
        file_format = file_format.lower()
        
        if file_format == 'pdf':
            return TextExtractor.extract_pdf(filepath)
        elif file_format == 'hwp':
            return TextExtractor.extract_hwp(filepath)
        else:
            return f"[추출 실패: 알 수 없는 파일 형식 ({file_format})]"


# ============================================================
# 텍스트 정제 클래스
# ============================================================

class TextCleaner:
    """텍스트 정제 및 검증"""
    
    @staticmethod
    def clean(text: str) -> str:
        """
        텍스트 정제
        - 특수문자 제거 (한글, 영문, 숫자, 기본 공백문자만 유지)
        - NULL 문자 제거
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정제된 텍스트
        """
        # 허용: 영문, 숫자, 공백, 탭, 줄바꿈, 한글
        cleaned = re.sub(
            r'[^\x20-\x7E\n\r\t\uAC00-\uD7AF]', 
            '', 
            str(text)
        )
        
        # NULL 문자 제거
        cleaned = cleaned.replace('\x00', '')
        
        return cleaned
    
    @staticmethod
    def validate(text: str, min_length: int = 100) -> bool:
        """
        텍스트 유효성 검사
        
        Args:
            text: 검증할 텍스트
            min_length: 최소 길이
            
        Returns:
            유효 여부
        """
        if not text or text.strip() == "":
            return False
        
        if "[추출 실패" in text:
            return False
        
        if len(text) < min_length:
            return False
        
        return True
    
    @staticmethod
    def get_stats(text: str) -> dict:
        """
        텍스트 통계 정보
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            통계 딕셔너리
        """
        return {
            'length': len(text),
            'lines': text.count('\n') + 1,
            'words': len(text.split()),
            'is_valid': TextCleaner.validate(text)
        }


# ============================================================
# 문서 청킹 클래스
# ============================================================

class DocumentChunker:
    """문서를 청크로 분할"""
    
    def __init__(self, config: PreprocessConfig):
        """
        초기화
        
        Args:
            config: 전처리 설정 객체
        """
        self.config = config
        
        # LangChain 텍스트 분할기 초기화
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.SEPARATORS,
            length_function=len,
        )
    
    def chunk_document(self, text: str, metadata: dict) -> list:
        """
        단일 문서 청킹
        
        Args:
            text: 문서 텍스트
            metadata: 문서 메타데이터
            
        Returns:
            청크 리스트
        """
        try:
            chunks = self.splitter.split_text(text)
        except Exception as e:
            print(f"WARNING: 문서 분할 실패 - {e}")
            return []
        
        chunk_records = []
        filename = metadata.get('파일명', 'unknown')
        
        for i, chunk_content in enumerate(chunks, 1):
            chunk_record = metadata.copy()
            chunk_record['chunk_id'] = f"{filename}_chunk_{i:04d}"
            chunk_record['chunk_content'] = chunk_content
            chunk_records.append(chunk_record)
        
        return chunk_records
    
    def chunk_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text_content'
    ) -> pd.DataFrame:
        """
        DataFrame 전체 청킹
        
        Args:
            df: 원본 DataFrame
            text_column: 텍스트가 들어있는 컬럼명
            
        Returns:
            청크 DataFrame
        """
        print(f"청킹 시작 (크기: {self.config.CHUNK_SIZE}, "
              f"오버랩: {self.config.CHUNK_OVERLAP})...")
        
        all_chunks = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="청킹"):
            text = row[text_column]
            
            # 메타데이터 준비 (텍스트 컬럼 제외)
            metadata = row.to_dict()
            metadata.pop(text_column, None)
            metadata.pop('text_length', None)
            
            # 청킹
            chunks = self.chunk_document(text, metadata)
            all_chunks.extend(chunks)
        
        df_chunks = pd.DataFrame(all_chunks)
        
        print(f"청킹 완료: 원본 {len(df)}개 → 청크 {len(df_chunks)}개")
        
        return df_chunks


# ============================================================
# RAG 전처리 파이프라인
# ============================================================

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
        
        # [추가] 필터링 전 상태 확인
        original_count = len(df)
        print(f"🔍 필터링 전 문서 수: {original_count}")
        
        # 샘플 텍스트 미리보기
        if len(df) > 0:
            sample = df['text_content'].iloc[0]
            print(f"🔍 첫 번째 문서 미리보기:")
            print(f"   시작 부분: {sample[:100]}...")
            print(f"   전체 길이: {len(sample)}자")
            
            # 추출 실패 패턴이 있는지 확인
            has_failure = any([
                '[추출 실패' in sample,
                '[PDF 추출 실패' in sample,
                '[HWP 추출 실패' in sample
            ])
            print(f"   추출 실패 포함?: {has_failure}")
        
        # 추출 실패 문서 필터링 (raw string 사용)
        df = df[~df['text_content'].str.contains(r'\[추출 실패', na=False)]
        df = df[~df['text_content'].str.contains(r'\[PDF 추출 실패', na=False)]
        df = df[~df['text_content'].str.contains(r'\[HWP 추출 실패', na=False)]
        
        filtered_count = original_count - len(df)
        
        print(f"\n📊 필터링 결과:")
        print(f"   제외된 문서: {filtered_count}개")
        print(f"   남은 문서: {len(df)}개")
        
        if len(df) == 0:
            print("\n❌ 경고: 모든 문서가 필터링되었습니다!")
            print("   → 추출이 모두 실패했거나 필터링 조건이 너무 엄격합니다.")
            return pd.DataFrame()
        
        if filtered_count > 0:
            print(f"⚠️  추출 실패 문서 제외: {filtered_count}개")
            print(f"✅ 유효한 문서: {len(df)}개")
        
        # 청킹 시작
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