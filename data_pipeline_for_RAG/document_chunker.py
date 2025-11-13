"""
문서를 청크로 분할
LangChain의 RecursiveCharacterTextSplitter 사용
"""

import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

from preprocess_config import PreprocessConfig


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