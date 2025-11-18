"""
ChromaDB 벡터 데이터베이스 로더
임베딩 벡터와 메타데이터를 추출하여 시각화용 데이터 준비
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

from src.utils.config import RAGConfig


class VectorDBLoader:
    """ChromaDB에서 벡터와 메타데이터를 추출하는 클래스"""
    
    def __init__(self, config: RAGConfig = None):
        """
        초기화
        
        Args:
            config: RAG 설정 객체
        """
        self.config = config or RAGConfig()
        self.vectorstore = None
        self.embeddings = None
        
        self._initialize()
    
    def _initialize(self):
        """임베딩 모델 및 벡터스토어 초기화"""
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=self.config.EMBEDDING_MODEL_NAME,
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        # 벡터스토어 연결
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.config.DB_DIRECTORY,
            collection_name=self.config.COLLECTION_NAME
        )
        
        print(f"✅ ChromaDB 연결 완료")
        print(f"   경로: {self.config.DB_DIRECTORY}")
        print(f"   Collection: {self.config.COLLECTION_NAME}")
    
    def get_collection_info(self) -> Dict:
        """
        Collection 기본 정보 반환
        
        Returns:
            dict: Collection 통계 정보
        """
        collection = self.vectorstore._collection
        count = collection.count()
        
        if count == 0:
            return {
                'total_documents': 0,
                'embedding_dimension': 0,
                'metadata_keys': [],
                'collection_name': self.config.COLLECTION_NAME
            }
        
        # 샘플 데이터 가져오기
        sample = collection.get(limit=1, include=['embeddings', 'metadatas'])
        
        # 임베딩 차원 확인
        embedding_dim = 0
        if sample.get('embeddings') is not None and len(sample['embeddings']) > 0:
            embedding_dim = len(sample['embeddings'][0])
        
        # 메타데이터 키 확인
        metadata_keys = []
        if sample.get('metadatas') is not None and len(sample['metadatas']) > 0:
            if sample['metadatas'][0]:
                metadata_keys = list(sample['metadatas'][0].keys())
        
        info = {
            'total_documents': count,
            'embedding_dimension': embedding_dim,
            'metadata_keys': metadata_keys,
            'collection_name': self.config.COLLECTION_NAME
        }
        
        return info
    
    def extract_all_data(self) -> Dict:
        """
        모든 데이터를 추출
        
        Returns:
            dict: {
                'embeddings': 임베딩 벡터 배열 (numpy),
                'metadatas': 메타데이터 리스트,
                'documents': 문서 텍스트 리스트,
                'ids': 문서 ID 리스트
            }
        """
        print("\n데이터 추출 중...")
        
        collection = self.vectorstore._collection
        
        # 모든 데이터 가져오기
        results = collection.get(
            include=['embeddings', 'metadatas', 'documents']
        )
        
        # 데이터가 없는 경우 처리
        if not results['ids'] or len(results['ids']) == 0:
            print("⚠️  ChromaDB에 데이터가 없습니다!")
            print("   먼저 임베딩 단계를 실행하세요: python main.py --step embed")
            return {
                'embeddings': np.array([]),
                'metadatas': [],
                'documents': [],
                'ids': []
            }
        
        # numpy array로 변환
        embeddings_array = np.array(results['embeddings'])
        
        print(f"✅ 총 {len(results['ids'])}개의 청크를 불러왔습니다.")
        if embeddings_array.ndim == 2:  # 2D 배열인 경우에만
            print(f"✅ 임베딩 차원: {embeddings_array.shape[1]}차원")
        
        return {
            'embeddings': embeddings_array,
            'metadatas': results['metadatas'],
            'documents': results['documents'],
            'ids': results['ids']
        }
    
    def to_dataframe(self, data: Dict = None) -> pd.DataFrame:
        """
        추출한 데이터를 DataFrame으로 변환
        
        Args:
            data: extract_all_data()의 결과 (None이면 자동 추출)
            
        Returns:
            pd.DataFrame: 정리된 데이터프레임
        """
        if data is None:
            data = self.extract_all_data()
        
        # 데이터가 없으면 빈 DataFrame 반환
        if len(data['ids']) == 0:
            return pd.DataFrame()
        
        # 기본 컬럼
        df = pd.DataFrame({
            'id': data['ids'],
            'document': data['documents'],
        })
        
        # 메타데이터를 개별 컬럼으로 분리
        if data['metadatas']:
            # 메타데이터의 모든 키 추출
            metadata_keys = set()
            for metadata in data['metadatas']:
                if metadata:
                    metadata_keys.update(metadata.keys())
            
            # 각 메타데이터 키를 컬럼으로 추가
            for key in metadata_keys:
                df[key] = [
                    metadata.get(key, None) if metadata else None 
                    for metadata in data['metadatas']
                ]
        
        # 임베딩 벡터 추가 (numpy array로)
        df['embedding'] = list(data['embeddings'])
        
        print(f"\n📊 DataFrame 정보:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {df.columns.tolist()}")
        
        return df
    
    def get_metadata_stats(self, df: pd.DataFrame = None) -> Dict:
        """
        메타데이터 통계 정보
        
        Args:
            df: DataFrame (None이면 자동 생성)
            
        Returns:
            dict: 메타데이터별 통계
        """
        if df is None or len(df) == 0:
            return {}
        
        stats = {}
        
        # embedding과 document 컬럼 제외
        metadata_cols = [col for col in df.columns 
                        if col not in ['id', 'document', 'embedding']]
        
        for col in metadata_cols:
            if df[col].dtype == 'object':  # 범주형 데이터
                stats[col] = {
                    'type': 'categorical',
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(10).to_dict()
                }
            else:  # 숫자형 데이터
                stats[col] = {
                    'type': 'numerical',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return stats
    
    def print_summary(self):
        """데이터 요약 정보 출력"""
        print("\n" + "="*60)
        print("ChromaDB 데이터 요약")
        print("="*60)
        
        # Collection 정보
        info = self.get_collection_info()
        print(f"\n📦 Collection: {info['collection_name']}")
        print(f"📊 총 문서 수: {info['total_documents']}")
        
        # 데이터가 없으면 여기서 종료
        if info['total_documents'] == 0:
            print("\n⚠️  ChromaDB에 데이터가 없습니다!")
            print("   먼저 임베딩 단계를 실행하세요:")
            print("   python main.py --step embed")
            print("="*60)
            return None
        
        print(f"🧮 임베딩 차원: {info['embedding_dimension']}")
        print(f"🏷️  메타데이터 키: {', '.join(info['metadata_keys'])}")
        
        # DataFrame 생성
        df = self.to_dataframe()
        
        if len(df) == 0:
            print("\n⚠️  DataFrame 생성 실패")
            print("="*60)
            return None
        
        # 메타데이터 통계
        stats = self.get_metadata_stats(df)
        
        if stats:
            print("\n📈 메타데이터 분포:")
            for key, stat in stats.items():
                if stat['type'] == 'categorical':
                    print(f"\n  [{key}]")
                    print(f"    - 고유값: {stat['unique_count']}개")
                    print(f"    - 상위 값:")
                    for val, count in list(stat['top_values'].items())[:5]:
                        print(f"      • {val}: {count}개")
        
        print("\n" + "="*60)
        
        return df


# ===== 단독 실행용 =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ChromaDB 데이터 추출 및 확인')
    parser.add_argument(
        '--export',
        type=str,
        help='DataFrame을 CSV로 저장할 경로 (선택사항)'
    )
    
    args = parser.parse_args()
    
    # 설정 초기화
    config = RAGConfig()
    
    # 데이터 로더 초기화
    loader = VectorDBLoader(config)
    
    # 요약 정보 출력 및 DataFrame 생성
    df = loader.print_summary()
    
    # CSV 저장 (옵션)
    if df is not None and args.export:
        # 임베딩 벡터를 제외하고 저장 (파일 크기 때문)
        df_export = df.drop(columns=['embedding'])
        df_export.to_csv(args.export, index=False, encoding='utf-8-sig')
        print(f"\n✅ 데이터 저장 완료: {args.export}")