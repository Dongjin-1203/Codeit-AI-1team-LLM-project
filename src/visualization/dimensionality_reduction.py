"""
차원 축소 (Dimensionality Reduction)
1536차원 임베딩 벡터 → 2D/3D 좌표로 변환
"""

import numpy as np
import pandas as pd
from typing import Literal, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class DimensionalityReducer:
    """차원 축소 클래스"""
    
    def __init__(
        self,
        method: Literal['pca', 'tsne'] = 'pca',
        n_components: int = 2,
        random_state: int = 42
    ):
        """
        초기화
        
        Args:
            method: 차원 축소 방법 ('pca' 또는 'tsne')
            n_components: 축소할 차원 (2 또는 3)
            random_state: 랜덤 시드
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        
        self._initialize_reducer()
    
    def _initialize_reducer(self):
        """차원 축소 모델 초기화"""
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
            print(f"✅ PCA 초기화 완료 ({self.n_components}D)")
            
        elif self.method == 'tsne':
            self.reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=30,  # 기본값
                max_iter=1000,  # n_iter → max_iter로 변경
                verbose=0  # verbose=1 → 0 (Streamlit에서는 0이 좋음)
            )
            print(f"✅ t-SNE 초기화 완료 ({self.n_components}D)")
            
        else:
            raise ValueError(f"지원하지 않는 방법: {self.method}")

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        차원 축소 실행
        
        Args:
            embeddings: 원본 임베딩 벡터 (N, 1536)
            
        Returns:
            축소된 좌표 (N, 2) 또는 (N, 3)
        """
        print(f"\n차원 축소 시작...")
        print(f"  방법: {self.method.upper()}")
        print(f"  입력 shape: {embeddings.shape}")
        print(f"  목표 차원: {self.n_components}D")
        
        # t-SNE의 경우 perplexity 재설정
        if self.method == 'tsne':
            n_samples = embeddings.shape[0]
            perplexity = min(30, n_samples - 1)
            self.reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=perplexity,
                max_iter=1000  # n_iter → max_iter로 변경
            )
        
        # 차원 축소
        reduced = self.reducer.fit_transform(embeddings)
        
        print(f"  출력 shape: {reduced.shape}")
        print(f"✅ 차원 축소 완료")
        
        # PCA인 경우 설명된 분산 비율 출력
        if self.method == 'pca':
            explained_var = self.reducer.explained_variance_ratio_
            print(f"  설명된 분산:")
            for i, var in enumerate(explained_var, 1):
                print(f"    PC{i}: {var:.2%}")
            print(f"    총합: {explained_var.sum():.2%}")
        
        return reduced
    
    def add_coordinates_to_dataframe(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray
    ) -> pd.DataFrame:
        """
        DataFrame에 2D/3D 좌표 추가
        
        Args:
            df: 원본 DataFrame
            embeddings: 임베딩 벡터
            
        Returns:
            좌표가 추가된 DataFrame
        """
        # 차원 축소
        reduced = self.fit_transform(embeddings)
        
        # DataFrame에 추가
        df = df.copy()
        
        if self.n_components == 2:
            df['x'] = reduced[:, 0]
            df['y'] = reduced[:, 1]
            print(f"\n✅ 2D 좌표 추가 완료 (x, y)")
            
        elif self.n_components == 3:
            df['x'] = reduced[:, 0]
            df['y'] = reduced[:, 1]
            df['z'] = reduced[:, 2]
            print(f"\n✅ 3D 좌표 추가 완료 (x, y, z)")
        
        return df


def compare_methods(
    embeddings: np.ndarray,
    methods: list = ['pca', 'tsne'],
    n_components: int = 2
) -> dict:
    """
    여러 차원 축소 방법 비교
    
    Args:
        embeddings: 임베딩 벡터
        methods: 비교할 방법 리스트
        n_components: 차원
        
    Returns:
        {method: reduced_coords} 딕셔너리
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"{method.upper()} 실행 중...")
        print('='*60)
        
        reducer = DimensionalityReducer(
            method=method,
            n_components=n_components
        )
        
        reduced = reducer.fit_transform(embeddings)
        results[method] = reduced
    
    return results


# ===== 단독 실행용 =====
if __name__ == "__main__":
    import argparse
    from src.visualization.vector_db_loader import VectorDBLoader
    from src.utils.rag_config import RAGConfig
    
    parser = argparse.ArgumentParser(description='차원 축소 테스트')
    parser.add_argument(
        '--method',
        type=str,
        choices=['pca', 'tsne', 'both'],
        default='pca',
        help='차원 축소 방법'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        choices=[2, 3],
        default=2,
        help='축소할 차원 (2D 또는 3D)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='결과를 CSV로 저장할 경로 (선택)'
    )
    
    args = parser.parse_args()
    
    # 1. 데이터 로드
    print("="*60)
    print("ChromaDB 데이터 로드")
    print("="*60)
    
    config = RAGConfig()
    loader = VectorDBLoader(config)
    df = loader.to_dataframe()
    
    print(f"\n로드된 데이터: {len(df)}개")
    
    # 2. 임베딩 추출
    embeddings = np.array(df['embedding'].tolist())
    print(f"임베딩 shape: {embeddings.shape}")
    
    # 3. 차원 축소
    if args.method == 'both':
        results = compare_methods(embeddings, methods=['pca', 'tsne'], n_components=args.n_components)
        
        # PCA 결과를 DataFrame에 추가
        reducer = DimensionalityReducer(method='pca', n_components=args.n_components)
        df = reducer.add_coordinates_to_dataframe(df, embeddings)
        
    else:
        reducer = DimensionalityReducer(method=args.method, n_components=args.n_components)
        df = reducer.add_coordinates_to_dataframe(df, embeddings)
    
    # 4. 결과 확인
    print("\n" + "="*60)
    print("결과 요약")
    print("="*60)
    print(f"최종 DataFrame shape: {df.shape}")
    print(f"좌표 컬럼: {['x', 'y', 'z'][:args.n_components]}")
    
    # 좌표 통계
    print(f"\n좌표 범위:")
    print(f"  x: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"  y: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    if args.n_components == 3:
        print(f"  z: [{df['z'].min():.2f}, {df['z'].max():.2f}]")
    
    # 5. CSV 저장 (옵션)
    if args.export:
        df_export = df.drop(columns=['embedding'])
        df_export.to_csv(args.export, index=False, encoding='utf-8-sig')
        print(f"\n✅ 데이터 저장 완료: {args.export}")