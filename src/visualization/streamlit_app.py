"""
벡터DB 시각화 Streamlit 앱
ChromaDB 데이터를 2D/3D로 시각화
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.visualization.vector_db_loader import VectorDBLoader
from src.visualization.dimensionality_reduction import DimensionalityReducer
from src.utils.rag_config import RAGConfig


# ===== 페이지 설정 =====
st.set_page_config(
    page_title="벡터DB 시각화",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== 스타일 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ===== 캐싱 함수 =====
@st.cache_data
def load_data():
    """ChromaDB 데이터 로드 (캐싱)"""
    config = RAGConfig()
    loader = VectorDBLoader(config)
    df = loader.to_dataframe()
    
    # 추출 실패 문서 필터링
    df = df[~df['document'].str.contains('\[추출 실패', na=False)]
    df = df[~df['document'].str.contains('\[PDF 추출 실패', na=False)]
    df = df[~df['document'].str.contains('\[HWP 추출 실패', na=False)]
    
    # 인덱스 리셋
    df = df.reset_index(drop=True)
    
    print(f"✅ 유효한 문서: {len(df)}개")
    
    # 임베딩 벡터 추출
    embeddings = np.array(df['embedding'].tolist())
    
    return df, embeddings


@st.cache_data
def reduce_dimensions(embeddings, method, n_components):
    """차원 축소 (캐싱)"""
    reducer = DimensionalityReducer(
        method=method,
        n_components=n_components
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


# ===== 메인 앱 =====
def main():
    # 헤더
    st.markdown('<div class="main-header">🔍 벡터DB 시각화</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ChromaDB 임베딩 공간 탐색</div>', unsafe_allow_html=True)
    
    # 데이터 로드
    with st.spinner("데이터 로드 중..."):
        try:
            df, embeddings = load_data()
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {e}")
            st.info("먼저 임베딩 단계를 실행하세요: `python main.py --step embed`")
            return
    
    # 데이터가 없으면 종료
    if len(df) == 0:
        st.warning("⚠️ ChromaDB에 데이터가 없습니다!")
        st.info("먼저 임베딩 단계를 실행하세요: `python main.py --step embed`")
        return
    
    # ===== 사이드바 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 통계 정보
        st.markdown("### 📊 데이터 정보")
        st.metric("총 문서 수", len(df))
        st.metric("임베딩 차원", embeddings.shape[1])
        
        st.markdown("---")
        
        # 차원 축소 설정
        st.markdown("### 🎯 차원 축소")
        
        method = st.selectbox(
            "방법",
            options=['pca', 'tsne'],
            format_func=lambda x: {
                'pca': 'PCA (빠름)',
                'tsne': 't-SNE (느림, 더 정확)'
            }[x]
        )
        
        n_components = st.radio(
            "차원",
            options=[2, 3],
            format_func=lambda x: f"{x}D"
        )
        
        st.markdown("---")
        
        # 필터링 옵션
        st.markdown("### 🎨 시각화 옵션")
        
        # 색상 기준
        color_options = ['없음'] + [col for col in df.columns 
                                    if col not in ['id', 'document', 'embedding', 'x', 'y', 'z']]
        
        color_by = st.selectbox(
            "색상 기준",
            options=color_options
        )
        
        # 크기 옵션
        point_size = st.slider(
            "포인트 크기",
            min_value=3,
            max_value=15,
            value=8
        )
        
        # 투명도
        opacity = st.slider(
            "투명도",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        st.markdown("---")
        
        # 필터링
        st.markdown("### 🔍 필터")
        
        filter_col = st.selectbox(
            "필터링 기준",
            options=['없음'] + color_options[1:]  # '없음' 제외한 나머지
        )
        
        filter_values = []
        if filter_col != '없음':
            unique_values = df[filter_col].unique()
            filter_values = st.multiselect(
                f"{filter_col} 선택",
                options=unique_values,
                default=list(unique_values)[:5] if len(unique_values) > 5 else list(unique_values)
            )
    
    # ===== 차원 축소 =====
    with st.spinner(f"{method.upper()}로 차원 축소 중..."):
        reduced = reduce_dimensions(embeddings, method, n_components)
    
    # DataFrame에 좌표 추가
    df_viz = df.copy()
    df_viz['x'] = reduced[:, 0]
    df_viz['y'] = reduced[:, 1]
    if n_components == 3:
        df_viz['z'] = reduced[:, 2]
    
    # 필터링 적용
    if filter_col != '없음' and filter_values:
        df_viz = df_viz[df_viz[filter_col].isin(filter_values)]
        st.info(f"필터링 결과: {len(df_viz)}개 문서")
    
    # ===== 시각화 =====
    st.markdown("---")
    st.markdown("### 📈 임베딩 공간 시각화")
    
    # hover 데이터 준비
    hover_data = {
        'document': True,
        'x': ':.2f',
        'y': ':.2f'
    }
    
    if n_components == 3:
        hover_data['z'] = ':.2f'
    
    # 메타데이터 hover에 추가
    for col in ['파일명', '발주 기관', '사업명']:
        if col in df_viz.columns:
            hover_data[col] = True
    
    # 색상 설정
    color = None if color_by == '없음' else color_by
    
    # 2D 시각화
    if n_components == 2:
        fig = px.scatter(
            df_viz,
            x='x',
            y='y',
            color=color,
            hover_data=hover_data,
            title=f"벡터 임베딩 공간 ({method.upper()}, 2D)",
            labels={'x': 'PC1' if method == 'pca' else 'Dim 1',
                   'y': 'PC2' if method == 'pca' else 'Dim 2'},
            height=700,
            opacity=opacity
        )
        
        fig.update_traces(marker=dict(size=point_size))
        
    # 3D 시각화
    else:
        fig = px.scatter_3d(
            df_viz,
            x='x',
            y='y',
            z='z',
            color=color,
            hover_data=hover_data,
            title=f"벡터 임베딩 공간 ({method.upper()}, 3D)",
            labels={'x': 'PC1' if method == 'pca' else 'Dim 1',
                   'y': 'PC2' if method == 'pca' else 'Dim 2',
                   'z': 'PC3' if method == 'pca' else 'Dim 3'},
            height=700,
            opacity=opacity
        )
        
        fig.update_traces(marker=dict(size=point_size))
    
    # 레이아웃 업데이트
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== 통계 정보 =====
    st.markdown("---")
    st.markdown("### 📊 통계 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("표시된 문서", len(df_viz))
    
    with col2:
        st.metric("필터링된 문서", len(df) - len(df_viz))
    
    with col3:
        if method == 'pca':
            # PCA 설명된 분산 표시
            reducer = DimensionalityReducer(method='pca', n_components=n_components)
            reducer.fit_transform(embeddings)
            explained_var = reducer.reducer.explained_variance_ratio_.sum()
            st.metric("설명된 분산", f"{explained_var:.1%}")
        else:
            st.metric("차원 축소 방법", "t-SNE")
    
    with col4:
        st.metric("임베딩 차원", embeddings.shape[1])
    
    # ===== 데이터 테이블 =====
    if st.checkbox("📋 데이터 테이블 보기", value=False):
        st.markdown("---")
        st.markdown("### 📋 데이터 테이블")
        
        # 표시할 컬럼 선택
        display_cols = st.multiselect(
            "표시할 컬럼 선택",
            options=[col for col in df_viz.columns if col != 'embedding'],
            default=['파일명', '발주 기관', '사업명'][:min(3, len(df_viz.columns))]
        )
        
        if display_cols:
            st.dataframe(
                df_viz[display_cols],
                use_container_width=True,
                height=400
            )
    
    # ===== 다운로드 옵션 =====
    st.markdown("---")
    st.markdown("### 💾 데이터 다운로드")

    df_download = df_viz.drop(columns=['embedding'])

    # BytesIO 버퍼 생성
    buffer = io.BytesIO()

    # Excel 파일 생성
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_download.to_excel(writer, index=False, sheet_name='VectorDB')

    st.download_button(
        label="📥 Excel 다운로드",
        data=buffer.getvalue(),
        file_name="vectordb_visualization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.caption("💡 Excel에서 바로 열 수 있으며 한글이 정상 표시됩니다.")

if __name__ == "__main__":
    main()