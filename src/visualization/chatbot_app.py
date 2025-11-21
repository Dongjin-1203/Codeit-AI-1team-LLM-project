"""
공공기관 사업제안서 RAG 챗봇

기능:
- 모델 선택 (API/로컬)
- RAG 기반 질의응답 (Hybrid Search + Re-ranker)
- 참고 문서 표시
- 대화 히스토리 관리
- 검색 모드 선택
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.config import RAGConfig


# ===== 페이지 설정 =====
st.set_page_config(
    page_title="공공기관 사업제안서 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== 스타일 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .message-content {
        line-height: 1.6;
    }
    .source-document {
        background-color: #fff9c4;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #fbc02d;
    }
    .source-header {
        font-weight: bold;
        color: #f57f17;
        margin-bottom: 0.5rem;
    }
    .metadata {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .token-usage {
        background-color: #e8f5e9;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .search-mode-info {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ===== 세션 상태 초기화 =====
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'model_type' not in st.session_state:
    st.session_state.model_type = None


# ===== RAG 파이프라인 초기화 =====
@st.cache_resource
def initialize_rag(model_type):
    """RAG 파이프라인 초기화 (모델 타입에 따라 분기)"""
    try:
        config = RAGConfig()
        
        if model_type == "API 모델 (GPT)":
            from src.generator.generator import RAGPipeline
            rag = RAGPipeline(config=config)
            return rag, None, "API"
        
        else:  # "로컬 모델 (GGUF)"
            from src.generator.generator_gguf import GGUFRAGPipeline
            rag = GGUFRAGPipeline(config=config)
            return rag, None, "Local-GGUF"
            
    except Exception as e:
        return None, str(e), None


# ===== 답변 생성 =====
def generate_answer(query: str, top_k: int = 10, search_mode: str = "hybrid_rerank", alpha: float = 0.5):
    """질의에 대한 답변 생성"""
    try:
        result = st.session_state.rag_pipeline.generate_answer(
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            alpha=alpha
        )
        return result
        
    except Exception as e:
        return {
            'answer': f"❌ 오류가 발생했습니다: {str(e)}",
            'sources': [],
            'search_mode': search_mode,
            'usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        }


# ===== 메시지 표시 =====
def display_message(role: str, content: str, sources: list = None, usage: dict = None, search_mode: str = None):
    """
    메시지를 화면에 표시
    
    Args:
        role: 'user' 또는 'assistant'
        content: 메시지 내용
        sources: 참고 문서 리스트 (assistant만)
        usage: 토큰 사용량 (assistant만)
        search_mode: 검색 모드 (assistant만)
    """
    if role == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">
                👤 사용자
            </div>
            <div class="message-content">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # assistant
        # 답변
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-header">
                🤖 챗봇
            </div>
            <div class="message-content">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 검색 모드 정보
        if search_mode:
            mode_display = {
                'hybrid_rerank': '🔄 Hybrid + Re-ranker',
                'hybrid': '🔀 Hybrid Search',
                'embedding_rerank': '📊 임베딩 + Re-ranker',
                'embedding': '📊 임베딩 검색'
            }
            st.markdown(f"""
            <div class="search-mode-info">
                검색 모드: {mode_display.get(search_mode, search_mode)}
            </div>
            """, unsafe_allow_html=True)
        
        # 참고 문서
        if sources:
            st.markdown("### 📚 참고 문서")
            
            for i, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                
                # 관련도 점수
                score = source.get('score', 0)
                score_type = source.get('score_type', '')
                
                # 문서 내용 미리보기
                content_preview = source.get('content', '')[:200] + "..."
                
                st.markdown(f"""
                <div class="source-document">
                    <div class="source-header">
                        📄 문서 {i} (점수: {score:.3f} / {score_type})
                    </div>
                    <div>
                        {content_preview}
                    </div>
                    <div class="metadata">
                        📁 파일: {metadata.get('파일명', 'N/A')}<br>
                        🏢 발주기관: {metadata.get('발주 기관', 'N/A')}<br>
                        📋 사업명: {metadata.get('사업명', 'N/A')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 토큰 사용량
        if usage:
            st.markdown(f"""
            <div class="token-usage">
                🔢 토큰 사용량: {usage.get('total_tokens', 0)} 
                (프롬프트: {usage.get('prompt_tokens', 0)}, 
                 완성: {usage.get('completion_tokens', 0)})
            </div>
            """, unsafe_allow_html=True)


# ===== 메인 앱 =====
def main():
    # 헤더
    st.markdown('<div class="main-header">🤖 공공기관 사업제안서 챗봇</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG 기반 질의응답 시스템 (Hybrid Search + Re-ranker)</div>', unsafe_allow_html=True)
    
    # ===== 사이드바 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 모델 설정
        st.markdown("### 🤖 모델 설정")
        
        model_type = st.selectbox(
            "생성 모델 선택",
            options=[
                "API 모델 (GPT)",
                "로컬 모델 (GGUF)"  
            ],
            index=0,
            help="""
            • API 모델: 비용 발생, 빠르고 안정적, OpenAI API 사용
            • 로컬 모델 (GGUF): 무료 모델, CPU/GPU 효율적, 메모리 절약
            """
        )
        
        # 모델 정보 표시
        if model_type == "API 모델 (GPT)":
            st.info("🌐 OpenAI GPT 모델 사용 중")
        else:  # GGUF
            st.info("⚡ 로컬 GGUF 모델 사용 중 (메모리 효율)")
        
        st.markdown("---")
        
        # 검색 설정
        st.markdown("### 🔍 검색 설정")
        
        search_mode = st.selectbox(
            "검색 모드",
            options=["hybrid", "embedding"],
            index=0,
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid Search (BM25 + 임베딩)",
                "embedding": "📊 임베딩 검색"
            }[x],
            help="Hybrid: 키워드 + 의미 검색 병행 (권장)"
        )
        
        # Reranker 토글
        use_reranker = st.toggle(
            "🔄 Re-ranker 사용",
            value=True,
            help="검색 결과를 CrossEncoder로 재정렬하여 정확도 향상 (권장)"
        )
        
        # 실제 검색 모드 결정
        if use_reranker:
            if search_mode == "hybrid":
                actual_search_mode = "hybrid_rerank"
            else:  # embedding
                actual_search_mode = "embedding_rerank"
        else:
            actual_search_mode = search_mode
        
        top_k = st.slider(
            "검색할 문서 개수 (Top-K)",
            min_value=1,
            max_value=20,
            value=10,
            help="Re-ranker 사용 시 더 많이 검색해도 좋음"
        )
        
        alpha = st.slider(
            "임베딩 가중치 (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0: BM25만, 1: 임베딩만, 0.5: 동일 가중치 (Hybrid 모드에서만 사용)",
            disabled=(search_mode == "embedding")
        )
        
        st.markdown("---")
        
        # 대화 관리
        st.markdown("### 💬 대화 관리")
        
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("💾 대화 다운로드", use_container_width=True):
            if st.session_state.messages:
                # JSON으로 대화 저장
                chat_data = {
                    'timestamp': datetime.now().isoformat(),
                    'messages': st.session_state.messages
                }
                
                json_str = json.dumps(chat_data, ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="📥 JSON 다운로드",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # 통계
        st.markdown("### 📊 통계")
        st.metric("총 대화 수", len(st.session_state.messages) // 2)
        
        # 현재 설정 표시
        st.markdown("---")
        st.markdown("### 📋 현재 설정")
        st.text(f"모델: {model_type}")
        st.text(f"검색 모드: {search_mode}")
        st.text(f"Re-ranker: {'✅ ON' if use_reranker else '❌ OFF'}")
        st.text(f"실제 모드: {actual_search_mode}")
        st.text(f"Top-K: {top_k}")
        if search_mode == "hybrid":
            st.text(f"Alpha: {alpha}")
    
    # ===== RAG 파이프라인 초기화 =====
    # 모델 타입이 변경되었거나 파이프라인이 없으면 재초기화
    if (st.session_state.rag_pipeline is None or 
        st.session_state.model_type != model_type):
        
        with st.spinner(f"🔄 {model_type} 초기화 중..."):
            rag, error, rag_type = initialize_rag(model_type)
            
            if error:
                st.error(f"❌ RAG 파이프라인 초기화 실패: {error}")
                st.info("""
                ### 💡 해결 방법
                
                1. ChromaDB가 생성되었는지 확인:
```bash
python main.py --step embed
```
                
                2. OpenAI API 키가 설정되었는지 확인 (API 모델 사용 시):
```bash
# .env 파일
OPENAI_API_KEY=your-key-here
```
                
                3. 로컬 모델 어댑터 경로 확인 (로컬 모델 사용 시):
```bash
# config.py
FINETUNED_ADAPTER_PATH = "./models/qlora_adapter"
```
                
                4. 필요한 패키지 설치:
```bash
pip install rank-bm25 sentence-transformers
```
                """)
                return
            
            st.session_state.rag_pipeline = rag
            st.session_state.model_type = model_type
            st.success(f"✅ {rag_type} 모델 준비 완료!")
    
    # ===== 대화 히스토리 표시 =====
    st.markdown("---")
    
    if len(st.session_state.messages) == 0:
        st.info("""
        ### 👋 환영합니다!
        
        공공기관 사업제안서에 대해 질문해보세요. 예시:
        - "데이터 표준화 요구사항은 무엇인가요?"
        - "보안 관련 요구사항을 설명해주세요"
        - "사업 수행 시 산출물은 어디에 저장해야 하나요?"
        """)
    
    # 기존 메시지 표시
    for msg in st.session_state.messages:
        display_message(
            role=msg['role'],
            content=msg['content'],
            sources=msg.get('sources'),
            usage=msg.get('usage'),
            search_mode=msg.get('search_mode')
        )
    
    # ===== 질문 입력 =====
    st.markdown("---")
    
    with st.form(key='question_form', clear_on_submit=True):
        user_input = st.text_area(
            "질문을 입력하세요:",
            height=100,
            placeholder="예: 데이터 표준화 요구사항은 무엇인가요?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            submit_button = st.form_submit_button("📤 전송", use_container_width=True)
    
    # ===== 질문 처리 =====
    if submit_button and user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # 답변 생성
        with st.spinner("🤔 답변 생성 중..."):
            result = generate_answer(
                query=user_input,
                top_k=top_k,
                search_mode=actual_search_mode,
                alpha=alpha
            )
        
        # 어시스턴트 메시지 추가
        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result.get('sources', []),
            'usage': result.get('usage', {}),
            'search_mode': result.get('search_mode', actual_search_mode)
        })
        
        # 화면 새로고침
        st.rerun()


if __name__ == "__main__":
    main()