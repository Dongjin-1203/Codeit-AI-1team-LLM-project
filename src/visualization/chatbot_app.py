"""
공공기관 사업제안서 RAG 챗봇

기능:
- RAG 기반 질의응답
- 참고 문서 표시
- 대화 히스토리 관리
- 설정 조정 (Top-K, Temperature)
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.generator.rag_pipeline import RAGPipeline
from src.utils.rag_config import RAGConfig


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
</style>
""", unsafe_allow_html=True)


# ===== 세션 상태 초기화 =====
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None


# ===== RAG 파이프라인 초기화 =====
@st.cache_resource
def initialize_rag():
    """RAG 파이프라인 초기화 (캐싱)"""
    try:
        config = RAGConfig()
        rag = RAGPipeline(config=config)
        return rag, None
    except Exception as e:
        return None, str(e)


# ===== 답변 생성 =====
def generate_answer(query: str, top_k: int = 3, temperature: float = 0.1):
    """질의에 대한 답변 생성"""
    try:
        # 일단 파라미터 없이 호출
        result = st.session_state.rag_pipeline.generate_answer(query)
        
        # TODO: RAGPipeline에 top_k, temperature 파라미터 추가 필요
        
        return result
        
    except Exception as e:
        return {
            'answer': f"❌ 오류가 발생했습니다: {str(e)}",
            'sources': [],
            'usage': {'total_tokens': 0}
        }


# ===== 메시지 표시 =====
def display_message(role: str, content: str, sources: list = None, usage: dict = None):
    """
    메시지를 화면에 표시
    
    Args:
        role: 'user' 또는 'assistant'
        content: 메시지 내용
        sources: 참고 문서 리스트 (assistant만)
        usage: 토큰 사용량 (assistant만)
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
        
        # 참고 문서
        if sources:
            st.markdown("### 📚 참고 문서")
            
            for i, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                
                # 관련도 점수
                score = source.get('score', 0)
                score_percentage = score * 100 if score else 0
                
                # 문서 내용 미리보기
                content_preview = source.get('content', '')[:200] + "..."
                
                st.markdown(f"""
                <div class="source-document">
                    <div class="source-header">
                        📄 문서 {i} (관련도: {score_percentage:.1f}%)
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
    st.markdown('<div class="sub-header">RAG 기반 질의응답 시스템</div>', unsafe_allow_html=True)
    
    # ===== 사이드바 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # RAG 파라미터
        st.markdown("### 🔍 검색 설정")
        
        top_k = st.slider(
            "검색할 문서 개수 (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="관련성 높은 상위 K개 문서를 검색합니다"
        )
        
        st.markdown("---")
        
        st.markdown("### 🎯 생성 설정")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="낮을수록 일관된 답변, 높을수록 창의적인 답변"
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
    
    # ===== RAG 파이프라인 초기화 =====
    if st.session_state.rag_pipeline is None:
        with st.spinner("🔄 RAG 파이프라인 초기화 중..."):
            rag, error = initialize_rag()
            
            if error:
                st.error(f"❌ RAG 파이프라인 초기화 실패: {error}")
                st.info("""
                ### 💡 해결 방법
                
                1. ChromaDB가 생성되었는지 확인:
```bash
                python main.py --step embed
```
                
                2. OpenAI API 키가 설정되었는지 확인:
```bash
                # .env 파일
                OPENAI_API_KEY=your-key-here
```
                """)
                return
            
            st.session_state.rag_pipeline = rag
            st.success("✅ RAG 파이프라인 준비 완료!")
    
    # ===== 대화 히스토리 표시 =====
    st.markdown("---")
    
    if len(st.session_state.messages) == 0:
        st.info("""
        ### 👋 환영합니다!
        
        공공기관 사업제안서에 대해 질문해보세요. 예시:
        - "한영대학교 특성화 사업의 목적은 무엇인가요?"
        - "학사 정보 시스템 구축 사업에 대해 설명해주세요"
        - "서울시립대학교의 주요 사업은 무엇인가요?"
        """)
    
    # 기존 메시지 표시
    for msg in st.session_state.messages:
        display_message(
            role=msg['role'],
            content=msg['content'],
            sources=msg.get('sources'),
            usage=msg.get('usage')
        )
    
    # ===== 질문 입력 =====
    st.markdown("---")
    
    with st.form(key='question_form', clear_on_submit=True):
        user_input = st.text_area(
            "질문을 입력하세요:",
            height=100,
            placeholder="예: 한영대학교 특성화 사업의 목적은 무엇인가요?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            submit_button = st.form_submit_button("📤 전송", use_container_width=True)
        
        with col2:
            # 예시 질문 버튼은 form 밖에서 처리
            pass
    
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
                temperature=temperature
            )
        
        # 어시스턴트 메시지 추가
        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['answer'],
            'sources': result.get('sources', []),
            'usage': result.get('usage', {})
        })
        
        # 화면 새로고침
        st.rerun()
    
    # ===== 예시 질문 =====
    st.markdown("### 💡 예시 질문")
    
    col1, col2, col3 = st.columns(3)
    
    example_questions = [
        "한영대학교 특성화 사업은?",
        "학사 정보 시스템 구축 내용은?",
        "서울시립대 주요 사업은?"
    ]
    
    for col, question in zip([col1, col2, col3], example_questions):
        with col:
            if st.button(f"💬 {question}", use_container_width=True):
                st.session_state.messages.append({
                    'role': 'user',
                    'content': question
                })
                
                with st.spinner("🤔 답변 생성 중..."):
                    result = generate_answer(
                        query=question,
                        top_k=top_k,
                        temperature=temperature
                    )
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result.get('sources', []),
                    'usage': result.get('usage', {})
                })
                
                st.rerun()


if __name__ == "__main__":
    main()