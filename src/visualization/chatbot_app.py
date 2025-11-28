"""
공공기관 사업제안서 RAG 챗봇

기능:
- 사용자 API 키 입력 및 검증
- 사용 가능한 GPT 모델 자동 조회 및 선택
- 모델 선택 (API/로컬 GGUF)
- Query Router (검색 vs 직접 답변)
- RAG 기반 질의응답 (Hybrid Search + Re-ranker)
- 조건부 참고 문서 표시
- 대화 히스토리 관리
- 검색 모드 선택
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.utils.config import RAGConfig
from src.utils.conversation_manager import ConversationManager


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
    .routing-info {
        background-color: #fff3e0;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        border-left: 3px solid #ff9800;
    }
    .model-info {
        background-color: #f3e5f5;
        padding: 0.8rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        border-left: 3px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)


# ===== 세션 상태 초기화 =====
if 'conv_manager' not in st.session_state:
    st.session_state.conv_manager = ConversationManager()

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'model_type' not in st.session_state:
    st.session_state.model_type = None

if 'show_routing_info' not in st.session_state:
    st.session_state.show_routing_info = False

if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = None

if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

if 'available_models' not in st.session_state:
    st.session_state.available_models = []

if 'selected_gpt_model' not in st.session_state:
    st.session_state.selected_gpt_model = "gpt-4o-mini"


# ===== API 키로 사용 가능한 모델 조회 함수 =====
def get_available_models(api_key: str) -> tuple:
    """
    API 키로 실제 사용 가능한 모든 GPT/o 시리즈 모델 조회
    
    Args:
        api_key: OpenAI API 키
    
    Returns:
        (success, model_list, error_message)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        # 모델 목록 조회
        models_response = client.models.list()
        
        # Chat Completion 가능한 모델만 필터링
        available_models = []
        
        for model in models_response.data:
            model_id = model.id
            
            # GPT 시리즈, o1, o3 시리즈만 선택
            if (model_id.startswith('gpt-') or 
                model_id.startswith('o1-') or 
                model_id.startswith('o3-')):
                available_models.append(model_id)
        
        if not available_models:
            return False, [], "사용 가능한 모델을 찾을 수 없습니다."
        
        # 우선순위 정렬 (최신/고급 모델 우선)
        priority_map = {
            'o3': 1,
            'o1': 2,
            'gpt-5': 3,
            'gpt-4o': 4,
            'gpt-4o-mini': 5,
            'gpt-4-turbo': 6,
            'gpt-4': 7,
            'gpt-3.5-turbo': 8,
            'gpt-3.5': 9
        }
        
        def get_priority(model_name):
            for prefix, priority in priority_map.items():
                if model_name.startswith(prefix):
                    return priority
            return 99
        
        available_models.sort(key=get_priority)
        
        # 중복 제거 (날짜 버전 중 가장 최근 것만)
        unique_models = []
        seen_bases = {}
        
        for model in available_models:
            # 기본 모델명 추출 (날짜/버전 제거)
            base = model
            for suffix in ['-preview', '-latest']:
                base = base.replace(suffix, '')
            
            # 날짜 패턴 제거 (예: -20241120, -2024-11-20)
            import re
            base = re.sub(r'-\d{8}$', '', base)
            base = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', base)
            
            # 같은 base가 이미 있으면 더 긴 이름 선택 (보통 최신)
            if base not in seen_bases or len(model) > len(seen_bases[base]):
                seen_bases[base] = model
        
        unique_models = list(seen_bases.values())
        unique_models.sort(key=get_priority)
        
        return True, unique_models, ""
        
    except Exception as e:
        error_msg = str(e)
        
        if "Incorrect API key" in error_msg:
            return False, [], "❌ 잘못된 API 키입니다."
        elif "insufficient_quota" in error_msg:
            return False, [], "⚠️ API 크레딧이 부족합니다."
        else:
            return False, [], f"❌ 모델 조회 실패: {error_msg}"


# ===== API 키 검증 함수 =====
def validate_api_key(api_key: str) -> tuple:
    """
    OpenAI API 키 유효성 검증 및 사용 가능한 모델 조회
    
    Args:
        api_key: 검증할 API 키
    
    Returns:
        (is_valid, message, available_models)
    """
    try:
        # 모델 목록 조회로 검증 (chat completion보다 권한 요구사항 낮음)
        success, models, error = get_available_models(api_key)
        
        if not success:
            return False, error, []
        
        if len(models) == 0:
            return False, "❌ 사용 가능한 모델이 없습니다.", []
        
        return True, f"✅ API 키가 유효합니다! ({len(models)}개 모델 사용 가능)", models
        
    except Exception as e:
        error_msg = str(e)
        
        if "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
            return False, "❌ 잘못된 API 키입니다. 다시 확인해주세요.", []
        elif "insufficient_quota" in error_msg:
            return False, "⚠️ API 키는 유효하지만 크레딧이 부족합니다.", []
        elif "403" in error_msg or "Forbidden" in error_msg:
            return False, "❌ API 키 권한이 부족합니다. 키 권한을 확인해주세요.", []
        else:
            return False, f"❌ API 키 검증 실패: {error_msg}", []
        
    except Exception as e:
        error_msg = str(e)
        
        if "Incorrect API key" in error_msg or "invalid_api_key" in error_msg:
            return False, "❌ 잘못된 API 키입니다. 다시 확인해주세요.", []
        elif "insufficient_quota" in error_msg:
            return False, "⚠️ API 키는 유효하지만 크레딧이 부족합니다.", []
        else:
            return False, f"❌ API 키 검증 실패: {error_msg}", []


# ===== RAG 파이프라인 초기화 =====
@st.cache_resource
def initialize_rag(model_type, _user_api_key=None, gpt_model_name=None):
    """
    RAG 파이프라인 초기화
    
    Args:
        model_type: "API 모델 (GPT)" 또는 "로컬 모델 (GGUF)"
        _user_api_key: 사용자가 입력한 API 키 (None이면 .env 사용)
        gpt_model_name: 사용할 GPT 모델 이름 (예: "gpt-4o-mini")
    
    Returns:
        (rag_pipeline, error_message, model_name)
    """
    try:
        config = RAGConfig()
        
        # 사용자 API 키가 있으면 덮어쓰기
        if _user_api_key:
            config.OPENAI_API_KEY = _user_api_key
            os.environ["OPENAI_API_KEY"] = _user_api_key
        
        # GPT 모델 이름 설정
        if gpt_model_name:
            config.LLM_MODEL_NAME = gpt_model_name
        
        if model_type == "API 모델 (GPT)":
            # API 모델 사용
            from src.generator.generator import RAGPipeline
            rag = RAGPipeline(config=config)
            return rag, None, f"OpenAI {config.LLM_MODEL_NAME}"
            
        elif model_type == "로컬 모델 (GGUF)":
            # GGUF 모델 사용
            from src.generator.generator_gguf import GGUFRAGPipeline
            
            rag = GGUFRAGPipeline(
                config=config,
                n_gpu_layers=35,
                n_ctx=8192,
                n_threads=4,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9
            )
            return rag, None, "Llama-3-Ko-8B (GGUF)"
        
        else:
            return None, f"알 수 없는 모델 타입: {model_type}", None
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, f"{str(e)}\n\n{error_detail}", None


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
        import traceback
        error_detail = traceback.format_exc()
        return {
            'answer': f"❌ 오류가 발생했습니다: {str(e)}\n\n{error_detail}",
            'sources': [],
            'used_retrieval': False,
            'search_mode': search_mode,
            'routing_info': None,
            'usage': {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        }


# ===== 메시지 표시 =====
def display_message(
    role: str, 
    content: str, 
    sources: list = None, 
    usage: dict = None, 
    search_mode: str = None,
    used_retrieval: bool = None,
    routing_info: dict = None
):
    """메시지를 화면에 표시"""
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
        
        # 라우팅 정보 (개발 모드)
        if st.session_state.show_routing_info and routing_info:
            route_icon = "🔍" if routing_info.get('route') == 'rag' else "💬"
            st.markdown(f"""
            <div class="routing-info">
                {route_icon} 라우팅: {routing_info.get('route', 'N/A').upper()} 
                (신뢰도: {routing_info.get('confidence', 0):.2f}) - 
                {routing_info.get('reason', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        # 검색 모드 정보 (검색 사용 시만)
        if used_retrieval and search_mode:
            mode_display = {
                'hybrid_rerank': '🔄 Hybrid + Re-ranker',
                'hybrid': '🔀 Hybrid Search',
                'embedding_rerank': '📊 임베딩 + Re-ranker',
                'embedding': '📊 임베딩 검색',
                'direct': '💬 Direct (검색 없음)'
            }
            st.markdown(f"""
            <div class="search-mode-info">
                검색 모드: {mode_display.get(search_mode, search_mode)}
            </div>
            """, unsafe_allow_html=True)
        
        # 참고 문서 (검색 사용 시만)
        if used_retrieval and sources and len(sources) > 0:
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
        elif not used_retrieval:
            # 검색을 사용하지 않은 경우 안내
            st.info("💬 이 답변은 문서 검색 없이 생성되었습니다.")
        
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
    st.markdown('<div class="sub-header">Query Router + RAG 기반 질의응답 시스템</div>', unsafe_allow_html=True)
    
    # ===== 사이드바 =====
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # ===== 🔑 API 키 설정 =====
        st.markdown("### 🔑 API 키 설정")
        
        config = RAGConfig()
        has_env_key = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY != "")
        
        if has_env_key:
            st.success("✅ 서버 API 키 사용 중")
        else:
            st.warning("⚠️ 서버 API 키가 없습니다. 아래에 입력하세요.")
        
        use_custom_key = st.checkbox(
            "🔓 내 API 키 사용하기",
            value=not has_env_key,
            help="OpenAI API 키를 직접 입력하여 사용합니다."
        )
        
        if use_custom_key:
            user_key_input = st.text_input(
                "OpenAI API 키 입력",
                type="password",
                placeholder="sk-...",
                help="https://platform.openai.com/api-keys 에서 발급받으세요"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                validate_button = st.button(
                    "🔍 검증",
                    use_container_width=True,
                    disabled=not user_key_input
                )
            
            with col2:
                apply_button = st.button(
                    "✅ 적용",
                    use_container_width=True,
                    disabled=not user_key_input,
                    type="primary"
                )
            
            # 검증 버튼
            if validate_button and user_key_input:
                with st.spinner("🔄 API 키 검증 및 모델 조회 중..."):
                    is_valid, message, models = validate_api_key(user_key_input)
                    
                    if is_valid:
                        st.success(message)
                        st.session_state.api_key_validated = True
                        st.session_state.available_models = models
                        
                        # 사용 가능한 모델 표시
                        if models:
                            st.info(f"📋 사용 가능한 모델: {', '.join(models)}")
                    else:
                        st.error(message)
                        st.session_state.api_key_validated = False
                        st.session_state.available_models = []
            
            # 적용 버튼
            if apply_button and user_key_input:
                with st.spinner("🔄 API 키 적용 중..."):
                    is_valid, message, models = validate_api_key(user_key_input)
                    
                    if is_valid:
                        st.session_state.user_api_key = user_key_input
                        st.session_state.api_key_validated = True
                        st.session_state.available_models = models
                        
                        # RAG 파이프라인 재초기화 강제
                        st.session_state.rag_pipeline = None
                        st.session_state.model_type = None
                        
                        st.success("✅ API 키가 적용되었습니다!")
                        
                        if models:
                            st.info(f"💡 아래에서 사용할 모델을 선택하세요. ({len(models)}개 사용 가능)")
                    else:
                        st.error(message)
            
            # API 키 입력 가이드
            with st.expander("📖 API 키 발급 방법"):
                st.markdown("""
                1. [OpenAI Platform](https://platform.openai.com/api-keys) 접속
                2. 로그인 후 "Create new secret key" 클릭
                3. 생성된 키를 복사하여 위에 붙여넣기
                
                **주의사항:**
                - API 키는 안전하게 보관하세요
                - 무료 크레딧이 소진되면 사용 불가
                - 사용량에 따라 요금이 부과될 수 있습니다
                
                **모델별 가격 (1M 토큰 기준):**
                - gpt-4o: $2.50 (입력) / $10.00 (출력)
                - gpt-4o-mini: $0.15 (입력) / $0.60 (출력)
                - gpt-3.5-turbo: $0.50 (입력) / $1.50 (출력)
                """)
        
        else:
            # 서버 키 사용 중
            if has_env_key:
                st.info("ℹ️ 서버에 설정된 API 키를 사용합니다.")
                
                # 서버 키로 사용 가능한 모델 조회 (최초 1회)
                if not st.session_state.available_models:
                    with st.spinner("🔄 사용 가능한 모델 조회 중..."):
                        success, models, error = get_available_models(config.OPENAI_API_KEY)
                        if success:
                            st.session_state.available_models = models
            
            # 사용자가 입력한 키 초기화
            if st.session_state.user_api_key:
                st.session_state.user_api_key = None
                st.session_state.rag_pipeline = None
                st.session_state.model_type = None
        
        st.markdown("---")
        
        # ===== 🤖 모델 설정 =====
        st.markdown("### 🤖 모델 설정")
        
        can_use_gpt = has_env_key or (use_custom_key and st.session_state.api_key_validated)
        
        model_options = ["API 모델 (GPT)", "로컬 모델 (GGUF)"]
        
        if not can_use_gpt:
            st.warning("⚠️ API 키를 입력해야 GPT 모델을 사용할 수 있습니다.")
            default_index = 1
        else:
            default_index = 0
        
        model_type = st.selectbox(
            "생성 모델 선택",
            options=model_options,
            index=default_index,
            help="OpenAI API 또는 로컬 GGUF 모델 선택"
        )
        
        # ===== GPT 모델 상세 선택 =====
        selected_gpt_model = None
        
        if model_type == "API 모델 (GPT)" and can_use_gpt:
            available_models = st.session_state.available_models
            
            if available_models:
                # 모델 선택 UI
                st.markdown("#### 📋 GPT 모델 선택")
                
                # 모델 설명
                model_descriptions = {
                    'o3': '🌟 o3 시리즈 (최첨단 추론 모델)',
                    'o3-mini': '🌟 o3-mini (경량 추론 모델)',
                    'o1': '🧠 o1 시리즈 (고급 추론 모델)',
                    'o1-mini': '🧠 o1-mini (경량 추론 모델)',
                    'o1-preview': '🧪 o1 프리뷰 (베타)',
                    'gpt-5': '⚡ GPT-5 (차세대 모델)',
                    'gpt-5-turbo': '⚡ GPT-5 Turbo (고속)',
                    'gpt-4o': '🚀 GPT-4o (가장 강력)',
                    'gpt-4o-mini': '⚡ GPT-4o-mini (빠르고 저렴, 권장)',
                    'gpt-4-turbo': '💎 GPT-4 Turbo (고성능)',
                    'gpt-4': '🏆 GPT-4 (높은 품질)',
                    'gpt-3.5-turbo': '💰 GPT-3.5 Turbo (가성비)',
                    'gpt-3.5': '💰 GPT-3.5 (기본)'
                }
                
                # format 함수: 모델명으로 설명 찾기
                def get_model_display(model_name):
                    # 정확히 매칭되는 설명 찾기
                    if model_name in model_descriptions:
                        return f"{model_descriptions[model_name]} - {model_name}"
                    
                    # 부분 매칭 (예: gpt-4o-2024-11-20 → gpt-4o)
                    for key in model_descriptions.keys():
                        if model_name.startswith(key):
                            return f"{model_descriptions[key]} - {model_name}"
                    
                    # 매칭 안되면 모델명만
                    return model_name
                
                # 기본값 설정
                if st.session_state.selected_gpt_model not in available_models:
                    # 우선순위: gpt-4o-mini > gpt-3.5-turbo > 첫번째 모델
                    if 'gpt-4o-mini' in available_models:
                        st.session_state.selected_gpt_model = 'gpt-4o-mini'
                    elif 'gpt-3.5-turbo' in available_models:
                        st.session_state.selected_gpt_model = 'gpt-3.5-turbo'
                    else:
                        st.session_state.selected_gpt_model = available_models[0]
                
                # 모델 선택
                selected_gpt_model = st.selectbox(
                    "사용할 모델",
                    options=available_models,
                    index=available_models.index(st.session_state.selected_gpt_model),
                    format_func=get_model_display,
                    help="API 키로 사용 가능한 모델 중 선택하세요"
                )
                
                # 선택 저장
                st.session_state.selected_gpt_model = selected_gpt_model
                
                # 선택한 모델 정보 표시
                # 설명 찾기
                display_desc = "설명 없음"
                for key, desc in model_descriptions.items():
                    if selected_gpt_model.startswith(key):
                        display_desc = desc
                        break
                
                st.markdown(f"""
                <div class="model-info">
                    🎯 <b>선택된 모델</b><br>
                    • {selected_gpt_model}<br>
                    • {display_desc}
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ 사용 가능한 모델을 조회하지 못했습니다.")
                st.info("💡 '검증' 버튼을 눌러 모델 목록을 조회하세요.")
                
                # 기본값 사용
                selected_gpt_model = "gpt-4o-mini"
        
        elif model_type == "로컬 모델 (GGUF)":
            # GGUF 모델 정보 표시
            st.markdown("""
            <div class="model-info">
                🖥️ <b>Llama-3-Ko-8B (GGUF)</b><br>
                • T4 GPU 가속<br>
                • 로컬 실행 (무료)<br>
                • 초기 로딩 시간 소요<br>
                • 35개 레이어 GPU 사용
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== 🔍 검색 설정 =====
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
            help="검색할 문서 개수"
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
        
        # ===== 🛠️ 개발자 옵션 =====
        st.markdown("### 🛠️ 개발자 옵션")
        
        show_routing = st.toggle(
            "🔍 라우팅 정보 표시",
            value=False,
            help="Router의 판단 과정을 표시 (디버깅용)"
        )
        st.session_state.show_routing_info = show_routing
        
        st.markdown("---")
        
        # ===== 💬 대화 관리 =====
        st.markdown("### 💬 대화 관리")
        
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.conv_manager.clear()
            st.rerun()
        
        if st.button("💾 대화 다운로드", use_container_width=True):
            if len(st.session_state.conv_manager) > 0:
                json_str = st.session_state.conv_manager.export_to_json()
                
                st.download_button(
                    label="📥 JSON 다운로드",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # ===== 📊 통계 =====
        st.markdown("### 📊 통계")
        stats = st.session_state.conv_manager.get_statistics()

        st.metric("총 대화 수", stats.get('total', 0))
        
        # 현재 설정 표시
        st.markdown("---")
        st.markdown("### 📋 현재 설정")
        st.text(f"모델: {model_type}")
        if model_type == "API 모델 (GPT)" and selected_gpt_model:
            st.text(f"GPT 모델: {selected_gpt_model}")
        st.text(f"검색 모드: {search_mode}")
        st.text(f"Re-ranker: {'✅ ON' if use_reranker else '❌ OFF'}")
        st.text(f"실제 모드: {actual_search_mode}")
        st.text(f"Top-K: {top_k}")
        if search_mode == "hybrid":
            st.text(f"Alpha: {alpha}")
        st.text(f"Router Info: {'✅ ON' if show_routing else '❌ OFF'}")
    
    # ===== RAG 파이프라인 초기화 =====
    # 모델 타입이 변경되었거나 GPT 모델이 변경되었거나 파이프라인이 없으면 재초기화
    need_reinit = (
        st.session_state.rag_pipeline is None or 
        st.session_state.model_type != model_type or
        (model_type == "API 모델 (GPT)" and 
         selected_gpt_model and 
         hasattr(st.session_state.rag_pipeline, 'model') and
         st.session_state.rag_pipeline.model != selected_gpt_model)
    )
    
    if need_reinit:
        with st.spinner(f"🔄 {model_type} 초기화 중... (GGUF 모델은 1~2분 소요될 수 있습니다)"):
            rag, error, rag_type = initialize_rag(
                model_type, 
                _user_api_key=st.session_state.user_api_key,
                gpt_model_name=selected_gpt_model
            )
            
            if error:
                st.error(f"❌ RAG 파이프라인 초기화 실패")
                
                with st.expander("🔍 에러 상세 정보"):
                    st.code(error)
                
                st.info("""
                ### 💡 해결 방법
                
                **GGUF 모델 실패 시:**
                1. llama-cpp-python 설치 확인:
```bash
pip install llama-cpp-python
```
                
                2. GGUF 모델 파일 확인:
                   - config.yaml의 GGUF_MODEL_PATH 또는
                   - MODEL_HUB_REPO 설정 확인
                
                3. GPU 메모리 부족 시:
                   - n_gpu_layers 값 감소 (35 → 20)
                
                **API 모델 실패 시:**
                1. ChromaDB가 생성되었는지 확인:
```bash
python main.py --step embed
```
                
                2. OpenAI API 키 확인:
```bash
# .env 파일
OPENAI_API_KEY=your-key-here
```
                
                3. 필요한 패키지 설치:
```bash
pip install rank-bm25 sentence-transformers
```
                """)
                return
            
            st.session_state.rag_pipeline = rag
            st.session_state.model_type = model_type
            
            # API 키 및 모델 사용 정보 표시
            if st.session_state.user_api_key:
                st.success(f"✅ {rag_type} 준비 완료! (사용자 API 키)")
            else:
                st.success(f"✅ {rag_type} 준비 완료!")
    
    # ===== 대화 히스토리 표시 =====
    st.markdown("---")
    
    if len(st.session_state.conv_manager) == 0:
        st.info("""
        ### 👋 환영합니다!
        
        공공기관 사업제안서에 대해 질문해보세요.
        
        **예시 질문:**
        - "안녕하세요" (검색 안 함)
        - "데이터 표준화 요구사항은 무엇인가요?" (검색 수행)
        - "보안 관련 요구사항을 설명해주세요" (검색 수행)
        - "고마워요" (검색 안 함)
        """)
    
    # 기존 메시지 표시
    for msg in st.session_state.conv_manager.get_ui_history():
        display_message(
            role=msg['role'],
            content=msg['content'],
            sources=msg.get('sources'),
            usage=msg.get('usage'),
            search_mode=msg.get('search_mode'),
            used_retrieval=msg.get('used_retrieval'),
            routing_info=msg.get('routing_info')
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

        # 답변 생성
        with st.spinner("🤔 답변 생성 중..."):
            result = generate_answer(
                query=user_input,
                top_k=top_k,
                search_mode=actual_search_mode,
                alpha=alpha
            )
        
        # 어시스턴트 메시지 추가
        st.session_state.conv_manager.add_message(
            user_msg=user_input,
            ai_msg=result['answer'],
            query_type=result.get('query_type', 'unknown'),
            sources=result.get('sources', []),
            usage=result.get('usage', {}),
            search_mode=result.get('search_mode'),
            used_retrieval=result.get('used_retrieval', False),
            routing_info=result.get('routing_info')
        )
        
        # 화면 새로고침
        st.rerun()


if __name__ == "__main__":
    main()