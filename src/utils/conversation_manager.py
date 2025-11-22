# src/utils/conversation_manager.py

"""
대화 히스토리 관리자 (메모리 기반)

기능:
- UI 표시용 / 분석용 히스토리 분리
- 전체 대화 저장 (greeting, thanks, document, out_of_scope)
- JSON 내보내기
- 통계 기능
"""

from datetime import datetime
from typing import List, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    대화 히스토리 관리 (메모리 기반)
    
    Streamlit session_state와 함께 사용:
    - UI 히스토리: Streamlit 메시지 형식
    - DB 히스토리: 분석/저장용 형식
    """
    
    def __init__(self):
        """초기화"""
        self.ui_history: List[Dict] = []   # Streamlit 표시용
        self.db_history: List[Dict] = []   # 분석/저장용
        
        logger.info("💬 ConversationManager 초기화 완료")
    
    def add_message(
        self,
        user_msg: str,
        ai_msg: str,
        query_type: str,
        sources: Optional[List] = None,
        usage: Optional[Dict] = None,
        search_mode: Optional[str] = None,
        used_retrieval: bool = False,
        routing_info: Optional[Dict] = None
    ):
        """
        메시지 추가 (전체 저장)
        
        Args:
            user_msg: 사용자 질문
            ai_msg: AI 답변
            query_type: 질문 유형 (greeting/thanks/document/out_of_scope)
            sources: 참고 문서 리스트
            usage: 토큰 사용량
            search_mode: 검색 모드
            used_retrieval: 검색 사용 여부
            routing_info: 라우팅 정보
        """
        timestamp = datetime.now()
        
        # ===== UI 히스토리 (Streamlit 메시지 형식) =====
        # 사용자 메시지
        self.ui_history.append({
            'role': 'user',
            'content': user_msg,
            'timestamp': timestamp
        })
        
        # AI 메시지
        self.ui_history.append({
            'role': 'assistant',
            'content': ai_msg,
            'sources': sources or [],
            'usage': usage or {},
            'search_mode': search_mode,
            'used_retrieval': used_retrieval,
            'routing_info': routing_info,
            'type': query_type,  # 분석용 추가
            'timestamp': timestamp
        })
        
        # ===== DB 히스토리 (분석용) =====
        self.db_history.append({
            'user': user_msg,
            'assistant': ai_msg,
            'type': query_type,
            'timestamp': timestamp.isoformat(),
            'sources_count': len(sources) if sources else 0,
            'used_retrieval': used_retrieval,
            'search_mode': search_mode,
            'routing_info': routing_info
        })
        
        logger.info(f"💾 대화 저장: {query_type} - {user_msg[:30]}...")
    
    def get_ui_history(self) -> List[Dict]:
        """
        UI 표시용 히스토리 반환 (Streamlit 형식)
        
        Returns:
            Streamlit 메시지 리스트
        """
        return self.ui_history
    
    def get_db_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """
        분석/저장용 히스토리 반환
        
        Args:
            last_n: 최근 N개만 반환 (None이면 전체)
        
        Returns:
            대화 기록 리스트
        """
        if last_n:
            return self.db_history[-last_n:]
        return self.db_history
    
    def get_history_by_type(self, query_type: str) -> List[Dict]:
        """
        특정 질문 유형만 필터링
        
        Args:
            query_type: 'greeting', 'thanks', 'document', 'out_of_scope'
        
        Returns:
            필터링된 대화 리스트
        """
        return [
            msg for msg in self.db_history 
            if msg['type'] == query_type
        ]
    
    def get_statistics(self) -> Dict[str, int]:
        """
        질문 유형별 통계
        
        Returns:
            {'greeting': 5, 'document': 20, ...}
        """
        from collections import Counter
        
        types = [msg['type'] for msg in self.db_history]
        stats = dict(Counter(types))
        
        # 총 대화 수 추가
        stats['total'] = len(self.db_history)
        
        return stats
    
    def export_to_json(self) -> str:
        """
        JSON 형식으로 내보내기
        
        Returns:
            JSON 문자열
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_conversations': len(self.db_history),
            'statistics': self.get_statistics(),
            'conversations': self.db_history
        }
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def clear(self):
        """히스토리 초기화"""
        self.ui_history = []
        self.db_history = []
        logger.info("🗑️ 대화 히스토리 초기화")
    
    def __len__(self):
        """대화 개수 (사용자 질문 기준)"""
        return len(self.db_history)
    
    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"ConversationManager("
            f"total={stats.get('total', 0)}, "
            f"document={stats.get('document', 0)}, "
            f"greeting={stats.get('greeting', 0)}, "
            f"thanks={stats.get('thanks', 0)}, "
            f"out_of_scope={stats.get('out_of_scope', 0)})"
        )


# ===== 테스트 코드 =====
if __name__ == "__main__":
    # 테스트
    manager = ConversationManager()
    
    # 대화 추가
    manager.add_message(
        user_msg="안녕하세요",
        ai_msg="안녕하세요! 무엇을 도와드릴까요?",
        query_type="greeting"
    )
    
    manager.add_message(
        user_msg="예산이 얼마인가요?",
        ai_msg="예산은 5억원입니다.",
        query_type="document",
        sources=[{'content': '예산: 5억원', 'score': 0.95}],
        used_retrieval=True,
        search_mode="hybrid_rerank"
    )
    
    manager.add_message(
        user_msg="고마워요",
        ai_msg="천만에요! 언제든 질문하세요.",
        query_type="thanks"
    )
    
    # 통계 출력
    print("\n===== 통계 =====")
    print(manager.get_statistics())
    
    # 히스토리 출력
    print("\n===== DB 히스토리 =====")
    for msg in manager.get_db_history():
        print(f"{msg['type']}: {msg['user'][:20]}...")
    
    # JSON 내보내기
    print("\n===== JSON Export =====")
    print(manager.export_to_json())
    
    # Representation
    print("\n===== Manager Info =====")
    print(manager)