# src/router/query_router.py

import logging

logger = logging.getLogger(__name__)

class QueryRouter:
    """Query를 RAG vs Direct로 라우팅"""
    
    def __init__(self):
        # 키워드 정의
        self.greeting_keywords = [
            "안녕", "hi", "hello", "반가워", "처음"
        ]
        
        self.thanks_keywords = [
            "고마워", "감사", "thanks", "고맙"
        ]
        
        self.document_keywords = [
            # 돈 관련
            "예산", "비용", "금액", "원", "만원", "억",
            # 일정 관련
            "기한", "마감", "언제", "기간", "납기",
            # 문서 관련
            "요구사항", "제출", "서류", "양식", "평가",
            # 조직 관련
            "발주", "기관", "담당자", "연락처",
            # 사업 관련
            "사업명", "과업", "범위", "목적"
        ]
    
    def classify(self, query: str) -> dict:
        query_lower = query.lower()
        
        # 짧은 질문일 때만 인사/감사 체크
        if len(query) < 20:  # ← is_short 대신 직접 체크
            if any(kw in query_lower for kw in self.thanks_keywords):
                return {
                    'type': 'thanks',
                    'confidence': 0.9,
                    'reason': '감사 인사 감지'
                }

            elif any(kw in query_lower for kw in self.greeting_keywords):
                return {
                    'type': 'greeting',
                    'confidence': 0.9,
                    'reason': '인사 감지'
                }

        # 문서 관련 판별
        if any(kw in query_lower for kw in self.document_keywords):
            return {
                'type': 'document',
                'confidence': 0.85,
                'reason': '문서 키워드 감지'
            }
        
        # 3. 기본값
        return {
            'type': 'out_of_scope',
            'confidence': 0.5,
            'reason': 'RFP 키워드 없음'
        }