"""
텍스트 정제 및 검증
특수문자 제거, 유효성 검사
"""

import re


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