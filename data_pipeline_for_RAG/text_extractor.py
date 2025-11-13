"""
PDF 및 HWP 파일에서 텍스트 추출
"""

import os
import zlib
import struct
from pypdf import PdfReader
import olefile


class TextExtractor:
    """PDF 및 HWP 파일에서 텍스트 추출"""
    
    @staticmethod
    def extract_pdf(filepath: str) -> str:
        """
        PDF 파일에서 텍스트 추출
        
        Args:
            filepath: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            reader = PdfReader(filepath)
            page_texts = [
                page.extract_text() 
                for page in reader.pages 
                if page.extract_text()
            ]
            return "\n\n".join(page_texts)
        except Exception as e:
            return f"[PDF 추출 실패: {e}]"
    
    @staticmethod
    def extract_hwp(filepath: str) -> str:
        """
        HWP 파일에서 텍스트 추출
        
        Args:
            filepath: HWP 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            f = olefile.OleFileIO(filepath)
            dirs = f.listdir()
            
            # HWP 5.0 파일 검증
            if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
                return "[HWP 추출 실패: 유효한 HWP 5.0 파일이 아님]"
            
            # 압축 여부 확인
            header = f.openstream("FileHeader")
            header_data = header.read()
            is_compressed = (header_data[36] & 1) == 1
            
            # 섹션 번호 정렬
            nums = [
                int(d[1][len("Section"):]) 
                for d in dirs 
                if d[0] == "BodyText"
            ]
            sections = [f"BodyText/Section{x}" for x in sorted(nums)]
            
            # 텍스트 추출
            text = ""
            for section in sections:
                bodytext = f.openstream(section)
                data = bodytext.read()
                
                # 압축 해제
                if is_compressed:
                    unpacked_data = zlib.decompress(data, -15)
                else:
                    unpacked_data = data
                
                # 레코드 파싱
                i = 0
                size = len(unpacked_data)
                while i < size:
                    header = struct.unpack_from("<I", unpacked_data, i)[0]
                    rec_type = header & 0x3ff
                    rec_len = (header >> 20) & 0xfff
                    
                    # 텍스트 레코드 (타입 67)
                    if rec_type == 67:
                        rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                        text += rec_data.decode('utf-16', errors='ignore')
                    
                    i += 4 + rec_len
            
            f.close()
            return text
            
        except Exception as e:
            return f"[HWP 추출 실패: {e}]"
    
    @staticmethod
    def extract(filepath: str, file_format: str) -> str:
        """
        파일 형식에 따라 텍스트 추출
        
        Args:
            filepath: 파일 경로
            file_format: 파일 형식 ('pdf' 또는 'hwp')
            
        Returns:
            추출된 텍스트
        """
        if not os.path.exists(filepath):
            return "[추출 실패: 파일 없음]"
        
        file_format = file_format.lower()
        
        if file_format == 'pdf':
            return TextExtractor.extract_pdf(filepath)
        elif file_format == 'hwp':
            return TextExtractor.extract_hwp(filepath)
        else:
            return f"[추출 실패: 알 수 없는 파일 형식 ({file_format})]"