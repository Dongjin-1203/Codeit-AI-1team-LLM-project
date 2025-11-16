"""
ChromaDB ID 매핑 스크립트

이 스크립트는 ChromaDB에 저장된 실제 문서 ID를 확인하고,
test_queries.json(평가용 데이터셋)의 doc_001, doc_002 형식을 실제 ID로 매핑합니다.

사용법:
1. 이 스크립트를 프로젝트 폴더에 복사
2. ChromaDB 설정에 맞게 수정
3. python map_chromadb_ids.py 실행
"""

import json
import chromadb
from chromadb.config import Settings
from typing import Dict, List
import pandas as pd


class ChromaDBIDMapper:
    """ChromaDB ID 매핑 클래스"""
    
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "rag_documents"):
        """
        Args:
            chroma_db_path: ChromaDB 저장 경로
            collection_name: Collection 이름
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        print(f"✅ ChromaDB 연결 성공")
        print(f"   경로: {chroma_db_path}")
        print(f"   컬렉션: {collection_name}")
    
    def get_all_documents(self) -> Dict:
        """ChromaDB에서 모든 문서 정보 가져오기"""
        results = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        print(f"\n📊 총 {len(results['ids'])}개의 문서/청크 발견")
        return results
    
    def create_filename_to_ids_mapping(self, results: Dict) -> Dict[str, List[str]]:
        """
        파일명 -> ChromaDB ID 리스트 매핑 생성
        
        Returns:
            {
                "한영대학_...hwp": ["chroma_id_1", "chroma_id_2", ...],
                ...
            }
        """
        filename_map = {}
        
        for i, metadata in enumerate(results['metadatas']):
            # 메타데이터에서 파일명 추출 (키 이름은 프로젝트마다 다를 수 있음)
            # 일반적으로 'filename', 'source', 'file_name' 등
            filename = metadata.get('filename') or metadata.get('source') or metadata.get('file_name')
            
            if filename:
                # 파일명에서 경로 제거 (있다면)
                filename = filename.split('/')[-1].split('\\')[-1]
                
                if filename not in filename_map:
                    filename_map[filename] = []
                filename_map[filename].append(results['ids'][i])
        
        return filename_map
    
    def display_mapping_table(self, filename_map: Dict[str, List[str]]):
        """매핑 테이블을 보기 좋게 출력"""
        print("\n" + "="*80)
        print("파일명 -> ChromaDB ID 매핑")
        print("="*80)
        
        for filename, ids in sorted(filename_map.items()):
            print(f"\n📄 {filename}")
            print(f"   청크 개수: {len(ids)}")
            print(f"   ID 샘플: {ids[0] if ids else 'N/A'}")
    
    def create_test_query_mapping(self, filename_map: Dict[str, List[str]]) -> Dict:
        """
        test_queries.json의 document_mapping에 맞는 매핑 생성
        
        Returns:
            {
                "doc_001": {
                    "filename": "한영대학_...hwp",
                    "chunk_ids": ["chroma_id_1", "chroma_id_2", ...]
                },
                ...
            }
        """
        # 원본 document_mapping (test_queries.json에서)
        original_mapping = {
            "doc_001": "한영대학_한영대학교 특성화 맞춤형 교육환경 구축.hwp",
            "doc_002": "한국연구재단_2024년 대학산학협력활동 실태조사 시스템.hwp",
            "doc_003": "한국생산기술연구원_EIP3.0 고압가스 안전관리 시스템.hwp",
            "doc_004": "인천광역시_도시계획위원회 통합관리시스템.hwp",
            "doc_005": "광주문화재단_광주문화예술통합플랫폼.hwp",
            "doc_006": "나노종합기술원_스마트 팹 서비스.hwp"
        }
        
        new_mapping = {}
        
        for doc_id, target_filename in original_mapping.items():
            # 파일명으로 ChromaDB ID 찾기 (부분 매칭)
            matched_ids = []
            matched_filename = None
            
            for filename, chunk_ids in filename_map.items():
                # 파일명의 핵심 부분이 포함되어 있는지 확인
                # 예: "한영대학교" in "한영대학_한영대학교 특성화..."
                if self._fuzzy_match(target_filename, filename):
                    matched_ids = chunk_ids
                    matched_filename = filename
                    break
            
            new_mapping[doc_id] = {
                "target_filename": target_filename,
                "matched_filename": matched_filename,
                "chunk_ids": matched_ids,
                "num_chunks": len(matched_ids)
            }
        
        return new_mapping
    
    def _fuzzy_match(self, target: str, candidate: str) -> bool:
        """파일명 퍼지 매칭"""
        # 간단한 퍼지 매칭: 주요 키워드가 포함되어 있는지 확인
        target_keywords = target.replace('.hwp', '').replace('.pdf', '').split('_')
        candidate_lower = candidate.lower()
        
        # 최소 2개 이상의 키워드가 매칭되면 OK
        matches = sum(1 for keyword in target_keywords if keyword.lower() in candidate_lower)
        return matches >= 2
    
    def update_test_queries_json(self, 
                                  input_path: str = "test_queries_improved.json",
                                  output_path: str = "test_queries_mapped.json"):
        """
        test_queries.json의 doc_XXX ID를 실제 ChromaDB ID로 업데이트
        """
        # ChromaDB 데이터 가져오기
        results = self.get_all_documents()
        filename_map = self.create_filename_to_ids_mapping(results)
        self.display_mapping_table(filename_map)
        
        # 매핑 생성
        doc_mapping = self.create_test_query_mapping(filename_map)
        
        print("\n" + "="*80)
        print("Document ID 매핑 결과")
        print("="*80)
        for doc_id, info in doc_mapping.items():
            status = "✅" if info['matched_filename'] else "❌"
            print(f"{status} {doc_id}:")
            print(f"   목표: {info['target_filename']}")
            print(f"   매칭: {info['matched_filename']}")
            print(f"   청크: {info['num_chunks']}개")
        
        # test_queries.json 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            test_queries = json.load(f)
        
        # ID 교체
        updated_count = 0
        for test_case in test_queries['test_cases']:
            old_ids = test_case['ground_truth']['relevant_doc_ids']
            new_ids = []
            
            for old_id in old_ids:
                if old_id in doc_mapping:
                    # 해당 문서의 모든 청크 ID 추가
                    chunk_ids = doc_mapping[old_id]['chunk_ids']
                    if chunk_ids:
                        new_ids.extend(chunk_ids)
                        updated_count += 1
            
            # 업데이트 (원본도 유지)
            test_case['ground_truth']['original_doc_ids'] = old_ids
            test_case['ground_truth']['relevant_doc_ids'] = new_ids
        
        # 메타데이터 업데이트
        test_queries['metadata']['mapping_info'] = {
            'chromadb_path': self.chroma_db_path,
            'collection_name': self.collection_name,
            'total_documents': len(doc_mapping),
            'successfully_mapped': sum(1 for v in doc_mapping.values() if v['chunk_ids']),
            'doc_to_chroma_mapping': doc_mapping
        }
        
        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_queries, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 업데이트 완료!")
        print(f"   입력: {input_path}")
        print(f"   출력: {output_path}")
        print(f"   업데이트된 쿼리: {updated_count}개")
        
        return test_queries


def main():
    """메인 실행 함수"""
    
    print("\n" + "="*80)
    print("ChromaDB ID 매핑 스크립트")
    print("="*80)
    
    # ===== 여기를 수정하세요! =====
    CHROMA_DB_PATH = "./chroma_db"  # ChromaDB 저장 경로
    COLLECTION_NAME = "rag_documents"  # Collection 이름
    INPUT_JSON = "src/evaluation/test_queries.json"  # 입력 파일
    OUTPUT_JSON = "src/evaluation/test_queries_mapped.json"  # 출력 파일
    # =============================
    
    try:
        # 매퍼 생성
        mapper = ChromaDBIDMapper(
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        # 매핑 실행
        updated_queries = mapper.update_test_queries_json(
            input_path=INPUT_JSON,
            output_path=OUTPUT_JSON
        )
        
        print("\n🎉 매핑 완료! 다음 파일을 사용하세요:")
        print(f"   {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("\n💡 확인사항:")
        print("   1. ChromaDB 경로가 올바른가요?")
        print("   2. Collection 이름이 정확한가요?")
        print("   3. test_queries_improved.json 파일이 있나요?")
        raise


if __name__ == "__main__":
    main()