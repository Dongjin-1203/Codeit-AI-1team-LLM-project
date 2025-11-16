import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from langchain_core.documents import Document
from typing import List
from src.utils.rag_config import RAGConfig


def load_documents_from_chromadb() -> List[Document]:
    """
    ChromaDB에서 모든 문서를 가져와 LangChain Document 리스트로 변환
    
    Returns:
        List[Document]: RAGAS에서 사용할 Document 객체 리스트
    """
    # 1. RAG 설정 로드
    config = RAGConfig()
    
    # 2. ChromaDB 클라이언트 연결
    chroma_client = chromadb.PersistentClient(path=config.DB_DIRECTORY)
    
    # 3. Collection 가져오기
    collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
    
    print(f"✅ ChromaDB 연결 성공")
    print(f"   경로: {config.DB_DIRECTORY}")
    print(f"   Collection: {config.COLLECTION_NAME}")
    
    # 4. 모든 문서 가져오기
    results = collection.get(
        include=['documents', 'metadatas']  # embeddings는 불필요
    )
    
    total_chunks = len(results['ids'])
    print(f"📊 총 {total_chunks}개의 청크 발견")
    
    # 5. LangChain Document 객체로 변환
    documents = []
    
    for i, (doc_id, text, metadata) in enumerate(
        zip(results['ids'], results['documents'], results['metadatas'])
    ):
        # Document 객체 생성
        doc = Document(
            page_content=text,  # 문서 텍스트
            metadata={
                'id': doc_id,
                'filename': metadata.get('파일명', 'unknown'),
                'organization': metadata.get('발주 기관', 'unknown'),
                'chunk_index': i,
                **metadata  # 기존 메타데이터 모두 포함
            }
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)}개 Document 객체 생성 완료")
    
    # 6. 샘플 확인
    if documents:
        print(f"\n📝 샘플 Document:")
        print(f"   텍스트 길이: {len(documents[0].page_content)}자")
        print(f"   메타데이터: {list(documents[0].metadata.keys())}")
        print(f"   파일명 예시: {documents[0].metadata.get('filename', 'N/A')}")
    
    return documents


# def sample_documents(documents: List[Document], 
#                      sample_per_file: int = 5) -> List[Document]:
#     """
#     문서가 너무 많을 경우 샘플링 (선택사항)
    
#     Args:
#         documents: 전체 Document 리스트
#         sample_per_file: 파일당 샘플링할 청크 수
    
#     Returns:
#         샘플링된 Document 리스트
#     """
#     # 파일명별로 그룹화
#     file_groups = {}
#     for doc in documents:
#         filename = doc.metadata.get('filename', 'unknown')
#         if filename not in file_groups:
#             file_groups[filename] = []
#         file_groups[filename].append(doc)
    
#     # 각 파일에서 균등하게 샘플링
#     sampled = []
#     for filename, file_docs in file_groups.items():
#         # 파일의 청크 수가 적으면 모두 사용
#         if len(file_docs) <= sample_per_file:
#             sampled.extend(file_docs)
#         else:
#             # 균등 간격으로 샘플링
#             step = len(file_docs) // sample_per_file
#             sampled.extend([file_docs[i * step] for i in range(sample_per_file)])
    
#     print(f"\n📌 샘플링 결과:")
#     print(f"   전체: {len(documents)}개 → 샘플: {len(sampled)}개")
#     print(f"   파일 수: {len(file_groups)}개")
#     print(f"   파일당 평균: {len(sampled) / len(file_groups):.1f}개")
    
#     return sampled

# ===== 사용 예시 =====
if __name__ == "__main__":
    """
    문서 로드 테스트
    """
    print("\n" + "="*80)
    print("ChromaDB 문서 로드 테스트")
    print("="*80 + "\n")
    
    # 1. 모든 문서 로드
    documents = load_documents_from_chromadb()
    
    # 2. 샘플링 (선택사항)
    # 전체 사용하려면 이 부분 주석 처리
    # documents = sample_documents(documents, sample_per_file=5)
    
    # 3. 결과 확인
    print(f"\n{'='*80}")
    print("📊 최종 통계")
    print(f"{'='*80}")
    print(f"총 Document 수: {len(documents)}")
    print(f"평균 텍스트 길이: {sum(len(d.page_content) for d in documents) / len(documents):.0f}자")
    
    # 파일별 분포
    file_counts = {}
    for doc in documents:
        filename = doc.metadata.get('filename', 'unknown')
        file_counts[filename] = file_counts.get(filename, 0) + 1
    
    print(f"\n파일별 청크 분포 (상위 5개):")
    for filename, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {filename[:50]}: {count}개")
    
    print(f"\n✅ 문서 로드 완료! RAGAS 생성에 사용 가능합니다.")