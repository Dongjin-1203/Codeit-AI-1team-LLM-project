"""
RAGAS Synthetic Dataset 생성 (간단 버전)

Knowledge Graph 없이 직접 질문-답변 쌍 생성
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
import json
from datetime import datetime
from datasets import Dataset
import random

from utils.rag_config import RAGConfig

config = RAGConfig()

def load_documents_from_chromadb() -> List[Document]:
    """ChromaDB에서 문서 로드"""
    print("\n" + "="*80)
    print("Step 1: ChromaDB에서 문서 로드")
    print("="*80)
    
    chroma_client = chromadb.PersistentClient(path=config.DB_DIRECTORY)
    collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
    
    print(f"✅ ChromaDB 연결 성공")
    
    results = collection.get(include=['documents', 'metadatas'])
    total_chunks = len(results['ids'])
    
    print(f"📊 총 {total_chunks}개의 청크 발견")
    
    documents = []
    for doc_id, text, metadata in zip(results['ids'], results['documents'], results['metadatas']):
        doc = Document(
            page_content=text,
            metadata={
                'id': doc_id,
                'filename': metadata.get('filename', 'unknown'),
                **metadata
            }
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)}개 Document 생성 완료")
    return documents


def generate_qa_from_document(doc: Document, llm: ChatOpenAI, num_questions: int = 2) -> List[dict]:
    """
    단일 문서에서 질문-답변 쌍 생성
    
    Args:
        doc: 문서
        llm: LLM
        num_questions: 생성할 질문 수
    
    Returns:
        질문-답변 쌍 리스트
    """
    
    prompt = f"""다음 문서를 읽고 {num_questions}개의 질문-답변 쌍을 생성하세요.

문서:
{doc.page_content[:2000]}

요구사항:
1. 문서 내용에 기반한 실제 답변 가능한 질문만 생성
2. 다양한 유형의 질문 (단순, 추론, 비교 등)
3. 답변은 문서에서 직접 찾을 수 있어야 함

다음 JSON 형식으로 응답하세요:
{{
    "qa_pairs": [
        {{
            "question": "질문 내용",
            "answer": "정답",
            "context": "관련 문맥 (문서의 관련 부분)"
        }}
    ]
}}

JSON만 출력하고 다른 설명은 하지 마세요."""

    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        return result.get('qa_pairs', [])
    except Exception as e:
        print(f"   ⚠️ 생성 실패: {str(e)[:50]}")
        return []


def generate_synthetic_testset(documents: List[Document], testset_size: int = 100) -> Dataset:
    """
    Synthetic Dataset 생성 (간단 버전)
    """
    print("\n" + "="*80)
    print("Step 2: Synthetic Dataset 생성")
    print("="*80)
    
    print(f"📊 생성 설정:")
    print(f"   목표 개수: {testset_size}개")
    print(f"   문서 수: {len(documents)}개")
    
    # LLM 초기화
    llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0.7)
    print(f"✅ LLM 초기화: {config.LLM_MODEL_NAME}")
    
    # 문서 샘플링
    sampled_docs = random.sample(documents, min(len(documents), testset_size))
    
    all_qa_pairs = []
    
    print(f"\n⏳ 생성 시작...")
    
    for i, doc in enumerate(sampled_docs):
        if len(all_qa_pairs) >= testset_size:
            break
        
        # 문서당 1-2개 질문 생성
        qa_pairs = generate_qa_from_document(doc, llm, num_questions=1)
        all_qa_pairs.extend(qa_pairs)
        
        if (i + 1) % 10 == 0:
            print(f"   진행: {i+1}/{len(sampled_docs)} ({len(all_qa_pairs)}개 생성)")
    
    # testset_size만큼 자르기
    all_qa_pairs = all_qa_pairs[:testset_size]
    
    print(f"\n✅ 생성 완료: {len(all_qa_pairs)}개")
    
    # Dataset 형식으로 변환
    dataset_dict = {
        'user_input': [qa['question'] for qa in all_qa_pairs],
        'reference': [qa['answer'] for qa in all_qa_pairs],
        'retrieved_contexts': [[qa['context']] for qa in all_qa_pairs]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def analyze_testset(testset: Dataset):
    """결과 분석"""
    print("\n" + "="*80)
    print("Step 3: 생성 결과 분석")
    print("="*80)
    
    df = testset.to_pandas()
    
    print(f"📊 기본 통계:")
    print(f"   총 개수: {len(df)}개")
    
    if 'user_input' in df.columns:
        print(f"   평균 질문 길이: {df['user_input'].str.len().mean():.0f}자")
    
    if 'reference' in df.columns:
        print(f"   평균 정답 길이: {df['reference'].str.len().mean():.0f}자")
    
    # 샘플
    print(f"\n📝 랜덤 샘플 3개:")
    for idx in random.sample(range(len(df)), min(3, len(df))):
        row = df.iloc[idx]
        print(f"\n[샘플 {idx+1}]")
        print(f"질문: {row['user_input'][:100]}...")
        print(f"정답: {row['reference'][:100]}...")


def save_testset(testset: Dataset, output_path: str = "src/evaluation/results/synthetic_testset_100.json"):
    """저장"""
    print("\n" + "="*80)
    print("Step 4: 테스트셋 저장")
    print("="*80)
    
    testset.to_json(output_path)
    print(f"✅ JSON 저장: {output_path}")
    
    csv_path = output_path.replace('.json', '.csv')
    testset.to_pandas().to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장: {csv_path}")


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("RAGAS Synthetic Dataset 생성 (간단 버전)")
    print("="*80)
    
    try:
        # 문서 로드
        documents = load_documents_from_chromadb()
        
        # 생성 개수 설정
        testset_size = 100
        response = input(f"\n생성할 개수 (기본 {testset_size}): ")
        if response.strip():
            testset_size = int(response)
        
        # 생성
        testset = generate_synthetic_testset(documents, testset_size)
        
        # 분석
        analyze_testset(testset)
        
        # 저장
        save_testset(testset)
        
        print("\n" + "="*80)
        print("✅ 완료!")
        print("="*80)
        print(f"\n📁 생성된 파일:")
        print(f"   - synthetic_testset_100.json")
        print(f"   - synthetic_testset_100.csv")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()