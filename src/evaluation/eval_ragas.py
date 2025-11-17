import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
# RAGAS용 LLM 설정
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from generator.rag_pipeline import RAGPipeline
from utils.rag_config import RAGConfig

config = RAGConfig()

def load_testset(testset_path: str = "src/evaluation/results/synthetic_testset_100.json") -> Dataset:
    """테스트셋 로드"""
    print("\n" + "="*80)
    print("Step 1: 테스트셋 로드")
    print("="*80)
    
    try:
        testset = Dataset.from_json(testset_path)
        print(f"✅ Dataset 로드 성공")
    except:
        data_list = []
        with open(testset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        
        keys = data_list[0].keys()
        data_dict = {key: [item[key] for item in data_list] for key in keys}
        testset = Dataset.from_dict(data_dict)
        print(f"✅ JSONL 로드 성공")
    
    print(f"   파일: {testset_path}")
    print(f"   개수: {len(testset)}개")
    print(f"   컬럼: {testset.column_names}")
    
    return testset


def call_rag_with_retry(pipeline: RAGPipeline, question: str, max_retries: int = 3) -> dict:
    """재시도 로직이 있는 RAG 호출"""
    for attempt in range(max_retries):
        try:
            result = pipeline.generate_answer(question)
            
            if not result:
                raise ValueError("빈 결과 반환")
            
            if 'answer' not in result:
                raise ValueError(f"'answer' 키 없음")
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"   ⚠️ 시도 {attempt + 1} 실패, {wait_time}초 대기...")
                time.sleep(wait_time)
            else:
                raise


def run_rag_on_testset(testset: Dataset, pipeline: RAGPipeline, batch_size: int = 10) -> Dataset:
    """테스트셋의 모든 질문에 대해 RAG 실행"""
    print("\n" + "="*80)
    print("Step 2: RAG 파이프라인 실행")
    print("="*80)
    
    questions = []
    answers = []
    contexts_list = []
    references = []
    
    success_count = 0
    fail_count = 0
    
    print(f"⏳ {len(testset)}개 질문 처리 중...\n")
    
    for i, item in enumerate(testset):
        question = (item.get('user_input') or 
                   item.get('question') or 
                   item.get('query', ''))
        
        reference = (item.get('reference') or 
                    item.get('ground_truth') or 
                    item.get('answer', ''))
        
        questions.append(question)
        references.append(reference)
        
        try:
            result = call_rag_with_retry(pipeline, question, max_retries=3)
            
            answers.append(result.get('answer', ''))
            
            # Source에서 content 키 사용
            contexts = []
            for source in result.get('sources', []):
                text = source.get('content', '')
                if text:
                    contexts.append(text)
            
            contexts_list.append(contexts)
            success_count += 1
            
            if i == 0:
                print(f"📝 첫 번째 결과:")
                print(f"  질문: {question[:80]}...")
                print(f"  답변: {result.get('answer', '')[:80]}...")
                print(f"  문서 수: {len(contexts)}개\n")
            
        except Exception as e:
            print(f"❌ 질문 {i+1} 실패: {str(e)[:80]}")
            answers.append("")
            contexts_list.append([])
            fail_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"   진행: {i+1}/{len(testset)} (성공: {success_count}, 실패: {fail_count})")
        
        # Rate limit 대응
        if (i + 1) % batch_size == 0 and i + 1 < len(testset):
            print(f"   ⏸️  배치 대기 3초...")
            time.sleep(3)
    
    print(f"\n✅ RAG 실행 완료")
    print(f"   성공: {success_count}개, 실패: {fail_count}개")
    
    eval_dataset = Dataset.from_dict({
        'user_input': questions,
        'response': answers,
        'reference': references,
        'retrieved_contexts': contexts_list
    })
    
    return eval_dataset


def evaluate_with_ragas(testset: Dataset):
    """RAGAS로 평가 실행 (LLM 명시적 설정)"""
    print("\n" + "="*80)
    print("Step 3: RAGAS 평가 실행")
    print("="*80)
    
    # 🔥 RAGAS가 사용할 LLM 설정 (중요!)
    print("\n🔧 RAGAS LLM 설정 중...")
    
    evaluator_llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        temperature=0
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    print(f"   LLM: gpt-4o-mini")
    print(f"   Embeddings: text-embedding-3-small")
    
    # 메트릭 설정
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    ]
    
    print(f"\n📊 평가 메트릭:")
    for metric in metrics:
        print(f"   - {metric.name}")
    
    print(f"\n⏳ 평가 시작... (예상: 10-20분)")
    
    try:
        # RAGAS 평가 with LLM 설정
        result = evaluate(
            dataset=testset,
            metrics=metrics,
            llm=evaluator_llm,  # 🔥 LLM 명시!
            embeddings=embeddings  # 🔥 Embeddings 명시!
        )
        
        print(f"\n✅ 평가 완료!")
        return result
        
    except Exception as e:
        print(f"\n❌ 평가 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def extract_scores(result) -> dict:
    """점수 추출"""
    scores = {}
    
    try:
        df = result.to_pandas()
        for col in df.columns:
            if col not in ['user_input', 'response', 'reference', 'retrieved_contexts']:
                mean_val = df[col].mean()
                scores[col] = mean_val
        return scores
    except Exception as e:
        print(f"⚠️ 점수 추출 실패: {str(e)}")
        return {}


def print_results(result):
    """결과 출력"""
    print("\n" + "="*80)
    print("📊 RAGAS 평가 결과")
    print("="*80)
    
    scores = extract_scores(result)
    
    if not scores:
        print("\n⚠️ 점수 추출 실패")
        return
    
    print("\n🎯 메트릭별 점수:")
    print("-" * 80)
    
    valid_scores = []
    null_metrics = []
    
    for key, value in scores.items():
        if isinstance(value, float):
            if value != value:  # NaN
                print(f"  {key:25s}: NaN ❌")
                null_metrics.append(key)
            else:
                print(f"  {key:25s}: {value:.4f} ✅")
                valid_scores.append(value)
    
    # 결과 분석
    print("\n" + "="*80)
    
    if null_metrics:
        print(f"\n⚠️ {len(null_metrics)}개 메트릭 실패:")
        for metric in null_metrics:
            print(f"   - {metric}")
        print(f"\n가능한 원인:")
        print(f"   1. RAGAS 내부 LLM 호출 실패")
        print(f"   2. API Rate Limit")
        print(f"   3. 데이터 형식 문제")
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\n평균 점수 (유효한 메트릭만): {avg_score:.4f}")
        
        if avg_score >= 0.8:
            print("✅ 우수!")
        elif avg_score >= 0.7:
            print("⚠️ 양호")
        else:
            print("❌ 미흡")


def save_results(result, testset: Dataset, output_path: str = "src/evaluation/results/ragas_results.json"):
    """결과 저장"""
    print("\n" + "="*80)
    print("Step 4: 결과 저장")
    print("="*80)
    
    import pandas as pd
    
    scores = extract_scores(result)
    
    results_dict = {
        "overall_scores": scores,
        "test_count": len(testset),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 저장: {output_path}")
    
    try:
        df = result.to_pandas()
        csv_path = output_path.replace('.json', '_detail.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 저장: {csv_path}")
    except Exception as e:
        print(f"⚠️ CSV 저장 실패: {str(e)}")


def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("RAGAS RAG 평가 시스템 (LLM 설정 포함)")
    print("="*80)
    
    try:
        # 테스트셋 로드
        testset_path = input("\n테스트셋 경로 (기본: src/evaluation/results/synthetic_testset_100.json): ").strip()
        if not testset_path:
            testset_path = "src/evaluation/results/synthetic_testset_100.json"
        
        testset = load_testset(testset_path)
        
        # RAG 파이프라인 초기화
        print("\n" + "="*80)
        print("RAG 파이프라인 초기화")
        print("="*80)
        
        pipeline = RAGPipeline(config=config, model=config.LLM_MODEL_NAME)
        print(f"✅ RAG Pipeline 준비 완료")
        
        # RAG 실행
        eval_dataset = run_rag_on_testset(testset, pipeline, batch_size=10)
        
        # 평가 실행
        proceed = input("\nRAGAS 평가를 실행하시겠습니까? (y/n): ").strip().lower()
        if proceed != 'y':
            print("평가 건너뜀")
            return
        
        # RAGAS 평가 (LLM 설정 포함!)
        result = evaluate_with_ragas(eval_dataset)
        
        # 결과 출력
        print_results(result)
        
        # 결과 저장
        save_results(result, eval_dataset)
        
        print("\n" + "="*80)
        print("✅ 완료!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()