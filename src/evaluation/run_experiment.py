"""
RAG 검색 시스템 평가 도구
- LangSmith Experiment 실행
- Context Precision/Recall 평가
- 실험 추적 및 비교

사용법:
    python run_experiment.py              # 대화형 메뉴
    python run_experiment.py --run        # 실험 실행
    python run_experiment.py --compare    # 실험 비교
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from langsmith import Client, evaluate
from dotenv import load_dotenv

# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retriever.rag_retriever import RAGRetriever
from src.utils.config import RAGConfig
from src.evaluation.experiment_tracker import ExperimentTracker


# === 환경 설정 ===
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "RAG-Retriever-Eval"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# === 전역 변수 ===
retriever = None


# ============================================================
# Evaluator 함수들
# ============================================================

def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    # 소문자 변환
    normalized = text.lower()
    
    # 특수문자 제거
    normalized = re.sub(r'[\r\n\t]+', ' ', normalized)
    
    # 연속 공백 하나로
    normalized = ' '.join(normalized.split())
    
    return normalized.strip()


def is_matching_context(retrieved_text: str, ground_truth_text: str, threshold: float = 0.5) -> bool:
    """두 문서가 같은 문서인지 판단"""
    normalized_retrieved = normalize_text(retrieved_text)
    normalized_truth = normalize_text(ground_truth_text)
    
    # 완전 포함 체크
    if normalized_truth in normalized_retrieved:
        return True
    
    if normalized_retrieved in normalized_truth:
        return True
    
    # 단어 커버리지 체크
    truth_words = set(normalized_truth.split())
    retrieved_words = set(normalized_retrieved.split())
    
    if len(truth_words) == 0:
        return False
    
    matched_words = truth_words & retrieved_words
    coverage = len(matched_words) / len(truth_words)
    
    return coverage >= threshold


def count_matching_contexts(
    retrieved_contexts: List[str],
    ground_truth_contexts: List[str],
    threshold: float = 0.5
) -> int:
    """매칭되는 문서 개수 계산"""
    matched_count = 0
    
    for retrieved in retrieved_contexts:
        for truth in ground_truth_contexts:
            if is_matching_context(retrieved, truth, threshold):
                matched_count += 1
                break
    
    return matched_count


def context_precision_evaluator(run: Any, example: Any) -> Dict[str, float]:
    """Context Precision 평가"""
    try:
        # 검색 결과 추출
        if isinstance(run.outputs, dict):
            retrieved_results = run.outputs.get('output', [])
        else:
            retrieved_results = run.outputs
        
        # 텍스트만 추출
        retrieved_contexts = []
        for result in retrieved_results:
            if isinstance(result, dict):
                text = result.get('content', '')
                if text:
                    retrieved_contexts.append(text)
        
        # 정답 추출
        ground_truth_contexts = example.outputs.get('ground_truth_contexts', [])
        
        # 검증
        if len(retrieved_contexts) == 0:
            return {"key": "context_precision", "score": 0.0, "comment": "검색 결과 없음"}
        
        if len(ground_truth_contexts) == 0:
            return {"key": "context_precision", "score": 0.0, "comment": "정답 없음"}
        
        # 매칭 개수 계산
        matched_count = count_matching_contexts(
            retrieved_contexts,
            ground_truth_contexts,
            threshold=0.5
        )
        
        # Precision 계산
        precision = matched_count / len(retrieved_contexts)
        
        return {
            "key": "context_precision",
            "score": precision,
            "comment": f"매칭: {matched_count}/{len(retrieved_contexts)}"
        }
        
    except Exception as e:
        print(f"Context Precision 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"key": "context_precision", "score": 0.0, "comment": f"오류: {str(e)}"}


def context_recall_evaluator(run: Any, example: Any) -> Dict[str, float]:
    """Context Recall 평가"""
    try:
        # 검색 결과 추출
        if isinstance(run.outputs, dict):
            retrieved_results = run.outputs.get('output', [])
        else:
            retrieved_results = run.outputs
        
        retrieved_contexts = []
        for result in retrieved_results:
            if isinstance(result, dict):
                text = result.get('content', '')
                if text:
                    retrieved_contexts.append(text)
        
        # 정답 추출
        ground_truth_contexts = example.outputs.get('ground_truth_contexts', [])
        
        # 검증
        if len(ground_truth_contexts) == 0:
            return {"key": "context_recall", "score": 0.0, "comment": "정답 없음"}
        
        if len(retrieved_contexts) == 0:
            return {"key": "context_recall", "score": 0.0, "comment": "검색 결과 없음"}
        
        # 매칭 개수 계산
        matched_count = 0
        for truth in ground_truth_contexts:
            for retrieved in retrieved_contexts:
                if is_matching_context(retrieved, truth, threshold=0.5):
                    matched_count += 1
                    break
        
        # Recall 계산
        recall = matched_count / len(ground_truth_contexts)
        
        return {
            "key": "context_recall",
            "score": recall,
            "comment": f"발견: {matched_count}/{len(ground_truth_contexts)}"
        }
        
    except Exception as e:
        print(f"Context Recall 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"key": "context_recall", "score": 0.0, "comment": f"오류: {str(e)}"}


def retrieval_time_evaluator(run: Any, example: Any) -> Dict[str, float]:
    """검색 시간 측정"""
    try:
        latency = run.execution_time
        return {
            "key": "retrieval_time",
            "score": latency,
            "comment": f"{latency:.3f}초"
        }
    except Exception as e:
        return {"key": "retrieval_time", "score": 0.0, "comment": "시간 측정 실패"}


# ============================================================
# Target 함수
# ============================================================

def retriever_target(inputs: dict) -> dict:
    """LangSmith Experiment용 검색 함수"""
    question = inputs.get("question", "")
    
    if not question:
        return {"output": []}
    
    # 하이브리드 검색 + Re-ranker 실행
    results = retriever.search_with_mode(
        query=question, 
        top_k=None, 
        mode="hybrid_rerank", 
        alpha=0.5
    )
    
    return {"output": results}


# ============================================================
# 실험 실행
# ============================================================

def run_experiment(
    experiment_name: str,
    config: dict,
    dataset_name: str = "RAG-Retriever-TestSet-v1",
    notes: str = ""
) -> dict:
    """
    실험 실행 및 자동 추적
    
    Args:
        experiment_name: 실험 이름
        config: 실험 설정
        dataset_name: Dataset 이름
        notes: 메모
        
    Returns:
        실험 결과
    """
    global retriever
    
    print("\n" + "="*80)
    print(f"🚀 실험 시작: {experiment_name}")
    print("="*80)
    
    # 1. 검색기 초기화
    print("\n🔧 검색기 초기화...")
    rag_config = RAGConfig()
    
    # Config 적용
    if 'embedding_model' in config:
        rag_config.EMBEDDING_MODEL_NAME = config['embedding_model']
    if 'top_k' in config:
        rag_config.DEFAULT_TOP_K = config['top_k']
    
    retriever = RAGRetriever(config=rag_config)
    
    print(f"✅ 설정 완료:")
    print(f"   임베딩 모델: {rag_config.EMBEDDING_MODEL_NAME}")
    print(f"   Top-K: {rag_config.DEFAULT_TOP_K}")
    
    # 2. Evaluators 설정
    evaluators_list = [
        context_precision_evaluator,
        context_recall_evaluator,
    ]
    
    # 3. LangSmith Client 초기화
    client = Client()
    
    # 4. Experiment 실행
    print(f"\n⏳ Experiment 실행 중...")
    
    try:
        results = evaluate(
            retriever_target,
            data=dataset_name,
            evaluators=evaluators_list,
            experiment_prefix=experiment_name,
            max_concurrency=1,
        )
        
        print(f"\n✅ Experiment 완료!")
        
        # 5. 결과 추출
        df = results.to_pandas()
        
        metrics = {
            "precision": df["feedback.context_precision"].mean(),
            "recall": df["feedback.context_recall"].mean(),
            "avg_time": df["execution_time"].mean(),
        }
        
        # 6. 자동 추적 저장
        tracker = ExperimentTracker()
        
        langsmith_url = "https://smith.langchain.com/"
        
        tracker.log_experiment(
            experiment_name=experiment_name,
            config=config,
            metrics=metrics,
            langsmith_url=langsmith_url,
            notes=notes
        )
        
        # 7. 결과 출력
        print("\n" + "="*80)
        print("📊 실험 결과")
        print("="*80)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        f1 = 0
        if (metrics['precision'] + metrics['recall']) > 0:
            f1 = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        print(f"F1: {f1:.4f}")
        print(f"평균 검색 시간: {metrics['avg_time']:.3f}초")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# 대화형 메뉴
# ============================================================

def interactive_run():
    """대화형 실험 실행"""
    print("\n" + "="*80)
    print("🧪 RAG 검색 시스템 성능 실험")
    print("="*80)
    
    # 실험 설정 입력
    print("\n실험 설정을 입력하세요:")
    
    experiment_name = input("실험 이름 (예: baseline, hybrid-rerank): ").strip()
    if not experiment_name:
        experiment_name = "experiment"
    
    embedding_model = input("임베딩 모델 (엔터: text-embedding-3-small): ").strip()
    if not embedding_model:
        embedding_model = "text-embedding-3-small"
    
    top_k_input = input("Top-K (엔터: 10): ").strip()
    top_k = int(top_k_input) if top_k_input else 10
    
    notes = input("메모 (선택사항): ").strip()
    
    # 설정 구성
    config = {
        "embedding_model": embedding_model,
        "top_k": top_k,
    }
    
    # 확인
    print("\n" + "="*80)
    print("📋 실험 정보 확인")
    print("="*80)
    print(f"실험 이름: {experiment_name}")
    print(f"임베딩 모델: {embedding_model}")
    print(f"Top-K: {top_k}")
    if notes:
        print(f"메모: {notes}")
    print("="*80)
    
    confirm = input("\n실험을 시작하시겠습니까? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 취소됨")
        return
    
    # 실험 실행
    run_experiment(
        experiment_name=experiment_name,
        config=config,
        notes=notes
    )


def interactive_compare():
    """대화형 실험 비교"""
    tracker = ExperimentTracker()
    
    print("\n" + "="*80)
    print("🔍 실험 비교 도구")
    print("="*80)
    
    while True:
        print("\n메뉴:")
        print("  1. 모든 실험 목록 보기")
        print("  2. 최근 실험 비교 (최근 5개)")
        print("  3. 특정 실험 비교")
        print("  4. 개선 효과 확인")
        print("  5. 차트 생성")
        print("  6. 최적 설정 추천")
        print("  0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "1":
            tracker.list_experiments()
        
        elif choice == "2":
            tracker.compare_experiments(top_n=5)
        
        elif choice == "3":
            names = input("실험 이름들 (쉼표로 구분): ").strip()
            if names:
                experiment_names = [n.strip() for n in names.split(',')]
                tracker.compare_experiments(experiment_names=experiment_names)
        
        elif choice == "4":
            baseline = input("Baseline 실험 이름: ").strip()
            current = input("비교할 실험 이름: ").strip()
            
            if baseline and current:
                tracker.show_improvement(baseline, current)
        
        elif choice == "5":
            names_input = input("실험 이름들 (쉼표로 구분, 엔터: 전체): ").strip()
            
            if names_input:
                experiment_names = [n.strip() for n in names_input.split(',')]
            else:
                experiment_names = None
            
            tracker.plot_metrics(experiment_names=experiment_names)
        
        elif choice == "6":
            metric = input("기준 지표 (precision/recall/f1, 엔터: f1): ").strip()
            if not metric:
                metric = "f1"
            
            tracker.recommend_best(metric=metric)
        
        elif choice == "0":
            print("👋 종료합니다")
            break
        
        else:
            print("❌ 잘못된 선택입니다")


def main_menu():
    """메인 메뉴"""
    print("\n" + "="*80)
    print("🔬 RAG 평가 시스템")
    print("="*80)
    
    while True:
        print("\n메뉴:")
        print("  1. 실험 실행")
        print("  2. 실험 비교")
        print("  0. 종료")
        
        choice = input("\n선택: ").strip()
        
        if choice == "1":
            interactive_run()
        
        elif choice == "2":
            interactive_compare()
        
        elif choice == "0":
            print("👋 종료합니다")
            break
        
        else:
            print("❌ 잘못된 선택입니다")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='RAG 평가 시스템')
    
    parser.add_argument(
        '--run',
        action='store_true',
        help='실험 실행 모드'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='실험 비교 모드'
    )
    
    args = parser.parse_args()
    
    try:
        if args.run:
            interactive_run()
        elif args.compare:
            interactive_compare()
        else:
            main_menu()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()