# ===== run_experiment_with_tracking.py =====
"""
자동 추적 기능이 통합된 LangSmith Experiment 실행 스크립트
"""

import os
import sys
from pathlib import Path
from langsmith import Client, evaluate
from dotenv import load_dotenv

# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.retriever.rag_retriever import RAGRetriever
from src.utils.rag_config import RAGConfig
from src.evaluation.langsmith_evaluator import (
    context_precision_evaluator,
    context_recall_evaluator,
)
from src.evaluation.experiment_tracker import ExperimentTracker


# === 환경 설정 ===
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "RAG-Retriever-Eval"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


# === 전역 변수 ===
retriever = None


# === Target 함수 ===
def retriever_target(inputs: dict) -> dict:
    """LangSmith Experiment용 검색 함수"""
    question = inputs.get("question", "")
    
    if not question:
        return {"output": []}
    
    results = retriever.search(query=question, top_k=None)
    return {"output": results}


# === 실험 실행 함수 (추적 통합) ===
def run_experiment_with_tracking(
    experiment_name: str,
    config: dict,
    dataset_name: str = "RAG-Retriever-TestSet-v1",
    evaluators_list: list = None,
    notes: str = ""
) -> dict:
    """
    자동 추적이 통합된 실험 실행
    
    Args:
        experiment_name: 실험 이름 (예: "baseline", "embedding-large")
        config: 실험 설정 (embedding_model, top_k 등)
        dataset_name: Dataset 이름
        evaluators_list: Evaluator 리스트
        notes: 추가 메모
        
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
    if evaluators_list is None:
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
        
        # LangSmith URL 생성 (예시)
        langsmith_url = f"https://smith.langchain.com/"
        
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
        print(f"F1: {2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0:.4f}")
        print(f"평균 검색 시간: {metrics['avg_time']:.3f}초")
        print("="*80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()
        raise


# === 메인 실행 ===
def main():
    """메인 실행"""
    print("\n" + "="*80)
    print("🧪 RAG 검색 시스템 성능 실험 (자동 추적)")
    print("="*80)
    
    # 실험 설정 입력
    print("\n실험 설정을 입력하세요:")
    
    experiment_name = input("실험 이름 (예: baseline, embedding-large): ").strip()
    if not experiment_name:
        experiment_name = "experiment"
    
    embedding_model = input("임베딩 모델 (엔터: text-embedding-3-small): ").strip()
    if not embedding_model:
        embedding_model = "text-embedding-3-small"
    
    top_k_input = input("Top-K (엔터: 5): ").strip()
    top_k = int(top_k_input) if top_k_input else 5
    
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
    try:
        results = run_experiment_with_tracking(
            experiment_name=experiment_name,
            config=config,
            notes=notes
        )
        
        # 이전 실험과 비교 제안
        print("\n" + "="*80)
        tracker = ExperimentTracker()
        
        print("\n💡 다음 명령어로 실험을 비교할 수 있습니다:")
        print("   python -c \"from src.evaluation.experiment_tracker import ExperimentTracker; tracker = ExperimentTracker(); tracker.compare_experiments()\"")
        print("\n   또는:")
        print("   python -c \"from src.evaluation.experiment_tracker import ExperimentTracker; tracker = ExperimentTracker(); tracker.plot_metrics()\"")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 중단됨")
    except Exception as e:
        print(f"\n❌ 오류: {e}")


if __name__ == "__main__":
    main()