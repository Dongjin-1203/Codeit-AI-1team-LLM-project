"""
RAG 검색기 평가 실행 스크립트 (WandB 연동)

사용법:
    python run_evaluation.py

WandB 로그인이 필요합니다:
    wandb login
"""

import json
import wandb
from datetime import datetime
from pathlib import Path

# 기존 코드 임포트 (경로는 프로젝트에 맞게 수정)
from src.utils.preprocess_config import PreprocessConfig
from src.utils.rag_config import RAGConfig
from src.generator.rag_pipeline import RAGPipeline  # ← 변경
from src.evaluation.evaluator import RAGEvaluator


def main():
    """메인 평가 실행 함수"""
    
    print("\n" + "="*80)
    print("RAG 검색기 성능 평가 시작")
    print("="*80 + "\n")
    
    # ========== 설정 (여기를 수정하세요) ==========
    
    # 1. 테스트 데이터 경로
    TEST_QUERIES_PATH = "src/evaluation/test_queries_mapped.json"
    
    # 2. WandB 설정
    WANDB_PROJECT = "rag-evaluation"
    WANDB_RUN_NAME = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 3. 설정
    RAGconfig = RAGConfig()
    config = PreprocessConfig()
    
    # ==========================================
    
    # Step 1: WandB 초기화
    print("📊 WandB 초기화 중...")
    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "test_queries_path": TEST_QUERIES_PATH,
            "embedding_model": RAGconfig.EMBEDDING_MODEL_NAME,
            "llm_model": RAGconfig.LLM_MODEL_NAME,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": RAGconfig.DEFAULT_TOP_K,
        },
        tags=["evaluation", "baseline"]
    )
    print(f"✅ WandB 실행: {run.url}\n")
    
    # Step 2: RAG Pipeline 초기화 (변경)
    print("🔍 RAG Pipeline 초기화 중...")
    pipeline = RAGPipeline(config=RAGconfig, model="gpt-5-mini")
    print("✅ Pipeline 준비 완료\n")
    
    # Step 3: 평가기 초기화
    print("⚙️ 평가기 초기화 중...")
    evaluator = RAGEvaluator(
        pipeline=pipeline,  # ← retriever에서 pipeline으로 변경
        test_data_path=TEST_QUERIES_PATH,
        use_wandb=True
    )
    print("✅ 평가기 준비 완료\n")
    
    # Step 4: 평가 실행
    print("🚀 평가 시작...\n")
    results = evaluator.evaluate_all(save_results=True)
    
    # Step 5: 결과 요약 출력
    print("\n" + "="*80)
    print("📊 평가 결과 요약")
    print("="*80 + "\n")
    
    summary = results['summary']
    
    print("전체 평균 메트릭:")
    print(f"  Recall@3:    {summary.get('avg_recall@3', 0):.3f}")
    print(f"  Recall@5:    {summary.get('avg_recall@5', 0):.3f}")
    print(f"  Recall@10:   {summary.get('avg_recall@10', 0):.3f}")
    print(f"  MRR:         {summary.get('avg_mrr', 0):.3f}")
    print(f"  NDCG@5:      {summary.get('avg_ndcg@5', 0):.3f}")
    print(f"  Hit Rate@5:  {summary.get('avg_hit_rate@5', 0):.3f}")
    
    print("\n난이도별 성능:")
    for difficulty, stats in summary.get('by_difficulty', {}).items():
        print(f"\n  [{difficulty}]")
        print(f"    Recall@5: {stats.get('avg_recall@5', 0):.3f}")
        print(f"    MRR:      {stats.get('avg_mrr', 0):.3f}")
    
    print("\n카테고리별 성능:")
    for category, stats in summary.get('by_category', {}).items():
        print(f"\n  [{category}]")
        print(f"    Recall@5: {stats.get('avg_recall@5', 0):.3f}")
        print(f"    MRR:      {stats.get('avg_mrr', 0):.3f}")
    
    # Step 6: WandB Summary 업데이트
    print("\n📊 WandB에 최종 결과 업로드 중...")
    wandb.summary.update({
        "final_recall@5": summary.get('avg_recall@5', 0),
        "final_mrr": summary.get('avg_mrr', 0),
        "final_ndcg@5": summary.get('avg_ndcg@5', 0),
        "final_hit_rate@5": summary.get('avg_hit_rate@5', 0),
        "total_queries": len(results['details'])
    })
    
    # 실패 케이스 분석
    failed_queries = [
        r for r in results['details'] 
        if r.get('hit_rate@5', 0) == 0
    ]
    
    if failed_queries:
        print(f"\n⚠️ 실패한 쿼리: {len(failed_queries)}개")
        print("\n상위 5개 실패 케이스:")
        for i, fail in enumerate(failed_queries[:5], 1):
            print(f"\n  {i}. [{fail['category']}] {fail['query']}")
            print(f"     예상 문서: {fail['relevant_ids'][:2]}...")
    
    # Step 7: WandB 종료
    wandb.finish()
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("="*80)
    print(f"\n📊 WandB 대시보드: {run.url}")
    print(f"📁 결과 파일: evaluation/results/eval_*.json")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 평가가 중단되었습니다.")
        wandb.finish()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        wandb.finish()
        raise