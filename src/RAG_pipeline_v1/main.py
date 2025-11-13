import os
from RAG_pipeline_v1.rag_config import RAGConfig
from RAG_pipeline_v1.rag_data_processing import RAGVectorDBPipeline
from RAG_pipeline_v1.rag_pipeline import RAGPipeline
from RAG_pipeline_v1.rag_evaluator import RAGEvaluator


def main():
    """메인 실행 함수"""
    
    # ===== 환경 설정 =====
    print("="*60)
    print("RAG 시스템 초기화")
    print("="*60)
    
    os.environ["OPENAI_API_KEY"] = RAGConfig.OPENAI_API_KEY
    
    config = RAGConfig()
    config.validate()
    print(config)
    
    # ===== 1. Vector DB 구축 (최초 1회만) =====
    # 주석 해제하여 실행
    # print("\n" + "="*60)
    # print("Vector DB 구축")
    # print("="*60)
    # db_pipeline = RAGVectorDBPipeline(config)
    # vectorstore = db_pipeline.build()
    # db_pipeline.test_search()
    
    # ===== 2. RAG 파이프라인 초기화 =====
    print("\n" + "="*60)
    print("RAG 파이프라인 초기화")
    print("="*60)
    
    rag = RAGPipeline(config=config)
    
    # ===== 3. 테스트 쿼리 =====
    print("\n" + "="*60)
    print("테스트 쿼리")
    print("="*60)
    
    test_queries = [
        "한영대학교의 특성화 교육환경 구축 사업은 무엇인가요?",
        "재난 안전 관리 시스템 구축 사업은 어떤 것이 있나요?",
    ]
    
    for query in test_queries:
        result = rag.generate_answer(query)
        rag.print_result(result)
        print("\n")
    
    # ===== 4. 평가 =====
    print("\n" + "="*60)
    print("시스템 평가")
    print("="*60)
    
    evaluator = RAGEvaluator(rag)
    eval_results = evaluator.evaluate()
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료")
    print("="*60)


if __name__ == "__main__":
    main()