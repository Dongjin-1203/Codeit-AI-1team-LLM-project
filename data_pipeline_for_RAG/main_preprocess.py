"""
RAG 전처리 실행 스크립트

사용법:
    python main_preprocess.py
"""

import argparse
from preprocess_config import PreprocessConfig
from preprocess_pipeline import RAGPreprocessPipeline


def main():
    """메인 실행 함수"""
    
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='RAG 데이터 전처리')
    
    parser.add_argument(
        '--meta-csv', 
        type=str, 
        default='./data/data_list.csv',
        help='메타데이터 CSV 파일 경로'
    )
    
    parser.add_argument(
        '--files-dir', 
        type=str, 
        default='./data/files/',
        help='원본 파일 폴더 경로'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./data/rag_chunks_final.csv',
        help='출력 파일 경로'
    )
    
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=1000,
        help='청크 크기'
    )
    
    parser.add_argument(
        '--chunk-overlap', 
        type=int, 
        default=200,
        help='청크 오버랩'
    )
    
    args = parser.parse_args()
    
    # 설정 초기화
    config = PreprocessConfig()
    config.META_CSV_PATH = args.meta_csv
    config.BASE_FOLDER_PATH = args.files_dir
    config.OUTPUT_CHUNKS_PATH = args.output
    config.CHUNK_SIZE = args.chunk_size
    config.CHUNK_OVERLAP = args.chunk_overlap
    
    # 파이프라인 실행
    pipeline = RAGPreprocessPipeline(config)
    df_chunks = pipeline.run()
    
    print("\n" + "="*60)
    print("전처리 완료!")
    print("="*60)
    print(f"출력 파일: {config.OUTPUT_CHUNKS_PATH}")
    print(f"총 청크 수: {len(df_chunks)}")


if __name__ == "__main__":
    main()