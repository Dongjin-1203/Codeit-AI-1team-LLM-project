"""
RAG 전체 파이프라인 실행 스크립트

단계:
1. 전처리 (preprocess): 텍스트 추출 → 정제 → 청킹
2. 임베딩 (embed): 청크 벡터화 → ChromaDB 저장
3. RAG (rag): RAG 파이프라인 테스트 (선택)

사용법:
    python main.py --step all              # 전체 실행
    python main.py --step preprocess       # 전처리만
    python main.py --step embed            # 임베딩만
    python main.py --step rag              # RAG 테스트만
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import PreprocessConfig
from src.loader.preprocess_pipeline import RAGPreprocessPipeline


def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='RAG 전체 파이프라인 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --step all                    # 전체 파이프라인 실행
  python main.py --step preprocess             # 전처리만 실행
  python main.py --step embed                  # 임베딩만 실행
  python main.py --step rag --query "질문"    # RAG 테스트
  
  python main.py --step preprocess --chunk-size 500  # 청크 크기 조정
        """
    )
    
    # 실행 단계 선택
    parser.add_argument(
        '--step',
        type=str,
        choices=['all', 'preprocess', 'embed', 'rag'],
        default='all',
        help='실행할 단계 (기본값: all)'
    )
    
    # 전처리 관련 인자
    preprocess_group = parser.add_argument_group('전처리 옵션')
    preprocess_group.add_argument(
        '--meta-csv',
        type=str,
        default='./data/data_list.csv',
        help='메타데이터 CSV 파일 경로'
    )
    preprocess_group.add_argument(
        '--files-dir',
        type=str,
        default='./data/files/',
        help='원본 파일 폴더 경로'
    )
    preprocess_group.add_argument(
        '--output-chunks',
        type=str,
        default='./data/rag_chunks_final_small.csv',
        help='청크 출력 파일 경로'
    )
    preprocess_group.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='청크 크기'
    )
    preprocess_group.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='청크 오버랩'
    )
    
    # RAG 관련 인자
    rag_group = parser.add_argument_group('RAG 옵션')
    rag_group.add_argument(
        '--query',
        type=str,
        help='RAG 질의 (rag 단계에서만 사용)'
    )
    
    return parser.parse_args()


def step_preprocess(args):
    """1단계: 전처리 실행"""
    print("\n" + "="*70)
    print("🔧 1단계: 데이터 전처리 시작")
    print("="*70)
    
    # 설정 초기화
    config = PreprocessConfig()
    config.META_CSV_PATH = args.meta_csv
    config.BASE_FOLDER_PATH = args.files_dir
    config.OUTPUT_CHUNKS_PATH = args.output_chunks
    config.CHUNK_SIZE = args.chunk_size
    config.CHUNK_OVERLAP = args.chunk_overlap
    
    # 전처리 파이프라인 실행
    pipeline = RAGPreprocessPipeline(config)
    df_chunks = pipeline.run()
    
    print("\n" + "="*70)
    print("✅ 1단계: 전처리 완료")
    print("="*70)
    print(f"📁 출력 파일: {config.OUTPUT_CHUNKS_PATH}")
    print(f"📊 총 청크 수: {len(df_chunks)}")
    
    return df_chunks


def step_embed(args):
    """2단계: 임베딩 및 ChromaDB 저장"""
    print("\n" + "="*70)
    print("🔧 2단계: 임베딩 및 벡터DB 구축 시작")
    print("="*70)
    
    try:
        # 임베딩 모듈 임포트
        from src.embedding.rag_data_processing import RAGVectorDBPipeline
        
        # 임베딩 실행
        pipeline = RAGVectorDBPipeline()
        vectorstore = pipeline.build()
        
        print("\n" + "="*70)
        print("✅ 2단계: 임베딩 완료")
        print("="*70)
        
    except ImportError as e:
        print(f"⚠️  임베딩 모듈을 찾을 수 없습니다: {e}")
        print("   src/embedding/rag_data_processing.py 파일이 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 임베딩 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def step_rag(args):
    """3단계: RAG 파이프라인 테스트"""
    print("\n" + "="*70)
    print("🔧 3단계: RAG 파이프라인 테스트")
    print("="*70)
    
    try:
        # RAG 모듈 임포트
        from src.generator.generator import RAGPipeline
        from src.utils.rag_config import RAGConfig
        
        # RAG 설정
        config = RAGConfig()
        
        # RAG 파이프라인 초기화
        rag = RAGPipeline(config=config)
        
        # 테스트 질의 실행
        if args.query:
            print(f"\n📝 질의: {args.query}")
            result = rag.generate_answer(args.query)
            
            print(f"\n💬 답변:")
            print(result['answer'])
            print(f"\n📚 참고 문서: {len(result.get('sources', []))}개")
            print(f"🔢 토큰 사용: {result['usage']['total_tokens']}")
        else:
            print("\n⚠️  --query 인자가 없어 테스트 질의를 건너뜁니다.")
            print("   예시: python main.py --step rag --query '한영대학교 특성화 사업은?'")
        
        print("\n" + "="*70)
        print("✅ 3단계: RAG 파이프라인 완료")
        print("="*70)
        
    except ImportError as e:
        print(f"⚠️  RAG 모듈을 찾을 수 없습니다: {e}")
        print("   src/generator/rag_pipeline.py 파일이 있는지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ RAG 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    print("="*70)
    print("🚀 RAG 전체 파이프라인")
    print("="*70)
    print(f"실행 단계: {args.step}")
    
    try:
        if args.step == 'all':
            # 전체 파이프라인 실행
            step_preprocess(args)
            step_embed(args)
            
            # RAG 테스트는 선택적 (query가 있으면 실행)
            if args.query:
                step_rag(args)
            
        elif args.step == 'preprocess':
            step_preprocess(args)
            
        elif args.step == 'embed':
            step_embed(args)
            
        elif args.step == 'rag':
            step_rag(args)
        
        print("\n" + "="*70)
        print("🎉 모든 작업 완료!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()