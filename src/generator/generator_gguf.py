from llama_cpp import Llama
from typing import Optional, Dict, Any, List
import logging
import time

from src.router.query_router import QueryRouter
from src.prompts.dynamic_prompts import PromptManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GGUFGenerator:
    """
    GGUF 기반 Llama-3 생성기
    
    llama.cpp를 사용하여 GGUF 포맷 모델을 로드하고
    입찰 관련 질의응답을 수행합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_threads: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "당신은 RFP(제안요청서) 분석 및 요약 전문가입니다."
    ):
        """
        생성기 초기화
        
        Args:
            model_path: GGUF 모델 파일 경로
            n_gpu_layers: GPU에 올릴 레이어 수 (0 = CPU만, 35 = 전체 GPU)
            n_ctx: 최대 컨텍스트 길이
            n_threads: CPU 스레드 수
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (0.0~1.0)
            top_p: Nucleus sampling 파라미터
            system_prompt: 시스템 프롬프트
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 모델 (나중에 로드)
        self.model = None
        
        logger.info(f"GGUFGenerator 초기화 완료")
    
    def load_model(self) -> None:
        """
        GGUF 모델 로드
        
        Raises:
            FileNotFoundError: 모델 파일이 없는 경우
            RuntimeError: 모델 로드 실패
        """
        # 중복 로드 방지
        if self.model is not None:
            logger.info("모델이 이미 로드되어 있습니다.")
            return
        
        try:
            logger.info(f"GGUF 모델 로드 중: {self.model_path}")
            logger.info(f"설정: GPU 레이어={self.n_gpu_layers}, 컨텍스트={self.n_ctx}, 스레드={self.n_threads}")
            
            if self.n_gpu_layers > 0:
                logger.info(f"🚀 GPU 가속 사용 (레이어 {self.n_gpu_layers}개)")
            else:
                logger.info(f"💻 CPU 전용 모드")
            
            # GGUF 모델 로드
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            
            logger.info("✅ GGUF 모델 로드 완료!")
            
        except FileNotFoundError as e:
            logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise RuntimeError(f"모델 로드 중 오류 발생: {e}")
    
    def format_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Llama-3 Chat 템플릿으로 프롬프트 포맷팅
        
        Args:
            question: 사용자 질문
            context: 선택적 컨텍스트 (RAG 검색 결과)
            system_prompt: 선택적 시스템 프롬프트
        
        Returns:
            포맷된 프롬프트 문자열
        """
        # 시스템 프롬프트 설정
        if system_prompt is None:
            system_prompt = self.system_prompt
            logger.warning("⚠️ system_prompt가 None! 기본 프롬프트 사용")
        else:
            # 동적 프롬프트 미리보기 (처음 150자만)
            logger.info(f"✅ 동적 프롬프트 적용:\n{system_prompt[:150]}...")  # ← 추가
            
        # 컨텍스트 포함 여부
        if context is not None:
            user_message = f"참고 문서:\n{context}\n\n질문: {question}"
        else:
            user_message = question
        
        # Llama-3 Chat 템플릿 적용
        formatted_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        return formatted_prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        프롬프트를 입력받아 응답 생성
        
        Args:
            prompt: 포맷된 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성
            top_p: Nucleus sampling
        
        Returns:
            생성된 응답 텍스트
        
        Raises:
            RuntimeError: 모델이 로드되지 않은 경우
        """
        # 모델 로드 확인
        if self.model is None:
            raise RuntimeError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요."
            )
        
        # 파라미터 설정
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        try:
            logger.info(f"🔄 생성 시작 (max_tokens={max_new_tokens}, temp={temperature})")
            start_time = time.time()
            
            # 생성
            output = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,  # 프롬프트 반복 안 함
                stop=["<|eot_id|>", "<|end_of_text|>"],  # 종료 토큰
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✅ 생성 완료: {elapsed:.2f}초")
            
            # 응답 추출
            response = output['choices'][0]['text'].strip()
            
            logger.info(f"📝 응답 길이: {len(response)} 글자")
            return response
            
        except Exception as e:
            logger.error(f"❌ 생성 중 오류 발생: {e}")
            raise RuntimeError(f"텍스트 생성 실패: {e}")
    
    def chat(
        self,
        question: str,
        context: Optional[str] = None,
        system_prompt=None,
        **kwargs
    ) -> str:
        """
        질문에 대한 응답 생성 (통합 메서드)
        
        Args:
            question: 사용자 질문
            context: 선택적 컨텍스트 (RAG 결과)
            **kwargs: generate() 파라미터
        
        Returns:
            생성된 응답
        """
        # 프롬프트 포맷팅 (system_prompt 전달)
        formatted_prompt = self.format_prompt(
            question=question,
            context=context,
            system_prompt=system_prompt  # ← 추가!
        )
        
        # 응답 생성
        response = self.generate(formatted_prompt, **kwargs)

        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        info = {
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "is_loaded": self.model is not None,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        return info
    
    def __repr__(self):
        return f"GGUFGenerator(model={self.model_path}, loaded={self.model is not None})"


# ===== GGUF RAGPipeline: chatbot_app.py 호환용 =====

class GGUFRAGPipeline:
    """
    GGUF 모델 기반 RAG 파이프라인
    
    RAGPipeline(API 버전)과 동일한 인터페이스를 제공하여
    chatbot_app.py와 호환됩니다.
    """
    
    def __init__(self, config=None, model: str = None, top_k: int = None):
        """
        초기화
        
        Args:
            config: RAGConfig 객체
            model: 모델 이름 (사용 안 함, 호환성용)
            top_k: 기본 검색 문서 수
        """
        # Config import (지연 import로 순환 참조 방지)
        from src.utils.config import RAGConfig
        from src.retriever.retriever import RAGRetriever
        
        self.config = config or RAGConfig()
        self.top_k = top_k or self.config.DEFAULT_TOP_K
        
        # 검색 설정
        self.search_mode = self.config.DEFAULT_SEARCH_MODE
        self.alpha = self.config.DEFAULT_ALPHA
        
        # Retriever 초기화
        logger.info("RAGRetriever 초기화 중...")
        self.retriever = RAGRetriever(config=self.config)
        
        # GGUFGenerator 초기화
        logger.info("GGUFGenerator 초기화 중...")
        self.generator = GGUFGenerator(
            model_path=self.config.GGUF_MODEL_PATH,
            n_gpu_layers=self.config.GGUF_N_GPU_LAYERS,
            n_ctx=self.config.GGUF_N_CTX,
            n_threads=self.config.GGUF_N_THREADS,
            max_new_tokens=self.config.GGUF_MAX_NEW_TOKENS,
            temperature=self.config.GGUF_TEMPERATURE,
            top_p=self.config.GGUF_TOP_P,
            system_prompt=self.config.SYSTEM_PROMPT
        )
        
        # 모델 로드 (시간 소요)
        logger.info("GGUF 모델 로드 중...")
        self.generator.load_model()
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        # 마지막 검색 결과 저장 (sources 반환용)
        self._last_retrieved_docs = []
        
        logger.info("✅ GGUFRAGPipeline 초기화 완료")
        logger.info(f"   - 검색 모드: {self.search_mode}")
        logger.info(f"   - 기본 top_k: {self.top_k}")
    
    def _retrieve_and_format(self, query: str) -> str:
        """검색 수행 및 컨텍스트 포맷팅"""
        # 검색 모드에 따라 문서 검색
        if self.search_mode == "embedding":
            docs = self.retriever.search(query, top_k=self.top_k)
        elif self.search_mode == "embedding_rerank":
            docs = self.retriever.search_with_rerank(query, top_k=self.top_k)
        elif self.search_mode == "hybrid":
            docs = self.retriever.hybrid_search(
                query, top_k=self.top_k, alpha=self.alpha
            )
        elif self.search_mode == "hybrid_rerank":
            docs = self.retriever.hybrid_search_with_rerank(
                query, top_k=self.top_k, alpha=self.alpha
            )
        else:
            docs = self.retriever.search(query, top_k=self.top_k)
        
        # 마지막 검색 결과 저장
        self._last_retrieved_docs = docs
        
        # 컨텍스트 포맷팅
        return self._format_context(docs)
    
    def _format_context(self, retrieved_docs: list) -> str:
        """검색된 문서를 컨텍스트로 변환"""
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[문서 {i}]\n{doc['content']}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: list) -> list:
        """검색된 문서를 sources 형식으로 변환"""
        sources = []
        for doc in retrieved_docs:
            source_info = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'filename': doc.get('filename', 'N/A'),
                'organization': doc.get('organization', 'N/A')
            }
            
            # 검색 모드에 따라 점수 필드가 다름
            if 'rerank_score' in doc:
                source_info['score'] = doc['rerank_score']
                source_info['score_type'] = 'rerank'
            elif 'hybrid_score' in doc:
                source_info['score'] = doc['hybrid_score']
                source_info['score_type'] = 'hybrid'
            elif 'relevance_score' in doc:
                source_info['score'] = doc['relevance_score']
                source_info['score_type'] = 'embedding'
            else:
                source_info['score'] = 0
                source_info['score_type'] = 'unknown'
            
            sources.append(source_info)
        
        return sources
    
    def _estimate_usage(self, query: str, answer: str) -> dict:
        """토큰 사용량 추정"""
        # 간단한 단어 수 기반 추정
        prompt_tokens = len(query.split()) * 2
        completion_tokens = len(answer.split()) * 2
        
        return {
            'total_tokens': prompt_tokens + completion_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }
    
    def generate_answer(
        self,
        query: str,
        top_k: int = None,
        search_mode: str = None,
        alpha: float = None
    ) -> dict:
        """
        답변 생성 (chatbot_app.py 호환 메인 메서드)
        
        Args:
            query: 질문
            top_k: 검색할 문서 수
            search_mode: 검색 모드
            alpha: 임베딩 가중치
        
        Returns:
            dict: answer, sources, search_mode, usage, elapsed_time, used_retrieval
        """
        try:
            start_time = time.time()
            
            # 파라미터 설정 (검색 전에 먼저 설정)
            if top_k is not None:
                self.top_k = top_k
            if search_mode is not None:
                self.search_mode = search_mode
            if alpha is not None:
                self.alpha = alpha

            # ===== Router로 검색 여부 결정 =====
            router = QueryRouter()
            classification = router.classify(query)
            query_type = classification['type']  # 'greeting'/'thanks'/'document'/'out_of_scope'
            
            logger.info(f"📍 분류: {query_type} "
                f"(신뢰도: {classification['confidence']:.2f})")
            
            # 2. 타입별 처리
            if query_type in ['greeting', 'thanks', 'out_of_scope']:
                # 검색 스킵
                context = None
                used_retrieval = False
                self._last_retrieved_docs = []
                
                # 동적 프롬프트 선택
                system_prompt = PromptManager.get_prompt(query_type)
                logger.info(f"⏭️ RAG 스킵: {query_type}")
            
            elif query_type == 'document':
                # RAG 수행
                context = self._retrieve_and_format(query)
                used_retrieval = True
                
                # 동적 프롬프트 (context 포함)
                system_prompt = PromptManager.get_prompt('document')
                logger.info(f"🔍 RAG 수행: {len(self._last_retrieved_docs)}개 문서")
            
            # 3. 답변 생성 (system_prompt 전달)
            answer = self.generator.chat(
                question=query,
                context=context,
                system_prompt=system_prompt  # ← 추가!
            )
            
            elapsed_time = time.time() - start_time
            
            # 대화 히스토리에 추가
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # 결과 반환 (RAGPipeline과 동일 형식)
            return {
                'answer': answer,
                'sources': self._format_sources(self._last_retrieved_docs),
                'used_retrieval': used_retrieval,
                'query_type': query_type,  # ← 추가!
                'search_mode': self.search_mode if used_retrieval else 'direct',
                'routing_info': classification,
                'elapsed_time': elapsed_time,
                'usage': self._estimate_usage(query, answer)
            }
        
        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e
    
    def chat(self, query: str) -> str:
        """간단한 대화 인터페이스"""
        result = self.generate_answer(query)
        return result['answer']
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history = []
        logger.info("🗑️ 대화 히스토리가 초기화되었습니다.")
    
    def get_history(self) -> List[Dict]:
        """대화 히스토리 반환"""
        return self.chat_history.copy()
    
    def set_search_config(
        self,
        search_mode: str = None,
        top_k: int = None,
        alpha: float = None
    ):
        """검색 설정 변경"""
        if search_mode is not None:
            self.search_mode = search_mode
        if top_k is not None:
            self.top_k = top_k
        if alpha is not None:
            self.alpha = alpha
        
        logger.info(
            f"🔧 검색 설정 변경: mode={self.search_mode}, "
            f"top_k={self.top_k}, alpha={self.alpha}"
        )


# 테스트용
if __name__ == "__main__":
    from src.utils.config import RAGConfig
    
    config = RAGConfig()
    
    # GGUFRAGPipeline 초기화
    pipeline = GGUFRAGPipeline(config=config)
    
    # 테스트 질문들
    test_questions = [
        "안녕하세요",
        "본 사업의 예산 범위는 어떻게 되나요?",
        "고마워요!"
    ]
    
    for question in test_questions:
        print("\n" + "="*50)
        print("테스트 질문:", question)
        print("="*50)
        
        result = pipeline.generate_answer(question)
        
        print(f"\n라우팅: {result['routing_info']['route']}")
        print(f"검색 사용: {result['used_retrieval']}")
        print("\n응답:")
        print(result['answer'])
        print(f"\n소요 시간: {result['elapsed_time']:.2f}초")
        print(f"참고 문서: {len(result['sources'])}개")
        print("="*50)