import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from typing import Optional, Dict, Any, List
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaGenerator:
    """
    Fine-tuned Llama-3 모델 생성기
    
    4-bit 양자화된 베이스 모델 + LoRA 어댑터를 로드하여
    입찰 관련 질의응답을 수행합니다.
    """
    
    def __init__(
        self,
        adapter_path: str,
        base_model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "당신은 RFP(제안요청서) 분석 및 요약 전문가입니다."
    ):
        """
        생성기 초기화
        
        Args:
            adapter_path: LoRA 어댑터 경로
            base_model_name: 베이스 모델 이름
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (0.0~1.0)
            top_p: Nucleus sampling 파라미터
            system_prompt: 시스템 프롬프트
        """
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # 모델 & 토크나이저 (나중에 로드)
        self.model = None
        self.tokenizer = None
        
        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"LlamaGenerator 초기화 완료 (device: {self.device})")
    
    def load_model(self) -> None:
        """
        4-bit 베이스 모델 + LoRA 어댑터 로드
        
        Raises:
            FileNotFoundError: 어댑터 경로가 없는 경우
            RuntimeError: 모델 로드 실패
        """
        # 중복 로드 방지
        if self.model is not None:
            logger.info("모델이 이미 로드되어 있습니다.")
            return
        
        try:
            logger.info("모델 로드 시작...")
            
            # 1. 4-bit 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # 2. 베이스 모델 로드 (4-bit)
            logger.info(f"베이스 모델 로드 중: {self.base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # 3. LoRA 어댑터 로드
            logger.info(f"LoRA 어댑터 로드 중: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.adapter_path
            )
            
            # 4. 토크나이저 로드
            logger.info("토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            # 5. Evaluation 모드
            self.model.eval()
            
            logger.info("✅ 모델 로드 완료!")
            if torch.cuda.is_available():
                logger.info(f"   - VRAM 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        except FileNotFoundError as e:
            logger.error(f"❌ 어댑터 파일을 찾을 수 없습니다: {e}")
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
            system_prompt: 선택적 시스템 프롬프트 (기본값 사용)
        
        Returns:
            포맷된 프롬프트 문자열
        """
        # 시스템 프롬프트 설정
        if system_prompt is None:
            system_prompt = self.system_prompt
        
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
        do_sample: bool = True,
    ) -> str:
        """
        프롬프트를 입력받아 응답 생성
        
        Args:
            prompt: 포맷된 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성
            top_p: Nucleus sampling
            do_sample: 샘플링 여부
        
        Returns:
            생성된 응답 텍스트
        
        Raises:
            RuntimeError: 모델이 로드되지 않은 경우
        """
        # 모델 로드 확인
        if self.model is None or self.tokenizer is None:
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
            # 1. 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # 최대 입력 길이
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 2. 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 3. 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 4. 후처리: 프롬프트 제거
            # assistant 태그 이후 텍스트만 추출
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                response = generated_text.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[-1].strip()
            else:
                # 입력 프롬프트 길이만큼 제거
                prompt_text = self.tokenizer.decode(
                    inputs["input_ids"][0],
                    skip_special_tokens=True
                )
                response = generated_text[len(prompt_text):].strip()
            
            return response
            
        except torch.cuda.OutOfMemoryError:
            logger.error("❌ GPU 메모리 부족!")
            raise RuntimeError(
                "GPU 메모리가 부족합니다. max_new_tokens를 줄이거나 배치 크기를 줄이세요."
            )
        except Exception as e:
            logger.error(f"❌ 생성 중 오류 발생: {e}")
            raise RuntimeError(f"텍스트 생성 실패: {e}")
    
    def chat(
        self,
        question: str,
        context: Optional[str] = None,
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
        # 프롬프트 포맷팅
        formatted_prompt = self.format_prompt(question, context)
        
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
            "base_model": self.base_model_name,
            "adapter_path": self.adapter_path,
            "device": self.device,
            "is_loaded": self.model is not None,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if self.model is not None and torch.cuda.is_available():
            info["vram_usage_gb"] = torch.cuda.memory_allocated() / 1024**3
        
        return info
    
    def __repr__(self):
        return f"LlamaGenerator(base={self.base_model_name}, loaded={self.model is not None})"


# ===== LocalRAGPipeline: chatbot_app.py 호환용 =====

class LocalRAGPipeline:
    """
    로컬 Llama 모델 기반 RAG 파이프라인
    
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
        from src.retriever.rag_retriever import RAGRetriever
        
        self.config = config or RAGConfig()
        self.top_k = top_k or self.config.DEFAULT_TOP_K
        
        # 검색 설정
        self.search_mode = self.config.DEFAULT_SEARCH_MODE
        self.alpha = self.config.DEFAULT_ALPHA
        
        # Retriever 초기화
        logger.info("RAGRetriever 초기화 중...")
        self.retriever = RAGRetriever(config=self.config)
        
        # LlamaGenerator 초기화
        logger.info("LlamaGenerator 초기화 중...")
        self.generator = LlamaGenerator(
            adapter_path=self.config.FINETUNED_ADAPTER_PATH,
            base_model_name=self.config.FINETUNED_BASE_MODEL,
            max_new_tokens=self.config.FINETUNED_MAX_NEW_TOKENS,
            temperature=self.config.FINETUNED_TEMPERATURE,
            top_p=self.config.FINETUNED_TOP_P,
            system_prompt=self.config.SYSTEM_PROMPT
        )
        
        # 모델 로드 (시간 소요)
        logger.info("로컬 모델 로드 중... (수 분 소요될 수 있습니다)")
        self.generator.load_model()
        
        # 대화 히스토리
        self.chat_history: List[Dict] = []
        
        # 마지막 검색 결과 저장 (sources 반환용)
        self._last_retrieved_docs = []
        
        logger.info("✅ LocalRAGPipeline 초기화 완료")
        logger.info(f"   - 검색 모드: {self.search_mode}")
        logger.info(f"   - 기본 top_k: {self.top_k}")
    
    def _retrieve_and_format(self, query: str) -> str:
        """
        검색 수행 및 컨텍스트 포맷팅
        
        Args:
            query: 검색 쿼리
        
        Returns:
            포맷된 컨텍스트 문자열
        """
        # 검색 모드에 따라 문서 검색
        if self.search_mode == "embedding":
            docs = self.retriever.search(query, top_k=self.top_k)
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
        """
        검색된 문서를 컨텍스트로 변환
        
        Args:
            retrieved_docs: 검색된 문서 리스트
        
        Returns:
            포맷된 컨텍스트
        """
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[문서 {i}]\n{doc['content']}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: list) -> list:
        """
        검색된 문서를 sources 형식으로 변환
        
        Args:
            retrieved_docs: 검색된 문서 리스트
        
        Returns:
            sources 리스트 (chatbot_app.py 호환 형식)
        """
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
        """
        토큰 사용량 추정
        
        로컬 모델이므로 정확한 토큰 수를 계산하기 어려워
        단어 수 기반으로 추정합니다.
        
        Args:
            query: 질문
            answer: 답변
        
        Returns:
            usage 딕셔너리
        """
        # 간단한 단어 수 기반 추정 (한국어는 대략 2배)
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
            search_mode: 검색 모드 ("embedding", "hybrid", "hybrid_rerank")
            alpha: 임베딩 가중치 (0~1)
        
        Returns:
            dict: answer, sources, search_mode, usage, elapsed_time
        """
        try:
            start_time = time.time()
            
            # 파라미터 설정
            if top_k is not None:
                self.top_k = top_k
            if search_mode is not None:
                self.search_mode = search_mode
            if alpha is not None:
                self.alpha = alpha
            
            # 검색 수행 및 컨텍스트 포맷팅
            context = self._retrieve_and_format(query)
            
            # 로컬 모델로 생성
            logger.info("로컬 모델로 답변 생성 중...")
            answer = self.generator.chat(
                question=query,
                context=context
            )
            
            elapsed_time = time.time() - start_time
            
            # 대화 히스토리에 추가
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # 결과 반환 (RAGPipeline과 동일 형식)
            return {
                'answer': answer,
                'sources': self._format_sources(self._last_retrieved_docs),
                'search_mode': self.search_mode,
                'elapsed_time': elapsed_time,
                'usage': self._estimate_usage(query, answer)
            }
        
        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"답변 생성 실패: {str(e)}") from e
    
    def chat(self, query: str) -> str:
        """
        간단한 대화 인터페이스
        
        Args:
            query: 질문
        
        Returns:
            str: 답변 텍스트만 반환
        """
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
        """
        검색 설정 변경
        
        Args:
            search_mode: 검색 모드
            top_k: 검색 문서 수
            alpha: 임베딩 가중치
        """
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
    
    def print_result(self, result: dict, query: str = None):
        """
        결과 출력 (디버깅용)
        
        Args:
            result: generate_answer() 결과
            query: 질문 (선택)
        """
        print("\n" + "="*60)
        if query:
            print(f"질문: {query}")
        print(f"검색 모드: {result.get('search_mode', 'N/A')}")
        if 'elapsed_time' in result:
            print(f"소요 시간: {result['elapsed_time']:.2f}초")
        print("="*60)
        print(f"\n💬 답변:\n{result['answer']}")
        print(f"\n📚 참고 문서 ({len(result['sources'])}개):")
        for i, source in enumerate(result['sources'], 1):
            score = source.get('score', 0)
            score_type = source.get('score_type', '')
            print(f"  [{i}] {source['filename']}")
            print(f"      점수: {score:.3f} ({score_type})")
        print("="*60)


# ===== 대화형 실행 모드 =====

def interactive_mode_local():
    """로컬 모델 대화형 모드 실행"""
    print("=" * 60)
    print("LocalRAGPipeline 대화형 모드 시작")
    print("=" * 60)
    
    # Config import
    from src.utils.config import RAGConfig
    config = RAGConfig()
    
    # LocalRAGPipeline 초기화
    print("\n초기화 중... (수 분 소요될 수 있습니다)\n")
    pipeline = LocalRAGPipeline(config=config)
    
    print("\n" + "=" * 60)
    print("명령어: 'quit' (종료), 'clear' (히스토리 초기화), 'mode' (검색모드 변경)")
    print("=" * 60)
    
    while True:
        user_query = input("\n질문: ").strip()
        
        if not user_query:
            continue
        
        if user_query.lower() in ['quit', 'exit', '종료', 'q']:
            print("시스템을 종료합니다.")
            break
        
        if user_query.lower() == 'clear':
            pipeline.clear_history()
            continue
        
        if user_query.lower() == 'mode':
            print("\n검색 모드 선택:")
            print("1. embedding - 임베딩 검색")
            print("2. hybrid - BM25 + 임베딩")
            print("3. hybrid_rerank - Hybrid + Re-ranker (권장)")
            choice = input("선택 (1/2/3): ").strip()
            modes = {'1': 'embedding', '2': 'hybrid', '3': 'hybrid_rerank'}
            if choice in modes:
                pipeline.set_search_config(search_mode=modes[choice])
            continue
        
        try:
            result = pipeline.generate_answer(query=user_query)
            pipeline.print_result(result, user_query)
            
            # 소스 출력 여부
            show_source = input("\n참조 문서 상세 보기? (y/n): ").strip().lower()
            if show_source == 'y':
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{'='*40}")
                    print(f"[문서 {i}] {source['filename']}")
                    print(f"발주기관: {source['organization']}")
                    print(f"내용:\n{source['content'][:500]}...")
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


# ===== 테스트용 =====

if __name__ == "__main__":
    # 대화형 모드 실행
    interactive_mode_local()