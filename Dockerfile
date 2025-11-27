# ===== API-Only Dockerfile (Fixed) =====
FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# ===== 환경변수 설정 (권한 문제 해결) =====
ENV HOME=/app
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
ENV OMP_NUM_THREADS=4

# 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/.cache/huggingface /app/.streamlit && \
    chmod -R 777 /app/.cache /app/.streamlit

# 의존성 복사
COPY requirements.txt .

# pip 업그레이드 & 의존성 설치
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# Streamlit 포트
EXPOSE 7860

# 실행
CMD ["streamlit", "run", "src/visualization/chatbot_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]