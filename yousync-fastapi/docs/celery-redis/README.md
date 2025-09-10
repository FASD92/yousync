# FastAPI 오디오 분석 서비스

FastAPI와 httpx를 활용한 완전 비동기 오디오 분석 백엔드 서비스입니다. 

## 🚀 주요 특징

- **완전 비동기 처리**: httpx와 FastAPI BackgroundTasks를 활용한 논블로킹 처리
- **S3 비동기 업로드**: ThreadPoolExecutor를 통한 S3 파일 업로드 최적화
- **실시간 진행 상황**: Server-Sent Events(SSE)를 통한 실시간 진행률 추적
- **DB 기반 상태 저장**: 작업 상태 및 결과를 AnalysisResult 테이블에 저장
- **RESTful API**: 직관적인 API 엔드포인트 설계

## 📁 프로젝트 구조

```
fast-api/
├── back-end/
│   ├── main.py                 # FastAPI 애플리케이션 진입점
│   ├── models.py              # 데이터베이스 모델 (Token, AnalysisResult 등)
│   ├── schemas.py             # Pydantic 스키마
│   ├── database.py            # 데이터베이스 연결 설정
│   ├── requirements.txt       # 의존성 패키지 목록
│   └── router/
│       ├── user_audio_router.py    # 오디오 분석 API (httpx 기반)
│       ├── script_router.py        # 스크립트 관련 API
│       ├── actor_router.py         # 배우 관련 API
│       └── token_router.py         # 토큰 관리 API
├── README.md                  # 이 파일
```

## 🛠️ 기술 스택
- **FastAPI**: 현대적이고 빠른 웹 프레임워크
- **httpx**: 완전 비동기 HTTP 클라이언트
- **boto3**: AWS S3 연동 (presigned URL 활용)
- **SQLAlchemy**: ORM
- **uvicorn**: ASGI 서버
- **AWS S3**: 오디오 파일 저장소
- **PostgreSQL**: 메인 데이터베이스
- **Railway**: 배포 플랫폼

## 🔧 설치 및 실행

### 1. 의존성 설치
```bash
cd back-end
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 예시
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET_NAME=your_s3_bucket_name
TARGET_SERVER_URL=http://your-analysis-server/analyze-voice
WEBHOOK_URL=https://your-domain/tokens/webhook/analysis-complete
DATABASE_URL=postgresql://username:password@localhost/dbname
```

### 3. 데이터베이스 마이그레이션
```bash
alembic upgrade head
```

### 4. 서버 실행
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API 엔드포인트

### 오디오 분석 API (`/tokens`)

#### 1. 오디오 업로드 및 분석 요청
```http
POST /tokens/{token_id}/upload-audio
Content-Type: multipart/form-data

Parameters:
- token_id (path): 토큰 ID
- file (form-data): 오디오 파일

Response:
{
  "message": "업로드 완료, 백그라운드에서 처리됩니다.",
  "job_id": "uuid-string",
  "status": "processing"
}
```

#### 2. 분석 결과 조회
```http
GET /tokens/analysis-result/{job_id}

Response:
{
  "status": "completed|processing|failed",
  "token_id": "string",
  "progress": 100,
  "result": {...},
  "message": "분석 완료"
}
```

#### 3. 실시간 진행 상황 (SSE)
```http
GET /tokens/analysis-progress/{job_id}
Accept: text/event-stream

Response Stream:
data: {"status": "processing", "progress": 40, "message": "S3 업로드 중..."}
data: {"status": "processing", "progress": 70, "message": "분석 서버 요청 중..."}
data: {"status": "completed", "progress": 100, "result": {...}}
```

#### 4. 웹훅 엔드포인트
```http
POST /tokens/webhook/analysis-complete?job_id=uuid-string

Request Body:
{
  "analysis_results": {...}
}
```

## 🏗️ 아키텍처

### 비동기 처리 흐름

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant S3
    participant AnalysisServer
    participant Webhook

    Client->>FastAPI: POST /tokens/{token_id}/upload-audio
    FastAPI->>Client: job_id (즉시 응답)
    Note over FastAPI: BackgroundTasks 시작
    FastAPI->>S3: 비동기 파일 업로드 (ThreadPoolExecutor)
    FastAPI->>AnalysisServer: httpx 비동기 요청 (presigned URL)
    Client->>FastAPI: GET /tokens/analysis-progress/{job_id} (SSE)
    FastAPI-->>Client: 실시간 진행 상황 스트림
    AnalysisServer->>Webhook: 분석 완료 알림
    FastAPI->>Client: 분석 완료 (SSE)
```

### 핵심 컴포넌트
- **httpx AsyncClient**: 외부 API 호출을 위한 완전 비동기 HTTP 클라이언트
- **ThreadPoolExecutor**: S3 업로드와 같은 I/O 작업의 비동기 처리
- **BackgroundTasks**: FastAPI의 백그라운드 작업 처리
- **Server-Sent Events**: 실시간 진행 상황 스트리밍
- **AnalysisResult 테이블**: 작업 상태 및 결과를 DB에 저장

## 🧪 테스트

### API 테스트 예시
```bash
# 오디오 업로드 테스트
curl -X POST "http://localhost:8000/tokens/1/upload-audio" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav"

# 결과 조회
curl http://localhost:8000/tokens/analysis-result/{job_id}

# SSE 스트림 테스트
curl -N http://localhost:8000/tokens/analysis-progress/{job_id}
```

## 🚧 개발 예정
- [ ] 인증/권한 시스템 강화
- [ ] 멀티파트 대용량 파일 업로드 지원
- [ ] 분석 결과 캐싱 시스템
- [ ] 모니터링 및 로깅 개선


해야할일
- 디비 테이블 추가(정민이형 요청)
- 로그인 회원가입 추가