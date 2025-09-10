# 🎵 YouSync FastAPI 포스트세션 완벽 대비 가이드

## 📋 목차
1. [프로젝트 개요 & 아키텍처](#프로젝트-개요--아키텍처)
2. [핵심 기술 스택 & 선택 이유](#핵심-기술-스택--선택-이유)
3. [비동기 처리 구현 방식](#비동기-처리-구현-방식)
4. [데이터베이스 설계 & ERD](#데이터베이스-설계--erd)
5. [API 설계 & 라우터 구조](#api-설계--라우터-구조)
6. [파일 업로드 & S3 연동](#파일-업로드--s3-연동)
7. [인증 & 보안](#인증--보안)
8. [실시간 통신 (SSE)](#실시간-통신-sse)
9. [성능 최적화](#성능-최적화)
10. [배포 & 인프라](#배포--인프라)
11. [예상 질문 & 답변](#예상-질문--답변)

---

## 🎯 프로젝트 개요 & 아키텍처

### 서비스 개요
**YouSync**는 영화/드라마 스크립트와 사용자 오디오를 분석하여 **발음, 억양, 감정**을 평가하는 AI 기반 더빙 연습 서비스입니다.

### 시스템 아키텍처
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Analysis      │
│   (React)       │◄──►│   Server        │◄──►│   Server        │
│                 │    │   (Python)      │    │   (AI Model)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   Database      │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   AWS S3        │
                       │   File Storage  │
                       └─────────────────┘
```

### 핵심 워크플로우
1. **사용자 로그인** → Google OAuth 2.0
2. **스크립트 선택** → 영화/드라마 대사 선택
3. **오디오 녹음/업로드** → 사용자 음성 파일
4. **비동기 분석** → AI 서버로 분석 요청
5. **실시간 진행상황** → SSE로 진행률 스트리밍
6. **결과 저장** → 분석 결과 DB 저장 및 마이페이지 제공

---

## 🛠️ 핵심 기술 스택 & 선택 이유

### Backend Framework
```python
# FastAPI 선택 이유
- 자동 API 문서 생성 (Swagger UI)
- 타입 힌팅 기반 검증
- 비동기 처리 네이티브 지원
- 높은 성능 (Starlette + Pydantic)
```

### 데이터베이스
```python
# PostgreSQL + SQLAlchemy
- 관계형 데이터 구조 (스크립트-토큰-분석결과)
- ACID 트랜잭션 지원
- JSON 컬럼 지원 (분석 결과 저장)
- Alembic 마이그레이션
```

### 비동기 처리
```python
# 현재 구현: BackgroundTasks + ThreadPoolExecutor
# 향후 계획: Celery + Redis
```

### 클라우드 & 스토리지
```python
# AWS S3: 오디오 파일 저장
# EC2: 서버 호스팅
# (향후) CloudWatch: 모니터링
```

---

## ⚡ 비동기 처리 구현 방식

### 1. 현재 구현: FastAPI BackgroundTasks
```python
# user_audio_router.py
@router.post("/tokens/{token_id}/upload-audio")
async def upload_audio(
    token_id: int,
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 즉시 job_id 반환
    job_id = str(uuid.uuid4())
    
    # 백그라운드에서 비동기 처리
    background_tasks.add_task(
        process_audio_analysis,
        job_id, token_id, audio_file, db
    )
    
    return {"job_id": job_id, "status": "processing"}
```

### 2. S3 업로드 비동기 처리
```python
# ThreadPoolExecutor로 S3 업로드 병렬 처리
import concurrent.futures

async def upload_to_s3_async(file_content, s3_key):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            upload_to_s3_sync, 
            file_content, s3_key
        )
    return result
```

### 3. 실시간 진행상황 (Server-Sent Events)
```python
# SSE로 실시간 진행률 스트리밍
@router.get("/tokens/analysis-progress/{job_id}")
async def get_analysis_progress(job_id: str):
    async def event_stream():
        while True:
            progress = get_job_progress(job_id)
            yield f"data: {json.dumps(progress)}\n\n"
            
            if progress["status"] == "completed":
                break
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_stream(), 
        media_type="text/plain"
    )
```

### 4. 웹훅 기반 완료 알림
```python
# 분석 서버에서 완료 시 웹훅 호출
@router.post("/tokens/webhook/analysis-complete")
async def analysis_complete_webhook(webhook_data: dict):
    job_id = webhook_data["job_id"]
    result = webhook_data["result"]
    
    # DB에 결과 저장
    save_analysis_result(job_id, result)
    
    # 진행상황 업데이트
    update_job_status(job_id, "completed")
```

---

## 🗄️ 데이터베이스 설계 & ERD

### 핵심 테이블 구조
```python
# models.py 주요 테이블들

class Token(Base):
    """스크립트 토큰 (문장 단위)"""
    __tablename__ = "tokens"
    
    id = Column(Integer, primary_key=True)
    script_id = Column(Integer, ForeignKey("scripts.id"))
    actor_id = Column(Integer, ForeignKey("actors.id"))
    text = Column(Text, nullable=False)
    start_time = Column(Float)  # 시작 시간
    end_time = Column(Float)    # 종료 시간
    s3_url = Column(String)     # 원본 오디오 S3 URL
    
class AnalysisResult(Base):
    """오디오 분석 결과"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token_id = Column(Integer, ForeignKey("tokens.id"))
    job_id = Column(String, unique=True)
    
    # 분석 결과 (JSON 형태)
    pronunciation_score = Column(Float)
    intonation_score = Column(Float)
    emotion_score = Column(Float)
    overall_score = Column(Float)
    detailed_feedback = Column(JSON)
    
    user_audio_s3_url = Column(String)  # 사용자 오디오 S3 URL
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 관계 설계
```
Users (1) ←→ (N) AnalysisResults
Scripts (1) ←→ (N) Tokens
Tokens (1) ←→ (N) AnalysisResults
Actors (1) ←→ (N) Tokens
Users (1) ←→ (N) Bookmarks ←→ (1) Tokens
```

---

## 🔌 API 설계 & 라우터 구조

### 라우터 분리 전략
```python
# main.py - 도메인별 라우터 분리
app.include_router(auth_router, prefix="/api")        # 인증
app.include_router(token_router, prefix="/api")       # 토큰/스크립트
app.include_router(user_audio_router, prefix="/api")  # 오디오 분석 (핵심)
app.include_router(mypage_router, prefix="/api")      # 마이페이지
app.include_router(script_audio_router, prefix="/api") # 스크립트 오디오
```

### 핵심 API 엔드포인트
```python
# 1. 오디오 분석 (핵심 기능)
POST /api/tokens/{token_id}/upload-audio
GET  /api/tokens/analysis-result/{job_id}
GET  /api/tokens/analysis-progress/{job_id}  # SSE
POST /api/tokens/webhook/analysis-complete

# 2. 마이페이지
GET  /api/mypage/bookmarks/
POST /api/mypage/bookmarks/
GET  /api/mypage/my-dubbed-tokens
GET  /api/mypage/overview

# 3. 인증
POST /api/auth/google
POST /api/auth/refresh
```

### API 응답 표준화
```python
# schemas.py - Pydantic 모델로 응답 표준화
class AnalysisResultResponse(BaseModel):
    job_id: str
    status: str
    pronunciation_score: Optional[float]
    intonation_score: Optional[float]
    emotion_score: Optional[float]
    overall_score: Optional[float]
    detailed_feedback: Optional[dict]
    created_at: datetime
```

---

## 📁 파일 업로드 & S3 연동

### S3 클라이언트 설정
```python
# main.py - 앱 시작시 S3 클라이언트 초기화
@asynccontextmanager
async def lifespan(app: FastAPI):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-northeast-2")
    )
    app.state.s3_client = s3_client
    yield
```

### 파일 업로드 처리
```python
# utils_s3.py - S3 업로드 유틸리티
async def upload_audio_to_s3(
    s3_client, 
    file_content: bytes, 
    bucket_name: str, 
    s3_key: str
) -> str:
    """오디오 파일을 S3에 비동기 업로드"""
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(
            executor,
            lambda: s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType="audio/wav"
            )
        )
    
    return f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"
```

### 파일 검증
```python
# 업로드 파일 검증
ALLOWED_AUDIO_TYPES = ["audio/wav", "audio/mp3", "audio/m4a"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_audio_file(file: UploadFile):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(400, "지원하지 않는 오디오 형식입니다")
    
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "파일 크기가 너무 큽니다 (최대 50MB)")
```

---

## 🔐 인증 & 보안

### Google OAuth 2.0 구현
```python
# auth_router.py
@router.post("/auth/google")
async def google_auth(auth_data: GoogleAuthRequest, db: Session = Depends(get_db)):
    # Google ID 토큰 검증
    idinfo = id_token.verify_oauth2_token(
        auth_data.id_token, 
        requests.Request(), 
        GOOGLE_CLIENT_ID
    )
    
    # 사용자 정보 추출
    email = idinfo['email']
    name = idinfo['name']
    
    # DB에서 사용자 조회 또는 생성
    user = get_or_create_user(db, email, name)
    
    # JWT 토큰 생성
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user
    }
```

### JWT 토큰 관리
```python
# dependencies.py
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_id(db, user_id=int(user_id))
    if user is None:
        raise credentials_exception
    return user
```

### 보안 설정
```python
# CORS 설정
origins = [
    "http://localhost:3000",
    "https://yousync.link"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수로 민감 정보 관리
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
```

---

## 📡 실시간 통신 (SSE)

### Server-Sent Events 구현
```python
# token_router.py
@router.get("/tokens/analysis-progress/{job_id}")
async def get_analysis_progress(job_id: str):
    """실시간 분석 진행상황 스트리밍"""
    
    async def event_stream():
        while True:
            # 진행상황 조회
            progress = analysis_progress.get(job_id, {
                "status": "not_found",
                "progress": 0
            })
            
            # SSE 형식으로 데이터 전송
            yield f"data: {json.dumps(progress)}\n\n"
            
            # 완료되면 스트림 종료
            if progress.get("status") in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)  # 1초마다 업데이트
    
    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
```

### 프론트엔드 SSE 연동
```javascript
// 프론트엔드에서 SSE 연결
const eventSource = new EventSource(`/api/tokens/analysis-progress/${jobId}`);

eventSource.onmessage = function(event) {
    const progress = JSON.parse(event.data);
    updateProgressBar(progress.progress);
    
    if (progress.status === 'completed') {
        eventSource.close();
        showResults(progress.result);
    }
};
```

---

## ⚡ 성능 최적화

### 1. 데이터베이스 최적화
```python
# database.py - 연결 풀 설정
engine = create_engine(
    DATABASE_URL,
    pool_size=20,          # 기본 연결 수
    max_overflow=30,       # 추가 연결 허용
    pool_pre_ping=True,    # 연결 상태 확인
    pool_recycle=3600      # 1시간마다 연결 재생성
)
```

### 2. 비동기 처리 최적화
```python
# 현재: BackgroundTasks
# 향후 계획: Celery + Redis
from celery import Celery

celery_app = Celery(
    "yousync",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def analyze_audio_task(job_id: str, audio_data: bytes):
    # 오디오 분석 작업
    result = analyze_audio(audio_data)
    return result
```

### 3. 캐싱 전략
```python
# 향후 Redis 캐싱 도입 계획
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@cache(expire=3600)  # 1시간 캐싱
async def get_popular_scripts():
    # 인기 스크립트 조회 (자주 조회되는 데이터)
    return db.query(Script).order_by(Script.view_count.desc()).limit(10).all()
```

---

## 🚀 배포 & 인프라

### 현재 배포 환경
```bash
# AWS EC2 배포
- 플랫폼: Amazon EC2 (Ubuntu)
- 웹서버: uvicorn (단일 워커)
- 데이터베이스: PostgreSQL
- 파일 저장소: AWS S3
- 도메인: EC2 퍼블릭 IP
```

### 환경 설정
```bash
# .env 파일
DATABASE_URL=postgresql://user:password@host:port/dbname
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=yousync-audio-files
TARGET_SERVER_URL=http://analysis-server-url
WEBHOOK_URL=http://your-domain/api/tokens/webhook/analysis-complete
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
SECRET_KEY=your_jwt_secret_key
```

### 서버 실행
```bash
# 개발 환경
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 환경 (향후 개선 계획)
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## ❓ 예상 질문 & 답변

### 🔥 기술적 질문

**Q1: 왜 FastAPI를 선택했나요?**
```
A: 1) 자동 API 문서 생성으로 개발 효율성 증대
   2) 타입 힌팅 기반 검증으로 런타임 에러 감소
   3) 비동기 처리 네이티브 지원으로 I/O 집약적 작업에 적합
   4) Pydantic을 통한 강력한 데이터 검증
   5) 높은 성능 (Starlette 기반)
```

**Q2: 비동기 처리를 어떻게 구현했나요?**
```
A: 현재는 FastAPI BackgroundTasks + ThreadPoolExecutor 조합:
   1) 오디오 업로드 요청 시 즉시 job_id 반환
   2) BackgroundTasks로 S3 업로드 + 분석 서버 요청
   3) ThreadPoolExecutor로 S3 업로드 병렬 처리
   4) SSE로 실시간 진행상황 스트리밍
   5) 웹훅으로 완료 알림 처리
   
   향후 Celery + Redis로 확장 예정
```

**Q3: 실시간 통신은 어떻게 구현했나요?**
```
A: Server-Sent Events (SSE) 사용:
   1) WebSocket 대신 SSE 선택 이유: 단방향 통신으로 충분
   2) StreamingResponse로 실시간 진행률 전송
   3) 1초마다 진행상황 업데이트
   4) 완료 시 자동으로 스트림 종료
```

**Q4: 데이터베이스 설계에서 고려한 점은?**
```
A: 1) 정규화: 스크립트-토큰-분석결과 관계 설계
   2) JSON 컬럼: 분석 결과의 유연한 구조 저장
   3) 인덱싱: user_id, token_id 등 자주 조회되는 컬럼
   4) 외래키 제약: 데이터 무결성 보장
   5) 타임스탬프: 생성/수정 시간 추적
```

**Q5: 보안은 어떻게 처리했나요?**
```
A: 1) Google OAuth 2.0: 안전한 소셜 로그인
   2) JWT 토큰: Stateless 인증
   3) CORS 설정: 허용된 도메인만 접근
   4) 환경 변수: 민감 정보 분리
   5) 파일 검증: 업로드 파일 타입/크기 제한
```

### 🚀 성능 & 확장성 질문

**Q6: 동시 사용자가 많아지면 어떻게 대응할 건가요?**
```
A: 1) 수직 확장: EC2 인스턴스 타입 업그레이드
   2) 수평 확장: 로드 밸런서 + 다중 인스턴스
   3) 데이터베이스: 읽기 전용 복제본 추가
   4) 캐싱: Redis 도입으로 DB 부하 감소
   5) CDN: 정적 파일 배포 최적화
```

**Q7: 현재 성능 병목점은 무엇인가요?**
```
A: 1) S3 업로드: 대용량 오디오 파일 업로드 시간
   2) 분석 서버: AI 모델 처리 시간
   3) 단일 워커: uvicorn 단일 프로세스 실행
   4) DB 연결: 동시 연결 수 제한
   
   해결 방안: 멀티파트 업로드, 워커 수 증가, 연결 풀 최적화
```

### 🔧 개발 & 운영 질문

**Q8: 테스트는 어떻게 하고 있나요?**
```
A: 1) 수동 테스트: request_test.py로 API 테스트
   2) 더미 데이터: add_dummy_data.py로 테스트 데이터 생성
   3) 향후 계획: pytest + TestClient로 자동화 테스트
```

**Q9: 모니터링은 어떻게 하고 있나요?**
```
A: 1) 현재: 서버 로그 + htop으로 리소스 모니터링
   2) 향후 계획: CloudWatch 알람 + Grafana 대시보드
   3) 에러 추적: Sentry 도입 예정
```

**Q10: 배포 프로세스는 어떻게 되나요?**
```
A: 1) 현재: 수동 배포 (git pull + 서버 재시작)
   2) 향후 계획: GitHub Actions CI/CD 파이프라인
   3) 무중단 배포: Blue-Green 배포 전략
```

### 💡 비즈니스 & 기획 질문

**Q11: 이 프로젝트의 핵심 가치는 무엇인가요?**
```
A: 1) AI 기반 발음/억양 분석으로 정확한 피드백 제공
   2) 실제 영화/드라마 스크립트로 실용적 학습
   3) 실시간 진행상황으로 사용자 경험 향상
   4) 개인화된 학습 기록 및 진도 관리
```

**Q12: 확장 가능성은 어떻게 보시나요?**
```
A: 1) 다국어 지원: 영어, 중국어, 일본어 등
   2) 실시간 대화: 다른 사용자와 대화 연습
   3) 게임화: 점수 시스템, 랭킹, 배지 등
   4) 교육 기관 연동: 학교/학원 대상 B2B 서비스
```

---

## 🎯 포스트세션 핵심 포인트

### 강조할 기술적 성과
1. **비동기 처리**: BackgroundTasks + SSE로 사용자 경험 향상
2. **확장 가능한 아키텍처**: 도메인별 라우터 분리, 모듈화
3. **실시간 통신**: SSE로 분석 진행상황 실시간 제공
4. **클라우드 연동**: S3 + EC2 활용한 확장 가능한 인프라
5. **보안**: OAuth 2.0 + JWT 기반 안전한 인증

### 개선 계획 어필
1. **성능 최적화**: Celery + Redis 도입
2. **모니터링**: CloudWatch + Grafana 대시보드
3. **CI/CD**: GitHub Actions 자동 배포
4. **테스트**: pytest 기반 자동화 테스트
5. **캐싱**: Redis 캐싱으로 응답 속도 향상

### 학습한 점 강조
1. **비동기 프로그래밍**: Python asyncio, FastAPI 비동기 처리
2. **클라우드 서비스**: AWS S3, EC2 실제 운영 경험
3. **API 설계**: RESTful API 설계 원칙 적용
4. **데이터베이스**: PostgreSQL, SQLAlchemy ORM 활용
5. **실시간 통신**: SSE 구현 및 프론트엔드 연동

---

**💡 TIP: 포스트세션에서는 기술적 구현보다는 "왜 이렇게 설계했는지", "어떤 문제를 해결했는지"에 초점을 맞춰 설명하세요!**
