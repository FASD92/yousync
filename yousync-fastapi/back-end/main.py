"""
FastAPI 영화/스크립트 관리 API 서버

이 애플리케이션은 영화와 스크립트 데이터를 관리하는 REST API 서버입니다.
PostgreSQL 데이터베이스를 사용하여 데이터를 저장하고 관리합니다.

주요 기능:
- 스크립트 CRUD 작업 (생성, 조회, 수정, 삭제)
- 영화 CRUD 작업 (생성, 조회, 수정, 삭제)
- 장르별/감독별 영화 검색
- API 문서 자동 생성 (Swagger UI)
"""



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import boto3
import os
from starlette.middleware.cors import CORSMiddleware

# 데이터베이스 관련 임포트
from database import engine
from models import Base

# 라우터 임포트 - 각 도메인별로 분리된 API 엔드포인트들
from router.script_router import router as script_router
from router.token_router import router as token_router
from router.user_audio_router import router as user_audio_router
from router.auth_router import router as auth_router
from router.actor_router import router as actor_router
from router.mypage_router import router as mypage_router
from router.script_audio_router import router as script_audio_router
from router.url_router import router as url_router
from router.score_router import router as score_router
from router.youtube_process_router import router as youtube_process_router
from router.duet_router import router as duet_router
from router.synthesize_router import router as synthesize_router
from router.request_router import router as request_router

# 데이터베이스 테이블 생성 (앱 시작시 자동으로 테이블이 생성됨)
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행될 코드
    print("FastAPI 애플리케이션 시작...")
    
    # .env 파일이 로드된 후, S3 클라이언트를 안전하게 생성
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-northeast-2") # 안전한 기본값 설정
    )
    
    # 생성된 클라이언트를 앱 상태(state)에 저장하여 어디서든 접근 가능하게 함
    app.state.s3_client = s3_client
    
    yield # --- 이 지점에서 애플리케이션이 실행됨 ---
    
    # 앱 종료 시 실행될 코드 (정리 작업)
    print("FastAPI 애플리케이션 종료.")


# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="영화/스크립트 관리 API",
    description="영화와 스크립트 데이터를 관리하는 REST API 서버입니다.",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI 경로
    redoc_url="/redoc",     # ReDoc 경로
    lifespan=lifespan
)
# 허용할 프론트엔드 주소 목록
origins = [
    "http://localhost:3000",
    "https://yousync.link"
    # 실제 프로덕션 프론트엔드 주소가 있다면 추가
    # "https://your-frontend-domain.com", 
]


# CORS 미들웨어 설정 - 프론트엔드에서 API 호출을 허용하기 위함
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        
    allow_credentials=True,     # 쿠키/인증 정보 포함한 요청 허용
    allow_methods=["*"],        # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],        # 모든 헤더 허용

)

# # API 라우터 등록 - 각 도메인별로 분리된 엔드포인트들을 메인 앱에 연결
app.include_router(auth_router, prefix="/api")    # /auth 경로로 인증 관련 API 등록
app.include_router(actor_router, prefix="/api")   # /actors 경로로 배우 관련 API 등록
app.include_router(token_router, prefix="/api")   # /tokens 경로로 토큰 관련 API 등록
app.include_router(script_router, prefix="/api")  # /scripts 경로로 스크립트 관련 API 등록
app.include_router(user_audio_router, prefix="/api") # /tokens/{token_id}/upload-audio 경로로 유저 음성 데이터 관련 API 등록
app.include_router(mypage_router, prefix="/api") # /mapage 결로로 스크립트 관련 API 등록
app.include_router(script_audio_router, prefix="/api")
app.include_router(url_router, prefix="/api") # url 관련 라우터
app.include_router(score_router, prefix="/api") # /score 경로로 점수 관련 API 등록
app.include_router(youtube_process_router, prefix="/api") # /youtube 경로로 유튜브 전처리 관련 API 등록
app.include_router(duet_router, prefix="/api")    # /duet 경로로 듀엣 관련 API 등록
app.include_router(synthesize_router, prefix="/api") # /유저 음성 합성 API 등록
app.include_router(request_router, prefix="/api") # /유저 URL 요청 API 등록

# 루트 엔드포인트 - API 서버 상태 확인용
@app.get("/")
def read_root():
    """
    API 서버 상태를 확인하는 루트 엔드포인트입니다.
    서버가 정상적으로 작동하는지 확인할 때 사용합니다.
    """
    return {
        "message": "토큰/스크립트 관리 API 서버가 정상적으로 작동중입니다!",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0"
    }

# Health Check 엔드포인트 - 서버 상태 모니터링용
@app.get("/health")
def health_check():
    """
    서버 상태를 체크하는 헬스체크 엔드포인트입니다.
    로드밸런서나 모니터링 도구에서 서버 상태를 확인할 때 사용합니다.
    """
    return {"status": "healthy", "service": "token-script-api"}