from sqlalchemy import create_engine  # SQLAlchemy 엔진 생성용
from sqlalchemy.ext.declarative import declarative_base  # ORM 모델 베이스 클래스 생성용
from sqlalchemy.orm import sessionmaker  # 데이터베이스 세션 관리용
import os  # 환경변수 접근용
from dotenv import load_dotenv  # .env 파일 로드용

# .env 파일에서 환경변수들을 읽어와서 현재 환경에 로드
load_dotenv()

# PostgreSQL 데이터베이스 연결 설정
# 환경변수 DATABASE_URL이 있으면 사용, 없으면 기본값 사용
DATABASE_URL = os.getenv("DATABASE_URL")
# SQLite를 사용하려면 다음 줄의 주석을 해제하세요:
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# SQLAlchemy 데이터베이스 엔진 생성
# SQLite 사용시에는 멀티스레딩 제약을 해제하는 옵션이 필요
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # 👇 이렇게 수정합니다.
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,          # 기본 5 → 20으로 증가
        max_overflow=30,       # 추가 연결 허용
        pool_pre_ping=True,    # 연결 상태 확인
        pool_recycle=3600      # 1시간마다 연결 재생성
    )

# 데이터베이스 세션 팩토리 생성
# autocommit=False: 명시적으로 commit() 호출해야 변경사항 저장
# autoflush=False: 명시적으로 flush() 호출해야 변경사항 DB에 반영
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 모델들이 상속받을 베이스 클래스 생성
Base = declarative_base()

# FastAPI의 의존성 주입에서 사용할 데이터베이스 세션 함수
# 요청마다 새로운 세션을 생성하고, 요청 완료 후 자동으로 세션 닫기
def get_db():
    db = SessionLocal()  # 새 세션 생성
    try:
        yield db  # 세션을 API 함수에 전달
    finally:
        db.close()  # 요청 완료 후 세션 닫기 (리소스 정리)
