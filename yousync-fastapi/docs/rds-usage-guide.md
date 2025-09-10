# 🗄️ AWS RDS PostgreSQL 사용 가이드

## 📋 목차
1. [기본 연결 방법](#기본-연결-방법)
2. [데이터 조회하기](#데이터-조회하기)
3. [데이터 추가/수정/삭제](#데이터-추가수정삭제)
4. [테이블 관리](#테이블-관리)
5. [백업 및 복원](#백업-및-복원)
6. [성능 모니터링](#성능-모니터링)
7. [GUI 도구 사용](#gui-도구-사용)
8. [FastAPI와 연동](#fastapi와-연동)

---

## 🔌 기본 연결 방법

### 1. 터미널에서 직접 연결
```bash
# 기본 연결
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com -U yousync -d fastapi_db

# 비밀번호 입력: 12345678
```

### 2. 연결 정보 확인
```sql
-- 현재 연결 정보
\conninfo

-- 데이터베이스 버전
SELECT version();

-- 현재 시간
SELECT NOW();
```

### 3. 기본 명령어
```sql
-- 테이블 목록 보기
\dt

-- 특정 테이블 구조 보기
\d users

-- 데이터베이스 목록
\l

-- 종료
\q
```

---

## 📊 데이터 조회하기

### 1. 기본 조회
```sql
-- 모든 사용자 보기
SELECT * FROM users;

-- 특정 컬럼만 보기
SELECT id, email, full_name FROM users;

-- 조건부 조회
SELECT * FROM users WHERE is_active = true;

-- 정렬해서 보기
SELECT * FROM users ORDER BY created_at DESC;

-- 개수 제한
SELECT * FROM users LIMIT 10;
```

### 2. 관계형 데이터 조회
```sql
-- 배우와 URL 함께 보기
SELECT a.name as actor_name, u.youtube_url 
FROM actors a 
JOIN urls u ON a.id = u.actor_id;

-- 토큰과 스크립트 함께 보기
SELECT t.token_name, s.script 
FROM tokens t 
JOIN scripts s ON t.id = s.token_id 
LIMIT 5;

-- 사용자의 북마크 보기
SELECT u.full_name, t.token_name 
FROM users u 
JOIN bookmarks b ON u.id = b.user_id 
JOIN tokens t ON b.token_id = t.id;
```

### 3. 집계 함수 사용
```sql
-- 총 사용자 수
SELECT COUNT(*) FROM users;

-- 배우별 URL 개수
SELECT a.name, COUNT(u.youtube_url) as url_count 
FROM actors a 
LEFT JOIN urls u ON a.id = u.actor_id 
GROUP BY a.name;

-- 가장 많이 본 토큰 (view_count 기준)
SELECT token_name, view_count 
FROM tokens 
ORDER BY view_count DESC 
LIMIT 10;
```

---

## ✏️ 데이터 추가/수정/삭제

### 1. 데이터 추가 (INSERT)
```sql
-- 새 사용자 추가
INSERT INTO users (email, full_name, google_id, is_active) 
VALUES ('test@example.com', '테스트 사용자', 'google123', true);

-- 새 배우 추가
INSERT INTO actors (name) 
VALUES ('김철수');

-- 여러 데이터 한번에 추가
INSERT INTO actors (name) VALUES 
('이영희'),
('박민수'),
('최지은');
```

### 2. 데이터 수정 (UPDATE)
```sql
-- 사용자 정보 수정
UPDATE users 
SET full_name = '수정된 이름' 
WHERE email = 'test@example.com';

-- 토큰 조회수 증가
UPDATE tokens 
SET view_count = view_count + 1 
WHERE id = 1;

-- 여러 조건으로 수정
UPDATE users 
SET is_active = false 
WHERE created_at < '2024-01-01' AND login_type = 'guest';
```

### 3. 데이터 삭제 (DELETE)
```sql
-- 특정 사용자 삭제
DELETE FROM users WHERE email = 'test@example.com';

-- 조건부 삭제
DELETE FROM tokens WHERE view_count = 0;

-- 주의: 모든 데이터 삭제 (위험!)
-- DELETE FROM users; -- 절대 실행하지 마세요!
```

---

## 🏗️ 테이블 관리

### 1. 새 테이블 생성
```sql
-- 예시: 카테고리 테이블 추가
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 추가
CREATE INDEX idx_categories_name ON categories (name);
```

### 2. 기존 테이블 수정
```sql
-- 새 컬럼 추가
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 컬럼 타입 변경
ALTER TABLE users ALTER COLUMN phone TYPE VARCHAR(30);

-- 컬럼 삭제
ALTER TABLE users DROP COLUMN phone;

-- 제약조건 추가
ALTER TABLE users ADD CONSTRAINT unique_email UNIQUE (email);
```

### 3. 테이블 정보 확인
```sql
-- 테이블 크기 확인
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'users';

-- 인덱스 확인
SELECT indexname, indexdef 
FROM pg_indexes 
WHERE tablename = 'users';
```

---

## 💾 백업 및 복원

### 1. 데이터 백업
```bash
# 전체 데이터베이스 백업
pg_dump -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
        -U yousync -d fastapi_db > backup_$(date +%Y%m%d).sql

# 특정 테이블만 백업
pg_dump -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
        -U yousync -d fastapi_db -t users > users_backup.sql

# 스키마만 백업 (데이터 제외)
pg_dump -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
        -U yousync -d fastapi_db --schema-only > schema_backup.sql
```

### 2. 데이터 복원
```bash
# 백업 파일로부터 복원
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
     -U yousync -d fastapi_db < backup_20250704.sql

# 특정 테이블만 복원
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
     -U yousync -d fastapi_db < users_backup.sql
```

### 3. CSV 파일로 내보내기/가져오기
```sql
-- CSV로 내보내기
COPY users TO '/tmp/users.csv' DELIMITER ',' CSV HEADER;

-- CSV에서 가져오기
COPY users FROM '/tmp/users.csv' DELIMITER ',' CSV HEADER;
```

---

## 📈 성능 모니터링

### 1. 현재 활동 확인
```sql
-- 현재 실행 중인 쿼리
SELECT pid, usename, application_name, state, query 
FROM pg_stat_activity 
WHERE state = 'active';

-- 데이터베이스 통계
SELECT datname, numbackends, xact_commit, xact_rollback 
FROM pg_stat_database 
WHERE datname = 'fastapi_db';
```

### 2. 테이블 사용량 확인
```sql
-- 테이블별 크기
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 테이블별 접근 통계
SELECT schemaname, tablename, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch 
FROM pg_stat_user_tables;
```

### 3. 느린 쿼리 찾기
```sql
-- 쿼리 실행 계획 보기
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';

-- 인덱스 사용률 확인
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename = 'users';
```

---

## 🖥️ GUI 도구 사용

### 1. pgAdmin 4 (추천)
```bash
# 설치 (macOS)
brew install --cask pgadmin4

# 연결 정보
Host: yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com
Port: 5432
Database: fastapi_db
Username: yousync
Password: 12345678
```

### 2. DBeaver (무료)
```bash
# 설치 (macOS)
brew install --cask dbeaver-community

# 연결 타입: PostgreSQL
# 위와 동일한 연결 정보 사용
```

### 3. TablePlus (유료, 깔끔한 UI)
```bash
# 설치 (macOS)
brew install --cask tableplus
```

---

## 🚀 FastAPI와 연동

### 1. 기본 CRUD 작업
```python
# models.py에서 정의된 모델 사용
from sqlalchemy.orm import Session
from models import User, Actor, Token

# 사용자 생성
def create_user(db: Session, email: str, full_name: str):
    db_user = User(email=email, full_name=full_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# 사용자 조회
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# 사용자 목록
def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()
```

### 2. API 엔드포인트 예시
```python
# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db

app = FastAPI()

@app.get("/users/")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = get_users(db, skip=skip, limit=limit)
    return users

@app.post("/users/")
def create_user_endpoint(email: str, full_name: str, db: Session = Depends(get_db)):
    return create_user(db=db, email=email, full_name=full_name)

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user
```

### 3. 데이터베이스 연결 테스트
```python
# test_db.py
from database import engine
from sqlalchemy import text

def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ 데이터베이스 연결 성공!")
            return True
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

---

## 🔧 유용한 스크립트

### 1. 데이터 초기화 스크립트
```python
# init_data.py
from database import SessionLocal
from models import Actor, User

def init_sample_data():
    db = SessionLocal()
    try:
        # 샘플 배우 데이터
        actors = [
            Actor(name="김철수"),
            Actor(name="이영희"),
            Actor(name="박민수")
        ]
        
        for actor in actors:
            db.add(actor)
        
        # 샘플 사용자 데이터
        users = [
            User(email="user1@example.com", full_name="사용자1"),
            User(email="user2@example.com", full_name="사용자2")
        ]
        
        for user in users:
            db.add(user)
            
        db.commit()
        print("✅ 샘플 데이터 생성 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_sample_data()
```

### 2. 데이터 정리 스크립트
```python
# cleanup_data.py
from database import SessionLocal
from models import User, Token
from datetime import datetime, timedelta

def cleanup_old_data():
    db = SessionLocal()
    try:
        # 30일 이전 비활성 사용자 삭제
        cutoff_date = datetime.now() - timedelta(days=30)
        old_users = db.query(User).filter(
            User.is_active == False,
            User.created_at < cutoff_date
        ).all()
        
        for user in old_users:
            db.delete(user)
            
        # 조회수 0인 오래된 토큰 삭제
        old_tokens = db.query(Token).filter(
            Token.view_count == 0
        ).all()
        
        for token in old_tokens:
            db.delete(token)
            
        db.commit()
        print(f"✅ 정리 완료: 사용자 {len(old_users)}명, 토큰 {len(old_tokens)}개 삭제")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    cleanup_old_data()
```

---

## ⚠️ 주의사항

### 1. 보안
- **절대 프로덕션에서 `DELETE FROM table;` 같은 명령어 사용 금지**
- 중요한 작업 전에는 반드시 백업
- 비밀번호를 코드에 하드코딩하지 말 것

### 2. 성능
- 대량 데이터 작업 시 배치 처리 사용
- 인덱스가 있는 컬럼으로 WHERE 조건 사용
- `SELECT *` 대신 필요한 컬럼만 조회

### 3. 트랜잭션
```python
# 올바른 트랜잭션 사용
def transfer_data(db: Session):
    try:
        # 여러 작업을 하나의 트랜잭션으로
        user = create_user(db, "test@example.com", "테스트")
        actor = create_actor(db, "새 배우")
        db.commit()  # 모든 작업이 성공하면 커밋
    except Exception as e:
        db.rollback()  # 오류 발생 시 롤백
        raise e
```

---

## 📞 문제 해결

### 연결 문제
```bash
# 연결 테스트
telnet yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com 5432

# DNS 확인
nslookup yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com
```

### 성능 문제
```sql
-- 현재 락 확인
SELECT * FROM pg_locks WHERE NOT granted;

-- 긴 실행 쿼리 확인
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

---

## 🎯 마무리

이제 AWS RDS PostgreSQL을 효과적으로 사용할 수 있습니다!

### 기억할 점
- **항상 백업 먼저**
- **테스트 환경에서 먼저 시도**
- **성능 모니터링 정기적으로 확인**
- **보안 규칙 준수**

### 다음 단계
1. 실제 데이터로 테스트해보기
2. 모니터링 대시보드 설정
3. 자동 백업 스케줄 확인
4. 성능 최적화 적용

---

**작성일**: 2025-07-04  
**버전**: 1.0  
**상태**: 완료 ✅
