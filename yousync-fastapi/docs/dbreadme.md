# Railway → AWS RDS PostgreSQL 마이그레이션 히스토리

##  목차
1. [프로젝트 개요](#프로젝트-개요)
2. [마이그레이션 이유](#마이그레이션-이유)
3. [최종 구성](#최종-구성)
4. [단계별 작업 과정](#단계별-작업-과정)
5. [연결 정보](#연결-정보)
6. [비용 분석](#비용-분석)
7. [트러블슈팅](#트러블슈팅)
8. [유용한 명령어](#유용한-명령어)

---

## 프로젝트 개요

### 마이그레이션 목표
- **기존**: Railway PostgreSQL 데이터베이스
- **신규**: AWS RDS PostgreSQL 데이터베이스
- **이유**: EC2 서버 이전에 따른 데이터베이스 통합 관리

---

## 마이그레이션 이유

### Railway의 한계
- 프리티어 사용중 돈이 없다

### AWS RDS의 장점
- ✅ **자동 백업 및 복구**
- ✅ **자동 소프트웨어 패치**
- ✅ **모니터링 및 알림**
- ✅ **고가용성 옵션**
- ✅ **EC2와 동일한 VPC 내 위치**
- ✅ **AWS 통합 관리**

---

## 최종 구성

### EC2 서버 정보
```yaml
인스턴스 ID: i-07cf12442674a7847
이름: fastapi-server
타입: t3.medium (2 vCPU, 4GB RAM)
리전: ap-northeast-2 (서울)
VPC: vpc-04d9d2eb660784bb2
보안그룹: sg-0dbc983bbe527aa77 (mfa-sg)
공인 IP: 3.34.190.149
```

### RDS 인스턴스 정보
```yaml
식별자: yousync-db
엔진: PostgreSQL 15.8
인스턴스 클래스: db.t3.micro
스토리지: 20GB (gp2, 비암호화)
백업 보존: 7일
Multi-AZ: 비활성화 (비용 절약)
퍼블릭 액세스: 활성화
데이터베이스명: fastapi_db
마스터 사용자: yousync
비밀번호: 12345678
엔드포인트: yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com
포트: 5432
```

---

## 단계별 작업 과정

### 1단계: 요구사항 분석 
- PostgreSQL 버전 -> (psycopg2-binary==2.9.10)
- SQLAlchemy 버전 -> (2.0.41)
- 마이그레이션 파일 분석 및 복제 (산하형이 최신화 한거 받아서)

### 2단계: RDS 인스턴스 생성 
```bash
# AWS CLI 명령어 (참고용)
aws rds create-db-instance \
  --db-instance-identifier yousync-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.8 \
  --master-username yousync \
  --master-user-password 12345678 \
  --allocated-storage 20 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-0dbc983bbe527aa77 \
  --db-name fastapi_db \
  --backup-retention-period 7 \
  --region ap-northeast-2
```

### 3단계: 보안 그룹 설정 
```bash
# PostgreSQL 포트 5432 추가
aws ec2 authorize-security-group-ingress \
  --group-id sg-0dbc983bbe527aa77 \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0 \
  --region ap-northeast-2
```

### 4단계: 환경 변수 설정 (수정함)
**파일 위치**: `/Users/jeonghwan/development/fast-api/back-end/.env`

```env
# 데이터베이스 설정 (AWS RDS PostgreSQL)
DATABASE_URL=postgresql://yousync:12345678@yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com:5432/fastapi_db

# JWT 토큰 암호화를 위한 시크릿 키
JWT_SECRET_KEY=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

# 구글 OAuth2 설정
GOOGLE_CLIENT_ID=your-google-oauth2-client-id

# AWS S3 설정
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-s3-bucket-name

# 외부 분석 서버 설정
ANALYSIS_SERVER_URL=http://your-analysis-server.com
```

### 5단계: 데이터베이스 스키마 생성 

#### 마이그레이션 파일 문제 해결
- **문제**: Alembic 마이그레이션 파일 간 의존성 오류
- **해결**: models.py 기반 SQL 스크립트 직접 작성

#### 생성된 테이블 (10개)
1. **users** - 사용자 정보
2. **actors** - 배우 정보  
3. **urls** - YouTube URL 정보
4. **tokens** - 토큰 단위 데이터
5. **scripts** - 스크립트/대사 정보
6. **words** - 단어 단위 데이터
7. **actor_aliases** - 배우 별칭
8. **token_actors** - 토큰-배우 관계
9. **bookmarks** - 북마크 정보
10. **analysis_results** - 분석 결과

#### SQL 스크립트 실행
```bash
cd /Users/jeonghwan/development/fast-api/back-end
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
     -U yousync -d fastapi_db -f create_tables.sql
```

### 6단계: 연결 테스트 
```bash
# PostgreSQL 직접 연결 테스트
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
     -U yousync -d fastapi_db -c "SELECT version();"

# 결과: PostgreSQL 15.8 연결 성공 
```

---

## 🔗 연결 정보

### DATABASE_URL
```
postgresql://yousync:12345678@yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com:5432/fastapi_db
```

### 개별 연결 정보
```yaml
호스트: yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com
포트: 5432
데이터베이스: fastapi_db
사용자명: yousync
비밀번호: 12345678
```

### FastAPI 애플리케이션 연결
```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

---

## 비용 분석

### 월간 예상 비용 (서울 리전 기준)
```yaml
RDS db.t3.micro: $12-15
스토리지 20GB (gp2): $2-3
백업 스토리지: $1-2
총 예상 비용: $15-20/월
```

### 비용 최적화 설정
- **db.t3.micro** (최소 인스턴스)
- **20GB 스토리지** (최소 크기)
- **Multi-AZ 비활성화**
- **7일 백업 보존** (적정 수준)


---

## 🔧 트러블슈팅

### 1. 마이그레이션 파일 의존성 오류
**문제**: `actors` 테이블이 없는데 `urls` 테이블에서 참조
```
sqlalchemy.exc.ProgrammingError: relation "actors" does not exist
```

**해결**: 
- Alembic 마이그레이션 대신 SQL 스크립트 직접 작성
- models.py 기반으로 올바른 순서로 테이블 생성

### 2. 보안 그룹 연결 오류
**문제**: PostgreSQL 포트(5432) 접근 불가
```
psql: error: connection to server failed: Operation timed out
```

**해결**:
```bash
aws ec2 authorize-security-group-ingress \
  --group-id sg-0dbc983bbe527aa77 \
  --protocol tcp --port 5432 --cidr 0.0.0.0/0
```

### 3. Python 패키지 설치 오류
**문제**: `ModuleNotFoundError: No module named 'dotenv'`

**해결**: 
- 시스템 전역 설치 대신 SQL 스크립트 직접 사용
- 또는 가상환경 설정 후 패키지 설치

---

## aws 명령어 모음

### AWS CLI 명령어
```bash
# RDS 인스턴스 상태 확인
aws rds describe-db-instances --db-instance-identifier yousync-db --region ap-northeast-2

# RDS 인스턴스 삭제
aws rds delete-db-instance --db-instance-identifier yousync-db --region ap-northeast-2

# 보안 그룹 규칙 확인
aws ec2 describe-security-groups --group-ids sg-0dbc983bbe527aa77 --region ap-northeast-2
```

### PostgreSQL 명령어
```bash
# 데이터베이스 연결
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com -U yousync -d fastapi_db

# 테이블 목록 확인
\dt

# 테이블 구조 확인
\d table_name

# 데이터베이스 크기 확인
SELECT pg_size_pretty(pg_database_size('fastapi_db'));

# 연결 정보 확인
\conninfo
```

### FastAPI 개발 명령어
```bash
# 개발 서버 실행
cd /Users/jeonghwan/development/fast-api/back-end
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 의존성 설치
pip install -r requirements.txt

# 환경변수 확인
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('DATABASE_URL'))"
```

---

## 앞으로의 단계

### 1. 애플리케이션 배포 업데이트
- [ ] EC2 서버의 환경변수 업데이트
- [ ] FastAPI 애플리케이션 재시작
- [ ] API 엔드포인트 테스트

### 2. 데이터 마이그레이션 (필요시)
```bash
# Railway에서 데이터 덤프
pg_dump [RAILWAY_DATABASE_URL] > railway_dump.sql

# RDS로 데이터 복원
psql -h yousync-db.cj60es4aa7pz.ap-northeast-2.rds.amazonaws.com \
     -U yousync -d fastapi_db < railway_dump.sql
```

### 3. 모니터링 설정
- [ ] CloudWatch 알림 설정
- [ ] 성능 모니터링 활성화
- [ ] 백업 정책 검토

### 4. 보안 강화
- [ ] 비밀번호 복잡도 증가
- [ ] SSL/TLS 연결 강제
- [ ] 접근 IP 제한 (필요시)

---

## 참고 자료

### 공식 문서
- [AWS RDS PostgreSQL 문서](https://docs.aws.amazon.com/rds/latest/userguide/CHAP_PostgreSQL.html)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)
- [FastAPI 데이터베이스 가이드](https://fastapi.tiangolo.com/tutorial/sql-databases/)

### 프로젝트 파일
- **환경변수**: `/Users/jeonghwan/development/fast-api/back-end/.env`
- **모델 정의**: `/Users/jeonghwan/development/fast-api/back-end/models.py`
- **데이터베이스 설정**: `/Users/jeonghwan/development/fast-api/back-end/database.py`
- **SQL 스크립트**: `/Users/jeonghwan/development/fast-api/back-end/create_tables.sql`

---

## 마이그레이션 완료 체크리스트

- [x] **RDS 인스턴스 생성** (yousync-db)
- [x] **보안 그룹 설정** (PostgreSQL 포트 5432 개방)
- [x] **데이터베이스 스키마 생성** (10개 테이블)
- [x] **환경변수 설정** (.env 파일)
- [x] **연결 테스트** (PostgreSQL 15.8 연결 성공)
- [x] **문서화** (완전한 가이드 작성)
- [ ] **애플리케이션 배포 업데이트**
- [ ] **기존 데이터 마이그레이션** (필요시)
- [ ] **성능 테스트**
- [ ] **모니터링 설정**

---

## 결론

Railway에서 AWS RDS PostgreSQL로의 데이터베이스 마이그레이션 -> 성공적

### 주요 성과
- ✅ **안정적인 관리형 데이터베이스** 구축
- ✅ **EC2와 동일한 VPC 내 배치**로 네트워크 최적화
- ✅ **자동 백업 및 모니터링** 기능 활성화
- ✅ **비용 최적화** 설정 적용
- ✅ **완전한 문서화** 완료

