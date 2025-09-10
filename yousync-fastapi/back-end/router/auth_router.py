"""
인증 관련 API 라우터

구글 OAuth2 로그인/회원가입과 JWT 토큰 발급/검증을 처리합니다.

주요 기능:
- 구글 OAuth2를 통한 회원가입/로그인
- JWT 토큰 발급 및 검증
- 사용자 인증 상태 확인
- 토큰 갱신(refresh) 기능
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.auth.transport import requests
from google.oauth2 import id_token
from sqlalchemy.orm import Session

from database import get_db
from models import User
from schemas import GoogleLoginRequest, AuthToken, UserResponse

# 라우터 인스턴스 생성
router = APIRouter(
    prefix="/auth",
    tags=["인증"],
    responses={404: {"description": "Not found"}},
)

# JWT 설정
# 환경변수에서 불러오기 (반드시 .env에 설정되어 있어야 함)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_fallback_value")


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# HTTP Bearer 토큰 스키마
security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    JWT 액세스 토큰을 생성합니다.
    
    Args:
        data: 토큰에 포함할 데이터 (사용자 ID 등)
        expires_delta: 토큰 만료 시간 (기본값: 30분)
    
    Returns:
        str: 생성된 JWT 토큰
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    JWT 토큰을 검증하고 페이로드를 반환합니다.
    
    Args:
        token: 검증할 JWT 토큰
    
    Returns:
        dict: 토큰 페이로드
    
    Raises:
        HTTPException: 토큰이 유효하지 않은 경우
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰이 만료되었습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    현재 인증된 사용자를 반환합니다.
    
    Args:
        credentials: HTTP Authorization Bearer 토큰
        db: 데이터베이스 세션
    
    Returns:
        User: 현재 사용자 모델
    
    Raises:
        HTTPException: 인증에 실패한 경우
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰에서 사용자 정보를 찾을 수 없습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


@router.post("/google-login", response_model=AuthToken, summary="구글 로그인/회원가입")
async def google_login(
    google_request: GoogleLoginRequest,
    db: Session = Depends(get_db)
):
    """
    구글 OAuth2 ID 토큰을 통한 로그인/회원가입을 처리합니다.
    
    프론트엔드에서 구글 OAuth2를 통해 받은 id_token을 전달하면,
    해당 토큰을 검증하고 사용자 정보를 추출하여 회원가입/로그인을 처리합니다.
    
    Flow:
    1. 구글 id_token 검증
    2. 구글 계정 정보 추출 (이메일, 이름, 프로필 사진 등)
    3. 기존 사용자인지 확인 (google_id 또는 email 기준)
    4. 신규 사용자면 회원가입, 기존 사용자면 로그인 처리
    5. JWT 액세스 토큰 발급
    
    Args:
        google_request: 구글 id_token을 포함한 요청 데이터
        db: 데이터베이스 세션
    
    Returns:
        AuthToken: JWT 액세스 토큰과 토큰 타입
    
    Raises:
        HTTPException: 구글 토큰 검증 실패 또는 기타 오류
    """
    try:
        # 구글 OAuth2 클라이언트 ID가 설정되지 않은 경우 에러
        if not GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="구글 OAuth2 설정이 필요합니다. GOOGLE_CLIENT_ID 환경변수를 설정해주세요."
            )
        
        # 구글 id_token 검증
        idinfo = id_token.verify_oauth2_token(
            google_request.id_token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        # 구글 계정 정보 추출
        google_id = idinfo['sub']
        email = idinfo['email']
        full_name = idinfo.get('name', '')
        profile_picture = idinfo.get('picture', '')
        
        # 기존 사용자 확인 (google_id 우선, 그 다음 email)
        user = db.query(User).filter(User.google_id == google_id).first()
        
        if not user:
            # google_id로 찾지 못했으면 email로 재시도
            user = db.query(User).filter(User.email == email).first()
            
            if user:
                # 기존 이메일 계정이 있으면 구글 정보 업데이트
                user.google_id = google_id
                user.login_type = "google"
                user.profile_picture = profile_picture
                if not user.full_name:  # 이름이 없으면 구글에서 가져온 이름으로 설정
                    user.full_name = full_name
            else:
                # 완전히 새로운 사용자 - 회원가입 처리
                user = User(
                    email=email,
                    full_name=full_name,
                    google_id=google_id,
                    login_type="google",
                    profile_picture=profile_picture,
                )
                db.add(user)
        else:
            # 기존 구글 사용자 - 프로필 정보 업데이트
            user.full_name = full_name
            user.profile_picture = profile_picture
        
        db.commit()
        db.refresh(user)
        
        # JWT 액세스 토큰 생성
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email},
            expires_delta=access_token_expires
        )
        
        return AuthToken(
            access_token=access_token,
            token_type="bearer"
        )
        
    except ValueError as e:
        # 구글 토큰 검증 실패
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"유효하지 않은 구글 토큰입니다: {str(e)}"
        )
    except Exception as e:
        # 기타 서버 오류
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"로그인 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/me", response_model=UserResponse, summary="현재 사용자 정보 조회")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    현재 인증된 사용자의 정보를 조회합니다.
    
    JWT 토큰을 통해 인증된 사용자의 기본 정보를 반환합니다.
    
    Args:
        current_user: 현재 인증된 사용자 (의존성 주입)
    
    Returns:
        UserResponse: 사용자 정보 (비밀번호 제외)
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        google_id=current_user.google_id,
        full_name=current_user.full_name,
        profile_picture=current_user.profile_picture,
        is_active=current_user.is_active,
        login_type=current_user.login_type,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )


@router.post("/refresh/", response_model=AuthToken, summary="토큰 갱신")
async def refresh_token(
    current_user: User = Depends(get_current_user)
):
    """
    JWT 액세스 토큰을 갱신합니다.
    
    기존 토큰이 유효한 상태에서 새로운 토큰을 발급받을 수 있습니다.
    토큰 만료 전에 미리 갱신하는 용도로 사용합니다.
    
    Args:
        current_user: 현재 인증된 사용자 (의존성 주입)
    
    Returns:
        AuthToken: 새로운 JWT 액세스 토큰
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(current_user.id), "email": current_user.email},
        expires_delta=access_token_expires
    )
    
    return AuthToken(
        access_token=access_token,
        token_type="bearer"
    )


@router.post("/logout", summary="로그아웃")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """
    사용자 로그아웃을 처리합니다.
    
    현재는 클라이언트 측에서 토큰을 삭제하는 것으로 충분하지만,
    향후 토큰 블랙리스트 등의 기능을 추가할 수 있습니다.
    
    Args:
        current_user: 현재 인증된 사용자 (의존성 주입)
    
    Returns:
        dict: 로그아웃 성공 메시지
    """
    return {
        "message": "성공적으로 로그아웃되었습니다",
        "user_id": current_user.id
    }