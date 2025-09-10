# routers/mypage.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, delete, func, desc

from database import get_db
from router.auth_router import get_current_user
from models import Bookmark, Token, User, AnalysisResult, Script
from schemas import (
    BookmarkCreate, BookmarkOut, BookmarkListOut, TokenInfo,
    MyDubbedTokenResponse, TokenAnalysisStatusResponse, 
    MyPageOverviewResponse, UserResponse
)

router = APIRouter(
    prefix="/mypage", 
    tags=["MyPage"]
)


# --- 북마크 생성 ---------------------------------------
@router.post(
    "/bookmarks",
    response_model=BookmarkOut,
    status_code=status.HTTP_201_CREATED,
)
def create_bookmark(
    data: BookmarkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1) Token 존재 검증
    token_exists = db.query(Token).filter(Token.id == data.token_id).first()
    if not token_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found",
        )
    
    # 1-1) 이미 북마크가 존재하는지 명시적으로 확인
    existing_bookmark = db.query(Bookmark).filter(
        Bookmark.user_id == current_user.id,
        Bookmark.token_id == data.token_id
    ).first()
    
    if existing_bookmark:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Already bookmarked - Detected by explicit check",
        )

    # 2) Bookmark 삽입
    bm = Bookmark(user_id=current_user.id, token_id=data.token_id)
    db.add(bm)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        # 이미 북마크된 경우 (더 자세한 에러 메시지 제공)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Already bookmarked - IntegrityError: {str(e)}",
        )
    db.refresh(bm)
    return bm


# --- 북마크 삭제 ---------------------------------------
@router.delete(
    "/bookmarks/{token_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_bookmark(
    token_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 먼저 북마크가 존재하는지 확인
    bookmark = db.query(Bookmark).filter(
        Bookmark.user_id == current_user.id,
        Bookmark.token_id == token_id,
    ).first()
    
    if not bookmark:
        # 디버깅을 위해 404 대신 200 응답으로 변경하고 메시지 추가
        return {
            "status": "not_found",
            "message": f"북마크를 찾을 수 없습니다. user_id: {current_user.id}, token_id: {token_id}",
            "user_id": current_user.id
        }
    
    # 북마크 삭제
    db.delete(bookmark)
    try:
        db.commit()
        return {
            "status": "success",
            "message": "북마크가 성공적으로 삭제되었습니다."
        }
    except Exception as e:
        db.rollback()
        return {
            "status": "error",
            "message": f"북마크 삭제 중 오류 발생: {str(e)}"
        }


# --- 북마크 목록 조회 (토큰 정보 포함) -----------------------------------
@router.get(
    "/bookmarks",
    response_model=List[BookmarkListOut],
)
def list_bookmarks(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 디버깅을 위해 먼저 사용자 ID와 북마크 수를 확인
    bookmark_count = db.query(func.count(Bookmark.id)).filter(
        Bookmark.user_id == current_user.id
    ).scalar()
    
    # JOIN을 통해 토큰 정보도 함께 가져오기
    bookmarks = (
        db.query(Bookmark)
        .options(joinedload(Bookmark.token))  # Token 정보를 함께 로드
        .filter(Bookmark.user_id == current_user.id)
        .order_by(Bookmark.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    
    # 응답 데이터 구성
    result = []
    for bookmark in bookmarks:
        result.append(BookmarkListOut(
            id=bookmark.id,  # 실제 ID 사용
            user_id=bookmark.user_id,
            token_id=bookmark.token_id,
            created_at=bookmark.created_at,
            token=TokenInfo(
                id=bookmark.token.id,
                token_name=bookmark.token.token_name,
                actor_name=bookmark.token.actor_name,
                category=bookmark.token.category,
                thumbnail_url=getattr(bookmark.token, 'thumbnail_url', None),
                youtube_url=getattr(bookmark.token, 'youtube_url', None)  # youtube_url 추가
            )
        ))
    
    # 디버깅 정보 추가
    if not result:
        # response_model을 무시하고 디버깅 정보 반환
        return JSONResponse(content={
            "debug_info": {
                "user_id": current_user.id,
                "bookmark_count": bookmark_count,
                "message": "북마크가 없거나 조회 중 문제가 발생했습니다."
            }
        })
    
    return result


@router.get(
    "/my-dubbed-tokens",
    response_model=List[MyDubbedTokenResponse],
    summary="내가 더빙한 토큰 목록 조회"
)
def get_my_dubbed_tokens(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    현재 로그인한 유저가 더빙한 토큰들의 목록을 반환합니다.
    각 토큰별로 마지막 더빙 시간, 전체 스크립트 수, 완료된 스크립트 수를 포함합니다.
    """
    # 내가 더빙한 토큰들 조회 (현재 DB 구조에 맞게 수정)
    subquery = (
        db.query(
            Token.id.label('token_id'),
            Token.token_name,
            Token.actor_name, 
            Token.category,
            Token.youtube_url, # youtube_url 추가
            func.max(AnalysisResult.created_at).label('last_dubbed_at'),
            func.count(func.distinct(Script.id)).label('total_scripts'),
            func.count(AnalysisResult.id).label('completed_scripts')
        )
        .join(AnalysisResult, AnalysisResult.token_id == Token.id)
        .join(Script, Script.token_id == Token.id)
        .filter(AnalysisResult.user_id == current_user.id)
        .group_by(Token.id, Token.token_name, Token.actor_name, Token.category, Token.youtube_url) # youtube_url 추가
        .order_by(desc('last_dubbed_at'))
        .limit(limit)
        .offset(offset)
        .all()
    )
    
    # 결과 변환
    results = []
    for row in subquery:
        results.append(MyDubbedTokenResponse(
            token_id=row.token_id,
            token_name=row.token_name,
            actor_name=row.actor_name,
            category=row.category,
            youtube_url=row.youtube_url, # youtube_url 추가
            last_dubbed_at=row.last_dubbed_at,
            total_scripts=row.total_scripts,
            completed_scripts=row.completed_scripts
        ))
    
    return results


@router.get(
    "/tokens/{token_id}/analysis-status",
    response_model=TokenAnalysisStatusResponse,
    summary="특정 토큰의 내 분석 상태 확인"
)
def get_token_analysis_status(
    token_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    특정 토큰에 대한 현재 유저의 분석 결과 존재 여부와 상세 정보를 반환합니다.
    더빙 기록이 있으면 결과창으로, 없으면 더빙 시작 화면으로 분기하는데 사용됩니다.
    """
    # 해당 토큰의 스크립트들과 내 분석 결과 조회 (현재 DB 구조에 맞게 수정)
    results = (
        db.query(Script, AnalysisResult)
        .outerjoin(AnalysisResult, 
                  (AnalysisResult.token_id == token_id) & 
                  (AnalysisResult.user_id == current_user.id))
        .filter(Script.token_id == token_id)
        .all()
    )
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Token not found"
        )
    
    # 분석 결과 정리
    script_results = []
    has_any_analysis = False
    
    for script, analysis in results:
        if analysis:
            has_any_analysis = True
            script_results.append({
                "script_id": script.id,
                "script_text": script.script,
                "has_result": True,
                "job_id": analysis.job_id,
                "status": analysis.status,
                "result": analysis.result,
                "created_at": analysis.created_at
            })
        else:
            script_results.append({
                "script_id": script.id,
                "script_text": script.script,
                "has_result": False
            })
    
    return TokenAnalysisStatusResponse(
        token_id=token_id,
        has_analysis=has_any_analysis,
        script_results=script_results
    )


@router.delete(
    "/tokens/{token_id}/my-results",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="재더빙을 위한 기존 분석 결과 삭제"
)
def delete_my_token_results(
    token_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    재더빙 시 해당 토큰의 모든 기존 분석 결과를 삭제합니다.
    한 유저당 동일 토큰의 분석결과는 1세트만 존재하도록 보장합니다.
    """
    # 해당 토큰의 내 분석 결과들 삭제 (현재 DB 구조에 맞게 수정)
    deleted_count = (
        db.query(AnalysisResult)
        .filter(
            AnalysisResult.token_id == token_id,
            AnalysisResult.user_id == current_user.id
        )
        .delete(synchronize_session=False)
    )
    
    db.commit()
    
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="삭제할 분석 결과가 없습니다."
        )


# --- 마이페이지 통합 정보 API (새로 구현) -------------
@router.get(
    "/overview",
    response_model=MyPageOverviewResponse,
    summary="마이페이지 통합 정보 조회"
)
def get_mypage_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    마이페이지에 필요한 모든 정보를 한 번에 반환
    - 사용자 기본 정보
    - 북마크 개수
    - 더빙한 토큰 개수  
    - 총 연습 횟수
    - 평균 완성도
    - 최근 북마크 목록 (5개)
    - 최근 더빙한 토큰 목록 (5개)
    """
    
    # 1. 사용자 기본 정보
    user_info = UserResponse(
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
    
    # 2. 북마크 개수
    total_bookmarks = db.query(func.count(Bookmark.token_id)).filter(
        Bookmark.user_id == current_user.id
    ).scalar() or 0
    
    # 3. 더빙한 토큰 개수 (중복 제거) - 현재 DB 구조에 맞게 수정
    total_dubbed_tokens = db.query(
        func.count(func.distinct(AnalysisResult.token_id))
    ).filter(AnalysisResult.user_id == current_user.id).scalar() or 0
    
    # 4. 총 연습 횟수 (분석 결과 개수)
    total_practice_count = db.query(func.count(AnalysisResult.id)).filter(
        AnalysisResult.user_id == current_user.id
    ).scalar() or 0
    
    # 5. 평균 완성도 계산 (JSON에서 점수 추출)
    analysis_results = db.query(AnalysisResult.result).filter(
        AnalysisResult.user_id == current_user.id,
        AnalysisResult.result.isnot(None)
    ).all()
    
    total_score = 0
    score_count = 0
    for result in analysis_results:
        if result.result and isinstance(result.result, dict):
            # result JSON에서 점수 추출 (구조에 따라 조정 필요)
            score = result.result.get('total_score') or result.result.get('score', 0)
            if score:
                total_score += float(score)
                score_count += 1
    
    average_completion_rate = (total_score / score_count) if score_count > 0 else 0.0
    
    # 6. 최근 북마크 목록 (5개)
    recent_bookmarks_query = (
        db.query(Bookmark)
        .options(joinedload(Bookmark.token))
        .filter(Bookmark.user_id == current_user.id)
        .order_by(Bookmark.created_at.desc())
        #.limit(5)
        .all()
    )
    
    recent_bookmarks = []
    for bookmark in recent_bookmarks_query:
        recent_bookmarks.append(BookmarkListOut(
            id=bookmark.id,  # 실제 ID 사용
            user_id=bookmark.user_id,
            token_id=bookmark.token_id,
            created_at=bookmark.created_at,
            token=TokenInfo(
                id=bookmark.token.id,
                token_name=bookmark.token.token_name,
                actor_name=bookmark.token.actor_name,
                category=bookmark.token.category,
                thumbnail_url=getattr(bookmark.token, 'thumbnail_url', None),
                youtube_url=getattr(bookmark.token, 'youtube_url', None)  # youtube_url 추가
            )
        ))
    
    # 7. 최근 더빙한 토큰 목록 (5개) - 현재 DB 구조에 맞게 수정
    recent_dubbed_query = (
        db.query(
            Token.id.label('token_id'),
            Token.token_name,
            Token.actor_name, 
            Token.category,
            Token.youtube_url, # youtube_url 추가
            func.max(AnalysisResult.created_at).label('last_dubbed_at'),
            func.count(func.distinct(Script.id)).label('total_scripts'),
            func.count(AnalysisResult.id).label('completed_scripts')
        )
        .join(AnalysisResult, AnalysisResult.token_id == Token.id)
        .join(Script, Script.token_id == Token.id)
        .filter(AnalysisResult.user_id == current_user.id)
        .group_by(Token.id, Token.token_name, Token.actor_name, Token.category, Token.youtube_url) # youtube_url 추가
        .order_by(desc('last_dubbed_at'))
        #.limit(5)
        .all()
    )
    
    recent_dubbed_tokens = []
    for row in recent_dubbed_query:
        recent_dubbed_tokens.append(MyDubbedTokenResponse(
            token_id=row.token_id,
            token_name=row.token_name,
            actor_name=row.actor_name,
            category=row.category,
            youtube_url=row.youtube_url, # youtube_url 추가
            last_dubbed_at=row.last_dubbed_at,
            total_scripts=row.total_scripts,
            completed_scripts=row.completed_scripts
        ))
    
    return MyPageOverviewResponse(
        user_info=user_info,
        total_bookmarks=total_bookmarks,
        total_dubbed_tokens=total_dubbed_tokens,
        total_practice_count=total_practice_count,
        average_completion_rate=round(average_completion_rate, 2),
        recent_bookmarks=recent_bookmarks,
        recent_dubbed_tokens=recent_dubbed_tokens
    )
