# app/routers/score_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, desc
from sqlalchemy.orm import Session
from typing import Optional, List

from database import get_db
from models import AnalysisResult, User
from schemas import TokenScore, UserScore, LeaderboardResponse, TopUser
from router.auth_router import get_current_user


router = APIRouter(prefix="/score", tags=["score"])


@router.get(
    "/{token_id}/score/",
    response_model=TokenScore,
    summary="특정 토큰의 내 평균 점수 조회"
)
def get_my_token_score(
    token_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> TokenScore:
    """
    특정 토큰에 대한 현재 유저의 평균 점수를 계산합니다.
    스크립트별 분석 결과들의 overall_score 평균을 반환합니다.
    """
    # 해당 토큰의 스크립트들에 대한 내 분석 결과 평균 계산
    from models import Script
    
    stmt = (
        select(func.avg(
            (AnalysisResult.result["result"]["overall_score"]).as_float()
        ))
        .join(Script, AnalysisResult.script_id == Script.id)
        .where(Script.token_id == token_id)
        .where(AnalysisResult.user_id == current_user.id)
    )
    avg_score: Optional[float] = db.execute(stmt).scalar()
    
    if avg_score is None:
        raise HTTPException(status_code=404, detail="해당 토큰에 대한 분석 결과가 없습니다.")
    
    return TokenScore(
        token_id=token_id,
        average_score=round(avg_score, 2)
    )


@router.get(
    "/me/average/",
    response_model=UserScore,
    summary="내 전체 평균 점수 조회"
)
def get_my_overall_average_score(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> UserScore:
    """
    현재 유저의 모든 분석 결과에 대한 전체 평균 점수를 계산합니다.
    """
    stmt = (
        select(func.avg(
            (AnalysisResult.result["result"]["overall_score"]).as_float()
        ))
        .where(AnalysisResult.user_id == current_user.id)
    )
    avg_score: Optional[float] = db.execute(stmt).scalar()

    if avg_score is None:
        raise HTTPException(
            status_code=404,
            detail="분석 결과가 없습니다."
        )

    return UserScore(
        user_id=current_user.id,
        average_score=round(avg_score, 2)
    )

@router.get("/leaderboard/top-recorders", response_model=LeaderboardResponse, summary="녹음 횟수 랭킹 TOP 3")
def get_top_recorders(db: Session = Depends(get_db)):
    """
    가장 많은 문장을 녹음한 상위 3명의 사용자 정보를 반환합니다.
    """
    top_users_query = (
        db.query(
            User.id,
            User.email,
            User.full_name,
            User.profile_picture,
            func.count(AnalysisResult.id).label("recording_count"),
        )
        .join(AnalysisResult, User.id == AnalysisResult.user_id)
        .group_by(User.id)
        .order_by(desc("recording_count"))
        .limit(3)
        .all()
    )

    top_users = [
        TopUser(
            user_id=user.id,
            email=user.email,
            full_name=user.full_name,
            profile_picture=user.profile_picture,
            recording_count=user.recording_count,
        )
        for user in top_users_query
    ]

    return LeaderboardResponse(users=top_users)

