from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List

from database import get_db
import models
import schemas

router = APIRouter(
    prefix="/duet",
    tags=["duet"],
)

@router.get("/scenes", response_model=List[schemas.DuetScene])
def get_duet_scenes(db: Session = Depends(get_db)):
    # 1. youtube_url 별로 토큰 개수를 세고, 개수가 2개인 그룹을 찾는다.
    duet_urls_query = (
        db.query(models.Token.youtube_url)
        .group_by(models.Token.youtube_url)
        .having(func.count(models.Token.id) == 2)
        .all()
    )
    
    # 쿼리 결과가 (url,) 형태의 튜플 리스트이므로 url만 추출
    duet_urls = [url for url, in duet_urls_query]

    if not duet_urls:
        return []

    # 2. 해당 URL에 속하는 모든 토큰 정보를 한 번의 쿼리로 가져온다.
    tokens_for_duets = (
        db.query(models.Token)
        .filter(models.Token.youtube_url.in_(duet_urls))
        .options(joinedload(models.Token.url)) # URL 정보도 함께 로드
        .order_by(models.Token.youtube_url, models.Token.id) # 정렬
        .all()
    )

    # 3. URL 별로 토큰을 그룹화하여 DuetScene 리스트를 만든다.
    duet_scenes = []
    current_url = None
    current_pair = []

    for token in tokens_for_duets:
        if token.youtube_url != current_url:
            # 새로운 URL 그룹 시작
            if current_url is not None and len(current_pair) == 2:
                # 이전 그룹을 DuetScene으로 만들어 추가
                scene = schemas.DuetScene(
                    youtube_url=current_url,
                    thumbnail_url=current_pair[0].thumbnail_url, # 첫 번째 토큰의 썸네일 사용
                    scene_title=current_pair[0].token_name, # 첫 번째 토큰의 이름을 대표 제목으로 사용
                    duet_pair=current_pair
                )
                duet_scenes.append(scene)
            
            # 현재 URL과 페어 초기화
            current_url = token.youtube_url
            current_pair = [token]
        else:
            # 같은 URL 그룹에 토큰 추가
            current_pair.append(token)

    # 마지막 그룹 처리
    if current_url is not None and len(current_pair) == 2:
        scene = schemas.DuetScene(
            youtube_url=current_url,
            thumbnail_url=current_pair[0].thumbnail_url,
            scene_title=current_pair[0].token_name,
            duet_pair=current_pair
        )
        duet_scenes.append(scene)

    return duet_scenes
