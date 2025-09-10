# app/routers/url_router.py

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from schemas import UrlExistsResponse, UrlCheckRequest, Token
from models import URL
from database import get_db

router = APIRouter(prefix="/urls", tags=["urls"])

@router.post("/check", response_model=UrlExistsResponse)
def check_url(request: UrlCheckRequest, db: Session = Depends(get_db)):
    exists = (
        db.query(URL)
          .filter(URL.youtube_url == request.youtube_url)
          .first() is not None
    )
    return UrlExistsResponse(exists=exists)
    

@router.get("/tokens", response_model=List[Token])
def list_tokens(
    youtube_url: str = Query(..., description="검색할 YouTube URL"),
    db: Session = Depends(get_db),
):
    url_obj = db.query(URL).filter(URL.youtube_url == youtube_url).first()
    if not url_obj:
        raise HTTPException(404, "URL not found")
    return url_obj.tokens
