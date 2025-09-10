# routes/video_request.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from models import VideoRequest, User
from schemas import VideoRequestCreate, VideoRequestResponse, VideoRequestStatusUpdate
from database import get_db
from router.auth_router import get_current_user
# from utils import email.py

router = APIRouter(prefix="/request", tags=["VideoRequest"])

@router.post("/", response_model=VideoRequestResponse)
def create_request(
    request: VideoRequestCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    existing = db.query(VideoRequest).filter(
        VideoRequest.user_id == current_user.id,
        VideoRequest.status == "심사중"
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="이미 심사 중인 요청이 존재합니다.")

    video_request = VideoRequest(
        user_id=current_user.id,
        actor=request.actor,
        content=request.content,
        url=request.url,
        status="심사중"
    )
    db.add(video_request)
    db.commit()
    db.refresh(video_request)

    return VideoRequestResponse(
        id=video_request.id,
        actor=video_request.actor,
        content=video_request.content,
        url=video_request.url,
        status=video_request.status,
        date=video_request.created_at,
        requester=current_user.full_name or current_user.email
    )

@router.get("/mine", response_model=list[VideoRequestResponse])
def get_my_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    requests = db.query(VideoRequest).filter(
        VideoRequest.user_id == current_user.id
    ).all()

    return [
        VideoRequestResponse(
            id=req.id,
            actor=req.actor,
            content=req.content,
            url=req.url,
            status=req.status,
            date=req.created_at,
            requester=current_user.full_name or current_user.email
        )
        for req in requests
    ]

@router.get("/all", response_model=list[VideoRequestResponse])
def get_all_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="관리자만 접근할 수 있습니다.")

    requests = db.query(VideoRequest).all()

    return [
        VideoRequestResponse(
            id=req.id,
            actor=req.actor,
            content=req.content,
            url=req.url,
            status=req.status,
            date=req.created_at,
            requester=req.user.full_name or req.user.email
        )
        for req in requests
    ]

@router.patch("/{id}/status")
def update_status(
    id: int,
    status_update: VideoRequestStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="관리자만 상태 변경 가능")

    req = db.query(VideoRequest).filter(VideoRequest.id == id).first()
    if not req:
        raise HTTPException(status_code=404, detail="요청을 찾을 수 없습니다")

    if status_update.status not in {"심사중", "승인됨", "거절됨"}:
        raise HTTPException(status_code=400, detail="유효하지 않은 상태입니다.")

    req.status = status_update.status
    db.commit()

    # 이메일 전송
    # send_status_email(
    #     to_email=req.user.email,
    #     actor=req.actor,
    #     new_status=req.status
    # )

    return {"detail": "상태가 변경되었습니다."}


@router.delete("/{id}")
def delete_request(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    req = db.query(VideoRequest).filter(
        VideoRequest.id == id,
        VideoRequest.user_id == current_user.id
    ).first()
    if not req:
        raise HTTPException(status_code=404, detail="해당 요청이 없습니다.")
    if req.status != "거절됨":
        raise HTTPException(status_code=400, detail="거절된 요청만 삭제할 수 있습니다.")

    db.delete(req)
    db.commit()
    return {"detail": "요청이 삭제되었습니다."}
