from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# === Actor Schemas ===
class ActorBase(BaseModel):
    name: str

class ActorCreate(ActorBase):
    pass

class Actor(ActorBase):
    id: int
    
    class Config:
        from_attributes = True


# === ScriptWord ===
class ScriptWordBase(BaseModel):
    script_id: int # script_id를 기반으로 연결
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    word: Optional[str] = None
    probability: Optional[float] = None


class ScriptWord(ScriptWordBase):
    id: int

    class Config:
        from_attributes = True


# 분석용 데이터
class ScriptWordUser(BaseModel):
    id: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    word: Optional[str] = None
    # JSON 컬럼의 2D 리스트를 받기 위한 필드
    mfcc: Optional[List[List[float]]] = Field(
        None,
        description="Frame-by-frame 13-dim MFCC vectors"
    )

    class Config: 
        from_attributes = True


# === Script Schemas ===
class ScriptBase(BaseModel):
    token_id: int  #  변경: token_id를 기반으로 연결
    start_time: float
    end_time: float
    script: str
    translation: Optional[str] = None


class ScriptCreate(ScriptBase):
    pass


class Script(ScriptBase):
    id: int
    # actor: Optional[Actor] = None  # 배우 정보 포함
    words: List[ScriptWord] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


# 분석용 데이터 Script
class ScriptUser(BaseModel):
    id: int 
    words: List[ScriptWordUser] = Field(default_factory=list)

    class Config:
        from_attributes = True



# === Token Schemas ===
class TokenBase(BaseModel):
    token_name: str
    actor_name: str
    category: Optional[str] = None
    start_time: float
    end_time: float
    s3_textgrid_url: Optional[str] = None
    s3_pitch_url: Optional[str] = None
    s3_bgvoice_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    youtube_url: Optional[str] = None
    view_count: int = Field(0, description="누적 조회수")

class TokenCreate(TokenBase):
    pass

class Token(TokenBase):
    id: int
    # scripts: List[Script] = []  # 영화에 속한 스크립트 목록
    
    class Config:
        from_attributes = True

class TokenDetail(TokenBase):
    id: int
    bgvoice_url: Optional[str] = None   # presigned URL 또는 퍼블릭 URL
    pitch: Optional[Any] = None         # pitch.json 딕셔너리
    scripts: List[Script] = Field(default_factory=list)

    class Config:
        from_attributes = True   

class ViewCountResponse(BaseModel):
    token_id: int
    view_count: int

    class Config:
        from_attributes = True   



# === Url Schemas ===
class UrlCheckRequest(BaseModel):
    youtube_url: str

class UrlExistsResponse(BaseModel):
    exists: bool

    class Config:
        orm_mode = True



# === User Schemas ===
class GoogleLoginRequest(BaseModel):
    id_token: str


class UserResponse(BaseModel):
    """사용자 정보 응답 스키마 (비밀번호 제외)"""
    id: int
    email: str
    google_id: Optional[str] = None
    full_name: Optional[str] = None
    profile_picture: Optional[str] = None
    is_active: bool = True
    login_type: str = "email"
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class AuthToken(BaseModel):
    """인증 토큰 스키마"""
    access_token: str
    token_type: str



# === Bookmark Schemas ===
class BookmarkCreate(BaseModel):
    token_id: int = Field(..., ge=1)

class BookmarkOut(BaseModel):
    token_id: int

    class Config:
        from_attributes = True

# 토큰 정보를 포함한 북마크 정보 (프론트엔드 요구사항)
class TokenInfo(BaseModel):
    id: int
    token_name: str
    actor_name: str
    category: Optional[str] = None
    thumbnail_url: Optional[str] = None
    youtube_url: Optional[str] = None  # youtube_url 필드 추가
    
    class Config:
        from_attributes = True

class BookmarkListOut(BaseModel):
    id: int
    user_id: int
    token_id: int
    created_at: datetime
    token: TokenInfo

    class Config:
        from_attributes = True


# === My Dubbed Tokens Schemas ===
class MyDubbedTokenResponse(BaseModel):
    token_id: int
    token_name: str
    actor_name: str
    category: Optional[str] = None
    youtube_url: Optional[str] = None # youtube_url 추가
    last_dubbed_at: datetime
    total_scripts: int
    completed_scripts: int
    
    class Config:
        from_attributes = True


class TokenAnalysisStatusResponse(BaseModel):
    token_id: int
    has_analysis: bool
    script_results: List[dict] = Field(default_factory=list)
    
    class Config:
        from_attributes = True
        

# === AnalysisResult Schemas ===
class TokenScore(BaseModel):
    token_id: int 
    average_score: float



class UserScore(BaseModel):
    user_id: int 
    average_score: float


class UserToken(BaseModel):
    user_id: int
    tokens: List[Token]

    class Config:
        from_attributes = True


# === MyPage Overview Schemas ===
class MyPageOverviewResponse(BaseModel):
    user_info: UserResponse
    total_bookmarks: int
    total_dubbed_tokens: int
    total_practice_count: int
    average_completion_rate: float
    recent_bookmarks: List[BookmarkListOut] = Field(default_factory=list)
    recent_dubbed_tokens: List[MyDubbedTokenResponse] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


# === Leaderboard Schemas ===
class TopUser(BaseModel):
    user_id: int
    email: str
    full_name: Optional[str] = None
    profile_picture: Optional[str] = None
    recording_count: int

    class Config:
        from_attributes = True

class LeaderboardResponse(BaseModel):
    users: List[TopUser]


# === Youtube Process Schemas ===
class YoutubeProcessRequest(BaseModel):
    youtube_url: str
    movie_name: str
    actor_name: str

class YoutubeProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str
    token_id: Optional[int] = None

class YoutubeProcessStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    token_id: Optional[int] = None
    token_ids: Optional[List[int]] = None
    result: Optional[Any] = None


# === Duet Schemas ===
class DuetScene(BaseModel):
    youtube_url: str
    thumbnail_url: Optional[str] = None
    scene_title: str
    duet_pair: List[Token]

    class Config:
        from_attributes = True


# === videoRequestCreate
class VideoRequestCreate(BaseModel):
    actor: str
    content: str
    url: str

class VideoRequestResponse(BaseModel):
    id: int
    actor: str
    content: str
    url: str
    status: str
    date: datetime
    requester: Optional[str]

    class Config:
        orm_mode = True

class VideoRequestStatusUpdate(BaseModel):
    status: str 


class DubbingUrlResponse(BaseModel):
    url: str 


class AudioURL(BaseModel):
    script_id: int
    url: str

class UserAudioResponse(BaseModel):
    audios: List[AudioURL] 


