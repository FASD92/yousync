from sqlalchemy import Column, Integer, String, Float, Text, JSON, Boolean, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class Token(Base):
    __tablename__ = "tokens"  # 토큰 단위 (token_id)

    id = Column(Integer, primary_key=True, index=True)  # token_id
    
    token_name = Column(String, nullable=False)
    actor_name = Column(String, nullable=False)
    category = Column(String, nullable=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    s3_textgrid_url = Column(Text, nullable=True)
    s3_pitch_url = Column(Text, nullable=True)
    s3_bgvoice_url = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)  # 썸네일 URL 추가
    youtube_url = Column(
        Text,
        ForeignKey("urls.youtube_url", ondelete="CASCADE"),  # 핵심!
        nullable=False,
        index=True,
    )
    view_count = Column(Integer, nullable=False, default=0, index=True)   # ← 추가

    # 관계
    url = relationship("URL", back_populates="tokens")
    scripts = relationship("Script",
                        back_populates="token",
                        cascade="all, delete",
                        passive_deletes=True,
                        order_by="Script.id"
                        )           


    token_actors = relationship(
        "TokenActor",
        back_populates="token",
        cascade="all, delete",     # Token 삭제→ TokenActor 삭제
        passive_deletes=True
    )

    bookmarked_by = relationship(
        "Bookmark",
        back_populates="token",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    
    analysis_results = relationship(
        "AnalysisResult", 
        back_populates="token", 
        cascade="all, delete",
        passive_deletes=True
    )

    dubbing_results = relationship(
        "DubbingResult",
        back_populates="token",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class URL(Base):
    __tablename__ = "urls"
    youtube_url = Column(Text, primary_key=True)   
    actor_id    = Column(Integer,
                         ForeignKey("actors.id", ondelete="CASCADE"),
                         nullable=False, index=True)


    actor  = relationship("Actor", back_populates="urls")
    tokens = relationship("Token",
                          back_populates="url",
                          cascade="all, delete",
                          passive_deletes=True)


class Actor(Base):
    __tablename__ = "actors"

    # autoicrement = True 설정해야하나???
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    
    # 관계
    urls = relationship("URL",
                        back_populates="actor",
                        cascade="all, delete",
                        passive_deletes=True)

    token_actors = relationship(
        "TokenActor",
        back_populates="actor",
        cascade="all, delete",     # Actor 삭제→ TokenActor 삭제
        passive_deletes=True
    )

        # ★ 별칭 목록
    aliases = relationship(
        "ActorAlias",
        back_populates="actor",
        cascade="all, delete-orphan",   # 배우 삭제 시 별칭도 삭제
        passive_deletes=True
    )


class ActorAlias(Base):
    __tablename__ = "actor_aliases"
    
    id       = Column(Integer, primary_key=True)
    actor_id = Column(Integer, ForeignKey("actors.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    name     = Column(String, nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("actor_id", "name", name="uq_actor_alias"),
    )

    actor = relationship("Actor", back_populates="aliases")



class Script(Base):
    __tablename__ = "scripts"  # 문장 단위

    id = Column(Integer, primary_key=True, index=True)  # 문장 고유 ID
    token_id = Column(
        Integer,
        ForeignKey("tokens.id", ondelete="CASCADE"),   # ✅ DB-레벨 연쇄 삭제
        nullable=False,
        index=True,
    )    
    
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    script = Column(Text, nullable=False)
    translation = Column(Text, nullable=True)

    token = relationship(
        "Token",
        back_populates="scripts",
        passive_deletes=True,                         # ✅ DB에 맡긴다
    )    
    words = relationship(
        "ScriptWord",
        back_populates="script",
        cascade="all, delete",
        passive_deletes=True,
        order_by="ScriptWord.id"
    )


class ScriptWord(Base):
    __tablename__ = "words"

    id = Column(Integer, primary_key=True, index=True)
    script_id = Column(
        Integer,
        ForeignKey("scripts.id", ondelete="CASCADE"),  # ← 핵심
        nullable=False,
        index=True,
    )
    word = Column(String)
    start_time = Column(Float)
    end_time = Column(Float)
    probability = Column(Float)
    mfcc        = Column(JSON, nullable=True)   # ★ 추가

    # 관계 설정
    script = relationship(
        "Script",
        back_populates="words",
        passive_deletes=True,           # DB가 직접 삭제하도록
    )


class TokenActor(Base):
    __tablename__ = "token_actors"
    id       = Column(Integer, primary_key=True, index=True)
    token_id = Column(Integer, ForeignKey("tokens.id", ondelete="CASCADE"), nullable=False, index=True)
    actor_id = Column(Integer, ForeignKey("actors.id", ondelete="CASCADE"), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("token_id", "actor_id", name="uq_token_actor"),
    )

    token = relationship("Token", back_populates="token_actors", passive_deletes=True)
    actor = relationship("Actor", back_populates="token_actors", passive_deletes=True)



class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    
    # 소셜 로그인 관련 필드
    google_id = Column(String, unique=True, index=True, nullable=True)
    profile_picture = Column(String, nullable=True)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    login_type = Column(String, default="email")  # "email" or "google"
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_admin: bool = Column(Boolean, default=False)
    
    # 관계 설정    
    analysis_results = relationship("AnalysisResult",back_populates="user")
    bookmarks = relationship(
        "Bookmark",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    analysis_results = relationship(
        "AnalysisResult",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    dubbing_results = relationship(
        "DubbingResult",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    
# 사용자 선호도 조사
# 사용자 id - AnlysisResult
# 사용자 id - 북마크


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    token_id = Column(Integer, ForeignKey("tokens.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    
    status = Column(String, nullable=False)
    progress = Column(Integer, nullable=False)
    result = Column(JSON, nullable=True)  # analysis_results 점수만 저장
    message = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    # 관계
    token = relationship("Token", back_populates="analysis_results")
    user = relationship("User", back_populates="analysis_results")
    



class Bookmark(Base):
    """
    북마크 테이블 - id를 primary key로 하고 (user_id, token_id)는 unique constraint
    → 같은 토큰을 중복 북마크하지 못하도록 보장
    """
    __tablename__ = "bookmarks"

    #id = Column(Integer, primary_key=True, index=True)
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer,
                     ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False,
                     index=True)
    token_id = Column(Integer,
                      ForeignKey("tokens.id", ondelete="CASCADE"),
                      nullable=False,
                      index=True)

    created_at = Column(DateTime,
                        server_default=func.now(),
                        nullable=False)

    # 중복 북마크 방지를 위한 유니크 제약조건
    __table_args__ = (
        UniqueConstraint("user_id", "token_id", name="uq_user_token_bookmark"),
    )

    # 양방향 편의를 위한 관계
    user = relationship("User", back_populates="bookmarks", passive_deletes=True)
    token = relationship("Token", back_populates="bookmarked_by", passive_deletes=True)


class YoutubeProcessJob(Base):
    __tablename__ = "youtube_process_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    status = Column(String, nullable=False) # processing, completed, failed
    progress = Column(Integer, nullable=False) # 0-100
    message = Column(String, nullable=True)
    token_id = Column(Integer, ForeignKey("tokens.id", ondelete="SET NULL"), nullable=True, index=True)
    result = Column(JSON, nullable=True) # 전처리 서버에서 받은 최종 결과 (예: token_id, 기타 메타데이터)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DubbingResult(Base):
    __tablename__ = "dubbing_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_id = Column(Integer, ForeignKey("tokens.id", ondelete="CASCADE"), nullable=False, index=True)
    s3_key = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="dubbing_results")
    token = relationship("Token", back_populates="dubbing_results")



class VideoRequest(Base):
    __tablename__ = "video_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    actor = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="심사중")  # "심사중", "승인됨", "거절됨"
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", backref="video_requests")