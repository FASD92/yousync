from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from test import get_scripts_by_token, prepare_dub_segments, synthesize_audio_from_segments, get_token_info
from router.utils_s3 import load_main_audio_from_s3

import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)
session = SessionLocal() 

actor_name, video_id, token_start_time = get_token_info(session, token_id=65)
background, original = load_main_audio_from_s3(actor_name, video_id)

scripts = get_scripts_by_token(session, token_id=65)
segments = prepare_dub_segments(user_id=7, token_id=65, scripts=scripts, token_start_time=token_start_time)

synthesize_audio_from_segments(background, original, segments)