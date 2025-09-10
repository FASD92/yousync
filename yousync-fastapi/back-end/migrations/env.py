from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
import os, sys
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# ① 프로젝트 루트 경로 추가
BACKEND_DIR = Path(__file__).resolve().parents[1]   # ← back-end/
sys.path.append(str(BACKEND_DIR))

# ② .env 로드
load_dotenv(BACKEND_DIR / ".env")                   # back-end/.env 읽기
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL not set in .env")

# ---------------------------------------------------------------------
# ③ Alembic 설정 객체
config = context.config
config.set_main_option("sqlalchemy.url", database_url)   # ★ URL 주입

# 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ④ 모델 메타데이터
from models import Base
target_metadata = Base.metadata

# ---------------------------------------------------------------------
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        {"sqlalchemy.url": database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection,
                          target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
