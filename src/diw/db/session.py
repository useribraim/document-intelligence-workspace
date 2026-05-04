from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_DATABASE_URL = "sqlite:///./diw_local.db"


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None):
    url = database_url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def build_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=build_engine(database_url), autoflush=False, expire_on_commit=False)

