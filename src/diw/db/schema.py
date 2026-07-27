from __future__ import annotations

from sqlalchemy import Engine, text

from diw.db.models import Base


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        create_pgvector_schema(engine)


def drop_schema(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        drop_pgvector_schema(engine)
    Base.metadata.drop_all(engine)


def create_pgvector_schema(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chunk_embedding_vectors (
                    embedding_id VARCHAR(160) PRIMARY KEY
                        REFERENCES chunk_embeddings(id) ON DELETE CASCADE,
                    chunk_id VARCHAR(80) NOT NULL
                        REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding_model VARCHAR(128) NOT NULL,
                    dimensions INTEGER NOT NULL,
                    content_hash VARCHAR(64) NOT NULL,
                    vector vector NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_chunk_embedding_vectors_chunk_id
                ON chunk_embedding_vectors (chunk_id)
                """
            )
        )


def drop_pgvector_schema(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS chunk_embedding_vectors"))
