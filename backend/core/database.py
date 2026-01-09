"""Database connection management for SQLModel ORM."""

from pathlib import Path
from typing import Generator

from sqlmodel import Session, create_engine, SQLModel

# Default database path
_DB_PATH: Path | None = None
_engine = None


def get_db_path() -> Path:
    """Get the database file path."""
    global _DB_PATH
    if _DB_PATH is None:
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        _DB_PATH = data_dir / "ke_workbench.db"
    return _DB_PATH


def set_db_path(path: Path | str) -> None:
    """Set a custom database path (useful for testing)."""
    global _DB_PATH, _engine
    _DB_PATH = Path(path)
    _engine = None  # Reset engine when path changes


def get_engine():
    """Get SQLAlchemy engine for SQLModel operations."""
    global _engine
    if _engine is None:
        db_path = get_db_path()
        _engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engine


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLModel session for dependency injection."""
    with Session(get_engine()) as session:
        yield session


def init_sqlmodel_tables() -> None:
    """Create SQLModel tables (for services using SQLModel ORM)."""
    SQLModel.metadata.create_all(get_engine())
