"""
Database connection management and initialization.

Uses SQLite for development with PostgreSQL-compatible schema design.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# Default database path
_DB_PATH: Path | None = None


def get_db_path() -> Path:
    """Get the database file path."""
    global _DB_PATH
    if _DB_PATH is None:
        # Default to data/ directory in project root
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        _DB_PATH = data_dir / "ke_workbench.db"
    return _DB_PATH


def set_db_path(path: Path | str) -> None:
    """Set a custom database path (useful for testing)."""
    global _DB_PATH
    _DB_PATH = Path(path)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with row factory enabled.

    Usage:
        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM rules")
            rows = cursor.fetchall()
    """
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database schema.

    Creates all tables if they don't exist. Safe to call multiple times.
    """
    with get_db() as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


# =============================================================================
# Database Schema (SQLite, PostgreSQL-compatible design)
# =============================================================================

_SCHEMA = """
-- =============================================================================
-- CORE RULE STORAGE
-- =============================================================================
-- Stores both the original YAML/JSON content and the compiled IR

CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,                -- UUID
    rule_id TEXT UNIQUE NOT NULL,       -- Human-readable ID (e.g., "mica_art36_public_offer")
    version INTEGER NOT NULL DEFAULT 1,

    -- Source content
    content_yaml TEXT NOT NULL,         -- Original YAML source
    content_json TEXT,                  -- Parsed JSON (for queries)

    -- Compiled IR (Phase 2)
    rule_ir TEXT,                       -- Compiled intermediate representation (JSON)
    ir_version INTEGER DEFAULT 1,       -- For IR schema migrations
    compiled_at TEXT,                   -- ISO timestamp when IR was generated

    -- Source reference
    source_document_id TEXT,            -- e.g., "mica_2023"
    source_article TEXT,                -- e.g., "36(1)"

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),

    -- Status
    is_active INTEGER NOT NULL DEFAULT 1
);

-- =============================================================================
-- PREMISE INDEX (for O(1) rule lookup)
-- =============================================================================
-- Maps premise keys (field:value pairs) to rules for fast applicability lookup

CREATE TABLE IF NOT EXISTS rule_premise_index (
    premise_key TEXT NOT NULL,          -- e.g., "instrument_type:art"
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL DEFAULT 1,
    premise_position INTEGER,           -- Position in rule's condition list
    selectivity REAL DEFAULT 0.5,       -- Estimated fraction of facts matching

    PRIMARY KEY (premise_key, rule_id, rule_version),
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

-- =============================================================================
-- VERIFICATION RESULTS
-- =============================================================================
-- Stores consistency check results per rule

CREATE TABLE IF NOT EXISTS verification_results (
    id TEXT PRIMARY KEY,                -- UUID
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL DEFAULT 1,

    -- Summary
    status TEXT NOT NULL,               -- verified, needs_review, inconsistent, unverified
    confidence REAL,                    -- 0.0 to 1.0

    -- Audit info
    verified_at TEXT NOT NULL DEFAULT (datetime('now')),
    verified_by TEXT,                   -- 'system' or 'human:username'
    notes TEXT,                         -- Optional reviewer notes

    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

-- =============================================================================
-- VERIFICATION EVIDENCE
-- =============================================================================
-- Individual evidence records from tier checks

CREATE TABLE IF NOT EXISTS verification_evidence (
    id TEXT PRIMARY KEY,                -- UUID
    verification_id TEXT NOT NULL,

    -- Evidence details
    tier INTEGER NOT NULL,              -- 0-4
    category TEXT NOT NULL,             -- e.g., "deontic_alignment"
    label TEXT NOT NULL,                -- pass, fail, warning
    score REAL,                         -- 0.0 to 1.0
    details TEXT,                       -- Human-readable explanation

    -- Source reference
    source_span TEXT,                   -- Relevant text from legal source
    rule_element TEXT,                  -- Path in rule (e.g., "applies_if.all[0]")

    -- Timestamp
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (verification_id) REFERENCES verification_results(id) ON DELETE CASCADE
);

-- =============================================================================
-- HUMAN REVIEWS
-- =============================================================================
-- Audit trail for human review decisions

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,                -- UUID
    rule_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL,             -- consistent, inconsistent, unknown
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT,                      -- JSON for additional context

    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(rule_id) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_rules_document ON rules(source_document_id);
CREATE INDEX IF NOT EXISTS idx_rules_compiled ON rules(rule_id) WHERE rule_ir IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_premise_lookup ON rule_premise_index(premise_key);
CREATE INDEX IF NOT EXISTS idx_premise_rule ON rule_premise_index(rule_id, rule_version);

CREATE INDEX IF NOT EXISTS idx_verification_rule ON verification_results(rule_id);
CREATE INDEX IF NOT EXISTS idx_verification_status ON verification_results(status, verified_at);

CREATE INDEX IF NOT EXISTS idx_evidence_verification ON verification_evidence(verification_id);
CREATE INDEX IF NOT EXISTS idx_evidence_tier ON verification_evidence(tier, label);

CREATE INDEX IF NOT EXISTS idx_reviews_rule ON reviews(rule_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id);
"""


def reset_db() -> None:
    """Drop all tables and recreate schema. USE WITH CAUTION."""
    with get_db() as conn:
        # Get all table names
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        # Drop all tables
        for table in tables:
            conn.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()

    # Recreate schema
    init_db()


def get_table_stats() -> dict[str, int]:
    """Get row counts for all tables (useful for diagnostics)."""
    with get_db() as conn:
        stats = {}
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        for table in tables:
            cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()["count"]

        return stats
