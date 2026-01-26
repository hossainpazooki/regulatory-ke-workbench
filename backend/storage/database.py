"""
Database connection management and initialization.

Supports both SQLite (local dev) and PostgreSQL (production) via DATABASE_URL.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection


# Global engine instance
_engine: Engine | None = None
_DB_PATH: Path | None = None


def _is_postgres() -> bool:
    """Check if using PostgreSQL database."""
    database_url = os.getenv("DATABASE_URL", "")
    return database_url.startswith("postgres")


def get_database_url() -> str:
    """Get database URL from environment or default to SQLite.

    Handles Railway's postgres:// URL format by converting to postgresql://.
    """
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        # Railway uses postgres:// but SQLAlchemy requires postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    # Default to SQLite for local development
    db_path = get_db_path()
    return f"sqlite:///{db_path}"


def get_db_path() -> Path:
    """Get the SQLite database file path (used when DATABASE_URL not set)."""
    global _DB_PATH
    if _DB_PATH is None:
        # Default to data/ directory in project root
        # storage/database.py -> backend/ -> project_root/
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


def get_engine() -> Engine:
    """Get SQLAlchemy engine for database operations."""
    global _engine
    if _engine is None:
        database_url = get_database_url()

        # SQLite-specific connection args
        connect_args = {}
        if database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engine = create_engine(database_url, echo=False, connect_args=connect_args)
    return _engine


def reset_engine() -> None:
    """Reset the engine (useful for testing or reconfiguration)."""
    global _engine
    _engine = None


@contextmanager
def get_db() -> Generator[Connection, None, None]:
    """Get a database connection.

    Usage:
        with get_db() as conn:
            result = conn.execute(text("SELECT * FROM rules"))
            rows = result.fetchall()
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Enable foreign keys for SQLite
        if not _is_postgres():
            conn.execute(text("PRAGMA foreign_keys = ON"))
        yield conn


# SQLModel utilities available via explicit import:
#   from backend.core.database import get_engine, get_session, init_sqlmodel_tables


def init_db() -> None:
    """Initialize database schema.

    Creates all tables if they don't exist. Safe to call multiple times.
    """
    engine = get_engine()

    # For SQLite, use executescript equivalent; for PostgreSQL, execute statements individually
    if not _is_postgres():
        # SQLite: Use raw connection's executescript for multi-statement execution
        raw_conn = engine.raw_connection()
        try:
            raw_conn.executescript(_SCHEMA)
            raw_conn.commit()
        finally:
            raw_conn.close()
    else:
        # PostgreSQL: Execute each statement individually
        with engine.connect() as conn:
            # Split by semicolons, but handle multi-line statements properly
            statements = []
            current_stmt = []
            for line in _SCHEMA.split("\n"):
                stripped = line.strip()
                # Skip pure comment lines
                if stripped.startswith("--"):
                    continue
                current_stmt.append(line)
                if stripped.endswith(";"):
                    statements.append("\n".join(current_stmt))
                    current_stmt = []

            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith("--"):
                    # Remove trailing semicolon for PostgreSQL
                    if statement.endswith(";"):
                        statement = statement[:-1]
                    if statement.strip():
                        conn.execute(text(statement))
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
    ir_version INTEGER DEFAULT 2,       -- For IR schema migrations (v2 = jurisdiction support)
    compiled_at TEXT,                   -- ISO timestamp when IR was generated

    -- Source reference
    source_document_id TEXT,            -- e.g., "mica_2023"
    source_article TEXT,                -- e.g., "36(1)"

    -- Jurisdiction scoping (v4 multi-jurisdiction support)
    jurisdiction_code VARCHAR(10) DEFAULT 'EU',  -- Primary jurisdiction
    regime_id VARCHAR(100) DEFAULT 'mica_2023',  -- Regulatory regime
    cross_border_relevant INTEGER DEFAULT 0,     -- Boolean: applies cross-border

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    updated_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),

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

    -- Jurisdiction support (v4)
    jurisdiction_code VARCHAR(10) DEFAULT 'EU',  -- For jurisdiction-filtered lookup
    regime_id VARCHAR(100),                      -- Regulatory regime

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
    verified_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
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
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),

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
    created_at TEXT NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    metadata TEXT,                      -- JSON for additional context

    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

-- =============================================================================
-- JURISDICTION REGISTRY (v4 multi-jurisdiction support)
-- =============================================================================

CREATE TABLE IF NOT EXISTS jurisdictions (
    code VARCHAR(10) PRIMARY KEY,        -- EU, US, UK, SG, CH
    name VARCHAR(255) NOT NULL,
    authority VARCHAR(255),              -- ESMA, SEC, FCA, MAS, FINMA
    parent_code VARCHAR(10),             -- For sub-jurisdictions (EU -> DE, FR)
    metadata JSON,
    FOREIGN KEY (parent_code) REFERENCES jurisdictions(code)
);

CREATE TABLE IF NOT EXISTS regulatory_regimes (
    id VARCHAR(100) PRIMARY KEY,         -- mica_2023, fca_crypto_2024
    jurisdiction_code VARCHAR(10) NOT NULL,
    name VARCHAR(255) NOT NULL,
    effective_date DATE,
    sunset_date DATE,                    -- NULL if still in force
    source_url TEXT,
    metadata JSON,
    FOREIGN KEY (jurisdiction_code) REFERENCES jurisdictions(code)
);

-- =============================================================================
-- EQUIVALENCE DETERMINATIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS equivalence_determinations (
    id TEXT PRIMARY KEY,
    from_jurisdiction VARCHAR(10) NOT NULL,
    to_jurisdiction VARCHAR(10) NOT NULL,
    scope VARCHAR(100) NOT NULL,         -- prospectus, authorization, custody
    status VARCHAR(50) NOT NULL,         -- equivalent, partial, not_equivalent, pending
    effective_date DATE,
    expiry_date DATE,
    source_reference TEXT,
    notes TEXT,
    metadata JSON,
    FOREIGN KEY (from_jurisdiction) REFERENCES jurisdictions(code),
    FOREIGN KEY (to_jurisdiction) REFERENCES jurisdictions(code)
);

-- =============================================================================
-- CROSS-JURISDICTION CONFLICTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS rule_conflicts (
    id TEXT PRIMARY KEY,
    rule_id_a VARCHAR(255) NOT NULL,
    rule_id_b VARCHAR(255) NOT NULL,
    conflict_type VARCHAR(100) NOT NULL, -- classification, obligation, timeline
    severity VARCHAR(50) NOT NULL,       -- blocking, warning, info
    description TEXT,
    resolution_strategy VARCHAR(100),    -- cumulative, stricter, home_jurisdiction
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- TEMPORAL VERSIONING (Temporal.io-inspired)
-- =============================================================================

-- Immutable rule version snapshots
CREATE TABLE IF NOT EXISTS rule_versions (
    id TEXT PRIMARY KEY,
    rule_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_yaml TEXT NOT NULL,
    content_json TEXT,
    content_hash TEXT NOT NULL,
    effective_from DATE,
    effective_to DATE,
    created_at TEXT NOT NULL,
    created_by TEXT,
    superseded_by INTEGER,
    superseded_at TEXT,
    jurisdiction_code VARCHAR(10),
    regime_id VARCHAR(100),
    UNIQUE(rule_id, version)
);

-- Event sourcing log for rule changes
CREATE TABLE IF NOT EXISTS rule_events (
    id TEXT PRIMARY KEY,
    sequence_number INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    event_type TEXT NOT NULL,  -- RuleCreated, RuleUpdated, RuleDeprecated
    event_data TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    actor TEXT,
    reason TEXT
);

-- =============================================================================
-- JURISDICTION CONFIG (replaces hardcoded dictionaries)
-- =============================================================================

-- Step timelines for compliance pathways
CREATE TABLE IF NOT EXISTS step_timelines (
    step_id TEXT NOT NULL,
    jurisdiction_code VARCHAR(10) DEFAULT '*',
    min_days INTEGER NOT NULL,
    max_days INTEGER NOT NULL,
    description TEXT,
    PRIMARY KEY (step_id, jurisdiction_code)
);

-- Step dependencies for compliance pathways
CREATE TABLE IF NOT EXISTS step_dependencies (
    step_id TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    PRIMARY KEY (step_id, depends_on)
);

-- Obligation conflict pairs
CREATE TABLE IF NOT EXISTS obligation_conflicts (
    obligation_a TEXT NOT NULL,
    obligation_b TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    resolution_hint TEXT,
    PRIMARY KEY (obligation_a, obligation_b)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_rules_active ON rules(rule_id) WHERE is_active = 1;
CREATE INDEX IF NOT EXISTS idx_rules_document ON rules(source_document_id);
CREATE INDEX IF NOT EXISTS idx_rules_compiled ON rules(rule_id) WHERE rule_ir IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_rules_jurisdiction ON rules(jurisdiction_code);
CREATE INDEX IF NOT EXISTS idx_rules_regime ON rules(regime_id);

CREATE INDEX IF NOT EXISTS idx_premise_lookup ON rule_premise_index(premise_key);
CREATE INDEX IF NOT EXISTS idx_premise_rule ON rule_premise_index(rule_id, rule_version);
CREATE INDEX IF NOT EXISTS idx_premise_jurisdiction ON rule_premise_index(premise_key, jurisdiction_code);
CREATE INDEX IF NOT EXISTS idx_premise_regime ON rule_premise_index(premise_key, regime_id);

CREATE INDEX IF NOT EXISTS idx_verification_rule ON verification_results(rule_id);
CREATE INDEX IF NOT EXISTS idx_verification_status ON verification_results(status, verified_at);

CREATE INDEX IF NOT EXISTS idx_evidence_verification ON verification_evidence(verification_id);
CREATE INDEX IF NOT EXISTS idx_evidence_tier ON verification_evidence(tier, label);

CREATE INDEX IF NOT EXISTS idx_reviews_rule ON reviews(rule_id);
CREATE INDEX IF NOT EXISTS idx_reviews_reviewer ON reviews(reviewer_id);

CREATE INDEX IF NOT EXISTS idx_jurisdictions_parent ON jurisdictions(parent_code);
CREATE INDEX IF NOT EXISTS idx_regimes_jurisdiction ON regulatory_regimes(jurisdiction_code);
CREATE INDEX IF NOT EXISTS idx_equivalence_from ON equivalence_determinations(from_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_equivalence_to ON equivalence_determinations(to_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_equivalence_scope ON equivalence_determinations(scope, status);
CREATE INDEX IF NOT EXISTS idx_conflicts_rules ON rule_conflicts(rule_id_a, rule_id_b);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON rule_conflicts(conflict_type, severity);

CREATE INDEX IF NOT EXISTS idx_rule_versions_rule ON rule_versions(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_versions_effective ON rule_versions(rule_id, effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_rule_versions_hash ON rule_versions(content_hash);

CREATE INDEX IF NOT EXISTS idx_rule_events_rule ON rule_events(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_events_sequence ON rule_events(sequence_number);
CREATE INDEX IF NOT EXISTS idx_rule_events_type ON rule_events(event_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_step_timelines_step ON step_timelines(step_id);
CREATE INDEX IF NOT EXISTS idx_step_dependencies_step ON step_dependencies(step_id);
CREATE INDEX IF NOT EXISTS idx_obligation_conflicts_a ON obligation_conflicts(obligation_a);
CREATE INDEX IF NOT EXISTS idx_obligation_conflicts_b ON obligation_conflicts(obligation_b);
"""


def reset_db() -> None:
    """Drop all tables and recreate schema. USE WITH CAUTION."""
    with get_db() as conn:
        # Get all table names (database-agnostic)
        if _is_postgres():
            result = conn.execute(text(
                "SELECT table_name as name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
        else:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ))
        tables = [row[0] for row in result.fetchall()]

        # Drop all tables (CASCADE for PostgreSQL foreign keys)
        for table in tables:
            if _is_postgres():
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            else:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))

        conn.commit()

    # Recreate schema
    init_db()


def get_table_stats() -> dict[str, int]:
    """Get row counts for all tables (useful for diagnostics)."""
    with get_db() as conn:
        stats = {}
        # Get all table names (database-agnostic)
        if _is_postgres():
            result = conn.execute(text(
                "SELECT table_name as name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
        else:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ))
        tables = [row[0] for row in result.fetchall()]

        for table in tables:
            result = conn.execute(text(f"SELECT COUNT(*) as count FROM {table}"))
            stats[table] = result.fetchone()[0]

        return stats


def seed_jurisdictions() -> None:
    """Seed jurisdiction registry with supported jurisdictions and regimes."""
    with get_db() as conn:
        # Seed jurisdictions
        jurisdictions = [
            ("EU", "European Union", "ESMA"),
            ("UK", "United Kingdom", "FCA"),
            ("US", "United States", "SEC"),
            ("SG", "Singapore", "MAS"),
            ("CH", "Switzerland", "FINMA"),
        ]
        for code, name, authority in jurisdictions:
            conn.execute(
                text("""
                INSERT INTO jurisdictions (code, name, authority)
                VALUES (:code, :name, :authority)
                ON CONFLICT (code) DO NOTHING
                """),
                {"code": code, "name": name, "authority": authority},
            )

        # Seed regulatory regimes
        regimes = [
            ("mica_2023", "EU", "Markets in Crypto-Assets Regulation", "2024-12-30"),
            ("mifid2_2014", "EU", "MiFID II / MiFIR", "2018-01-03"),
            ("dlt_pilot_2022", "EU", "DLT Pilot Regime", "2023-03-23"),
            ("fca_crypto_2024", "UK", "FCA Cryptoasset Regime", "2024-01-08"),
            ("finsa_dlt_2021", "CH", "FinSA / DLT Act", "2021-08-01"),
            ("psa_2019", "SG", "Payment Services Act", "2020-01-28"),
            ("securities_act_1933", "US", "Securities Act of 1933", "1933-05-27"),
            ("genius_act_2025", "US", "GENIUS Act", "2025-01-01"),
        ]
        for regime_id, jurisdiction_code, name, effective_date in regimes:
            conn.execute(
                text("""
                INSERT INTO regulatory_regimes (id, jurisdiction_code, name, effective_date)
                VALUES (:id, :jurisdiction_code, :name, :effective_date)
                ON CONFLICT (id) DO NOTHING
                """),
                {"id": regime_id, "jurisdiction_code": jurisdiction_code,
                 "name": name, "effective_date": effective_date},
            )

        # Seed known equivalence determinations
        equivalences = [
            ("ch_eu_prospectus", "CH", "EU", "prospectus", "partial",
             "Swiss prospectus partially recognized under MiFID II"),
            ("uk_eu_post_brexit", "UK", "EU", "authorization", "not_equivalent",
             "Post-Brexit, UK authorization not recognized in EU"),
        ]
        for eq_id, from_j, to_j, scope, status, notes in equivalences:
            conn.execute(
                text("""
                INSERT INTO equivalence_determinations
                (id, from_jurisdiction, to_jurisdiction, scope, status, notes)
                VALUES (:id, :from_j, :to_j, :scope, :status, :notes)
                ON CONFLICT (id) DO NOTHING
                """),
                {"id": eq_id, "from_j": from_j, "to_j": to_j,
                 "scope": scope, "status": status, "notes": notes},
            )

        conn.commit()


def init_db_with_seed() -> None:
    """Initialize database and seed with jurisdiction data."""
    init_db()
    seed_jurisdictions()
