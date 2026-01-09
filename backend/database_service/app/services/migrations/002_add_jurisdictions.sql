-- =============================================================================
-- Migration 002: Add Jurisdiction Support (v4 Architecture)
-- =============================================================================
-- Adds multi-jurisdiction support for cross-border compliance navigation.

-- =============================================================================
-- JURISDICTION REGISTRY
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
-- EXTENDED RULES TABLE
-- =============================================================================

-- Add jurisdiction columns to rules table
-- Using ALTER TABLE for incremental migration
-- Note: SQLite doesn't support ADD COLUMN IF NOT EXISTS, so we check first

-- These columns will be added if they don't exist (handled by Python migration runner)

-- =============================================================================
-- EXTENDED PREMISE INDEX
-- =============================================================================

-- Add jurisdiction columns to premise index
-- Note: These will be added via ALTER TABLE if not present

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
-- INDEXES FOR O(1) LOOKUP
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_jurisdictions_parent ON jurisdictions(parent_code);
CREATE INDEX IF NOT EXISTS idx_regimes_jurisdiction ON regulatory_regimes(jurisdiction_code);
CREATE INDEX IF NOT EXISTS idx_equivalence_from ON equivalence_determinations(from_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_equivalence_to ON equivalence_determinations(to_jurisdiction);
CREATE INDEX IF NOT EXISTS idx_equivalence_scope ON equivalence_determinations(scope, status);
CREATE INDEX IF NOT EXISTS idx_conflicts_rules ON rule_conflicts(rule_id_a, rule_id_b);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON rule_conflicts(conflict_type, severity);

-- =============================================================================
-- SEED DATA
-- =============================================================================

INSERT OR IGNORE INTO jurisdictions (code, name, authority) VALUES
    ('EU', 'European Union', 'ESMA'),
    ('UK', 'United Kingdom', 'FCA'),
    ('US', 'United States', 'SEC'),
    ('SG', 'Singapore', 'MAS'),
    ('CH', 'Switzerland', 'FINMA');

INSERT OR IGNORE INTO regulatory_regimes (id, jurisdiction_code, name, effective_date) VALUES
    ('mica_2023', 'EU', 'Markets in Crypto-Assets Regulation', '2024-12-30'),
    ('mifid2_2014', 'EU', 'MiFID II / MiFIR', '2018-01-03'),
    ('dlt_pilot_2022', 'EU', 'DLT Pilot Regime', '2023-03-23'),
    ('fca_crypto_2024', 'UK', 'FCA Cryptoasset Regime', '2024-01-08'),
    ('finsa_dlt_2021', 'CH', 'FinSA / DLT Act', '2021-08-01'),
    ('psa_2019', 'SG', 'Payment Services Act', '2020-01-28'),
    ('securities_act_1933', 'US', 'Securities Act of 1933', '1933-05-27'),
    ('genius_act_2025', 'US', 'GENIUS Act', '2025-01-01');

-- Seed known equivalence determinations
INSERT OR IGNORE INTO equivalence_determinations (id, from_jurisdiction, to_jurisdiction, scope, status, notes) VALUES
    ('ch_eu_prospectus', 'CH', 'EU', 'prospectus', 'partial', 'Swiss prospectus partially recognized under MiFID II'),
    ('uk_eu_post_brexit', 'UK', 'EU', 'authorization', 'not_equivalent', 'Post-Brexit, UK authorization not recognized in EU');
