"""
Jurisdiction resolver for cross-border compliance.

Resolves applicable jurisdictions and regimes based on issuer location
and target markets.
"""

from __future__ import annotations

from backend.core.ontology.jurisdiction import (
    JurisdictionCode,
    ApplicableJurisdiction,
    JurisdictionRole,
    EquivalenceDetermination,
    EquivalenceStatus,
)
from backend.database_service.app.services.database import get_db


# Default regime mappings
DEFAULT_REGIMES: dict[str, str] = {
    "EU": "mica_2023",
    "UK": "fca_crypto_2024",
    "US": "securities_act_1933",
    "CH": "finsa_dlt_2021",
    "SG": "psa_2019",
}


def resolve_jurisdictions(
    issuer: str,
    targets: list[str],
    instrument_type: str | None = None,
) -> list[ApplicableJurisdiction]:
    """
    Resolve applicable jurisdictions for a cross-border scenario.

    Determines which jurisdictions apply and their roles (issuer_home, target).

    Args:
        issuer: Issuer jurisdiction code
        targets: List of target jurisdiction codes
        instrument_type: Optional instrument type for regime selection

    Returns:
        List of ApplicableJurisdiction with roles
    """
    applicable = []

    # Add issuer jurisdiction as home
    issuer_code = issuer if isinstance(issuer, str) else issuer.value
    applicable.append(
        ApplicableJurisdiction(
            jurisdiction=JurisdictionCode(issuer_code),
            regime_id=_get_regime_for_jurisdiction(issuer_code, instrument_type),
            role=JurisdictionRole.ISSUER_HOME,
        )
    )

    # Add target jurisdictions
    for target in targets:
        target_code = target if isinstance(target, str) else target.value
        if target_code != issuer_code:  # Don't duplicate issuer
            applicable.append(
                ApplicableJurisdiction(
                    jurisdiction=JurisdictionCode(target_code),
                    regime_id=_get_regime_for_jurisdiction(target_code, instrument_type),
                    role=JurisdictionRole.TARGET,
                )
            )

    return applicable


def _get_regime_for_jurisdiction(
    jurisdiction_code: str,
    instrument_type: str | None = None,
) -> str:
    """Get the default regulatory regime for a jurisdiction.

    May vary by instrument type in future versions.
    """
    return DEFAULT_REGIMES.get(jurisdiction_code, "unknown")


def get_equivalences(
    from_jurisdiction: str,
    to_jurisdictions: list[str],
) -> list[dict]:
    """
    Get equivalence determinations between jurisdictions.

    Queries the database for known equivalence decisions that may
    reduce compliance requirements.

    Args:
        from_jurisdiction: Source jurisdiction code
        to_jurisdictions: Target jurisdiction codes

    Returns:
        List of equivalence determination dicts
    """
    if not to_jurisdictions:
        return []

    equivalences = []

    with get_db() as conn:
        # Query for equivalences from issuer to targets
        placeholders = ",".join("?" * len(to_jurisdictions))
        cursor = conn.execute(
            f"""
            SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                   effective_date, expiry_date, source_reference, notes
            FROM equivalence_determinations
            WHERE from_jurisdiction = ?
              AND to_jurisdiction IN ({placeholders})
            """,
            [from_jurisdiction] + to_jurisdictions,
        )

        for row in cursor.fetchall():
            equivalences.append({
                "id": row["id"],
                "from": row["from_jurisdiction"],
                "to": row["to_jurisdiction"],
                "scope": row["scope"],
                "status": row["status"],
                "effective_date": row["effective_date"],
                "expiry_date": row["expiry_date"],
                "source_reference": row["source_reference"],
                "notes": row["notes"],
            })

        # Also check reverse direction
        cursor = conn.execute(
            f"""
            SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                   effective_date, expiry_date, source_reference, notes
            FROM equivalence_determinations
            WHERE to_jurisdiction = ?
              AND from_jurisdiction IN ({placeholders})
            """,
            [from_jurisdiction] + to_jurisdictions,
        )

        for row in cursor.fetchall():
            equivalences.append({
                "id": row["id"],
                "from": row["from_jurisdiction"],
                "to": row["to_jurisdiction"],
                "scope": row["scope"],
                "status": row["status"],
                "effective_date": row["effective_date"],
                "expiry_date": row["expiry_date"],
                "source_reference": row["source_reference"],
                "notes": row["notes"],
            })

    return equivalences


def get_jurisdiction_info(code: str) -> dict | None:
    """Get jurisdiction information from database.

    Args:
        code: Jurisdiction code

    Returns:
        Jurisdiction info dict or None if not found
    """
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT code, name, authority, parent_code
            FROM jurisdictions
            WHERE code = ?
            """,
            (code,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "code": row["code"],
                "name": row["name"],
                "authority": row["authority"],
                "parent_code": row["parent_code"],
            }
    return None


def get_regime_info(regime_id: str) -> dict | None:
    """Get regulatory regime information from database.

    Args:
        regime_id: Regime identifier

    Returns:
        Regime info dict or None if not found
    """
    with get_db() as conn:
        cursor = conn.execute(
            """
            SELECT id, jurisdiction_code, name, effective_date, sunset_date, source_url
            FROM regulatory_regimes
            WHERE id = ?
            """,
            (regime_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "jurisdiction_code": row["jurisdiction_code"],
                "name": row["name"],
                "effective_date": row["effective_date"],
                "sunset_date": row["sunset_date"],
                "source_url": row["source_url"],
            }
    return None
