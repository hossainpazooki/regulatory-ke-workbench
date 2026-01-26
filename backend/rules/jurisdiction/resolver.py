"""
Jurisdiction resolver for cross-border compliance.

Resolves applicable jurisdictions and regimes based on issuer location
and target markets.
"""

from __future__ import annotations

from sqlalchemy import text

from backend.core.ontology.jurisdiction import (
    JurisdictionCode,
    ApplicableJurisdiction,
    JurisdictionRole,
    EquivalenceDetermination,
    EquivalenceStatus,
)
from backend.storage.database import get_db


# Default regime mappings
DEFAULT_REGIMES: dict[str, str] = {
    "EU": "mica_2023",
    "UK": "fca_crypto_2024",
    "US": "genius_act_2025",          # Primary tokenization regime
    "US_SEC": "securities_act_1933",   # Securities classification
    "US_CFTC": "cftc_digital_assets_2024",  # Derivatives/commodities
    "CH": "finsa_dlt_2021",
    "SG": "psa_2019",
    "HK": "sfc_vasp_2023",             # Hong Kong VASP regime
    "JP": "psa_japan_2023",            # Japan stablecoin rules
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

    try:
        with get_db() as conn:
            # Build dynamic IN clause with named parameters
            target_params = {f"target_{i}": t for i, t in enumerate(to_jurisdictions)}
            placeholders = ", ".join(f":target_{i}" for i in range(len(to_jurisdictions)))

            # Query for equivalences from issuer to targets
            result = conn.execute(
                text(f"""
                SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                       effective_date, expiry_date, source_reference, notes
                FROM equivalence_determinations
                WHERE from_jurisdiction = :from_j
                  AND to_jurisdiction IN ({placeholders})
                """),
                {"from_j": from_jurisdiction, **target_params},
            )

            for row in result.fetchall():
                equivalences.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "scope": row[3],
                    "status": row[4],
                    "effective_date": row[5],
                    "expiry_date": row[6],
                    "source_reference": row[7],
                    "notes": row[8],
                })

            # Also check reverse direction
            result = conn.execute(
                text(f"""
                SELECT id, from_jurisdiction, to_jurisdiction, scope, status,
                       effective_date, expiry_date, source_reference, notes
                FROM equivalence_determinations
                WHERE to_jurisdiction = :from_j
                  AND from_jurisdiction IN ({placeholders})
                """),
                {"from_j": from_jurisdiction, **target_params},
            )

            for row in result.fetchall():
                equivalences.append({
                    "id": row[0],
                    "from": row[1],
                    "to": row[2],
                    "scope": row[3],
                    "status": row[4],
                    "effective_date": row[5],
                    "expiry_date": row[6],
                    "source_reference": row[7],
                    "notes": row[8],
                })
    except Exception:
        # Table may not exist in production database
        # Return empty list gracefully
        pass

    return equivalences


def get_jurisdiction_info(code: str) -> dict | None:
    """Get jurisdiction information from database.

    Args:
        code: Jurisdiction code

    Returns:
        Jurisdiction info dict or None if not found
    """
    try:
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT code, name, authority, parent_code
                FROM jurisdictions
                WHERE code = :code
                """),
                {"code": code}
            )
            row = result.fetchone()
            if row:
                return {
                    "code": row[0],
                    "name": row[1],
                    "authority": row[2],
                    "parent_code": row[3],
                }
    except Exception:
        # Table may not exist in production database
        pass
    return None


def get_regime_info(regime_id: str) -> dict | None:
    """Get regulatory regime information from database.

    Args:
        regime_id: Regime identifier

    Returns:
        Regime info dict or None if not found
    """
    try:
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT id, jurisdiction_code, name, effective_date, sunset_date, source_url
                FROM regulatory_regimes
                WHERE id = :regime_id
                """),
                {"regime_id": regime_id}
            )
            row = result.fetchone()
            if row:
                return {
                    "id": row[0],
                    "jurisdiction_code": row[1],
                    "name": row[2],
                    "effective_date": row[3],
                    "sunset_date": row[4],
                    "source_url": row[5],
                }
    except Exception:
        # Table may not exist in production database
        pass
    return None
