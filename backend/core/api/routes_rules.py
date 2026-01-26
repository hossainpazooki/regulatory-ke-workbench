"""Routes for inspecting rules."""

import json

from fastapi import APIRouter, HTTPException, Query

from backend.rules import RuleLoader
from backend.config import get_settings
from backend.storage import RuleVersionRepository, RuleEventRepository
from .models import (
    RulesListResponse,
    RuleInfo,
    RuleDetailResponse,
    RuleVersionResponse,
    RuleVersionListResponse,
    RuleVersionDetailResponse,
    RuleEventResponse,
    RuleEventListResponse,
)

router = APIRouter(prefix="/rules", tags=["Rules"])

# Global instance
_loader: RuleLoader | None = None


def get_loader() -> RuleLoader:
    """Get or create the rule loader instance."""
    global _loader
    if _loader is None:
        settings = get_settings()
        _loader = RuleLoader(settings.rules_dir)
        try:
            _loader.load_directory()
        except FileNotFoundError:
            pass
    return _loader


@router.get("", response_model=RulesListResponse)
async def list_rules(tag: str | None = None) -> RulesListResponse:
    """List all available rules.

    Optionally filter by tag.
    """
    loader = get_loader()

    if tag:
        rules = loader.get_applicable_rules(tags=[tag])
    else:
        rules = loader.get_all_rules()

    rule_infos = []
    for rule in rules:
        # Build source string
        source_str = None
        if rule.source:
            parts = [rule.source.document_id]
            if rule.source.article:
                parts.append(f"Art. {rule.source.article}")
            source_str = " ".join(parts)

        rule_infos.append(
            RuleInfo(
                rule_id=rule.rule_id,
                version=rule.version,
                description=rule.description,
                effective_from=rule.effective_from.isoformat() if rule.effective_from else None,
                effective_to=rule.effective_to.isoformat() if rule.effective_to else None,
                tags=rule.tags,
                source=source_str,
            )
        )

    return RulesListResponse(rules=rule_infos, total=len(rule_infos))


@router.get("/{rule_id}", response_model=RuleDetailResponse)
async def get_rule(rule_id: str) -> RuleDetailResponse:
    """Get detailed information about a specific rule."""
    loader = get_loader()
    rule = loader.get_rule(rule_id)

    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

    # Convert to dict for flexible structure
    applies_if = None
    if rule.applies_if:
        applies_if = rule.applies_if.model_dump(exclude_none=True)

    decision_tree = None
    if rule.decision_tree:
        decision_tree = rule.decision_tree.model_dump(exclude_none=True)

    source = None
    if rule.source:
        source = rule.source.model_dump(exclude_none=True)

    return RuleDetailResponse(
        rule_id=rule.rule_id,
        version=rule.version,
        description=rule.description,
        effective_from=rule.effective_from.isoformat() if rule.effective_from else None,
        effective_to=rule.effective_to.isoformat() if rule.effective_to else None,
        tags=rule.tags,
        source=source,
        applies_if=applies_if,
        decision_tree=decision_tree,
        interpretation_notes=rule.interpretation_notes,
    )


@router.get("/tags/all")
async def list_tags() -> dict:
    """List all unique tags across rules."""
    loader = get_loader()
    rules = loader.get_all_rules()

    tags = set()
    for rule in rules:
        tags.update(rule.tags)

    return {"tags": sorted(tags)}


# =============================================================================
# Version Endpoints
# =============================================================================

# Repository instances
_version_repo: RuleVersionRepository | None = None
_event_repo: RuleEventRepository | None = None


def get_version_repo() -> RuleVersionRepository:
    """Get or create the version repository instance."""
    global _version_repo
    if _version_repo is None:
        _version_repo = RuleVersionRepository()
    return _version_repo


def get_event_repo() -> RuleEventRepository:
    """Get or create the event repository instance."""
    global _event_repo
    if _event_repo is None:
        _event_repo = RuleEventRepository()
    return _event_repo


@router.get("/{rule_id}/versions", response_model=RuleVersionListResponse)
async def list_versions(
    rule_id: str, limit: int = Query(100, ge=1, le=1000)
) -> RuleVersionListResponse:
    """List all versions of a rule."""
    repo = get_version_repo()
    versions = repo.get_version_history(rule_id, limit=limit)

    if not versions:
        raise HTTPException(
            status_code=404, detail=f"No versions found for rule: {rule_id}"
        )

    return RuleVersionListResponse(
        rule_id=rule_id,
        versions=[
            RuleVersionResponse(
                id=v.id,
                rule_id=v.rule_id,
                version=v.version,
                content_hash=v.content_hash,
                effective_from=v.effective_from,
                effective_to=v.effective_to,
                created_at=v.created_at,
                created_by=v.created_by,
                superseded_by=v.superseded_by,
                superseded_at=v.superseded_at,
                jurisdiction_code=v.jurisdiction_code,
                regime_id=v.regime_id,
            )
            for v in versions
        ],
        total=len(versions),
    )


@router.get("/{rule_id}/versions/{version}", response_model=RuleVersionDetailResponse)
async def get_version(rule_id: str, version: int) -> RuleVersionDetailResponse:
    """Get a specific version of a rule."""
    repo = get_version_repo()
    v = repo.get_version(rule_id, version)

    if not v:
        raise HTTPException(
            status_code=404, detail=f"Version {version} not found for rule: {rule_id}"
        )

    return RuleVersionDetailResponse(
        id=v.id,
        rule_id=v.rule_id,
        version=v.version,
        content_hash=v.content_hash,
        content_yaml=v.content_yaml,
        content_json=v.content_json,
        effective_from=v.effective_from,
        effective_to=v.effective_to,
        created_at=v.created_at,
        created_by=v.created_by,
        superseded_by=v.superseded_by,
        superseded_at=v.superseded_at,
        jurisdiction_code=v.jurisdiction_code,
        regime_id=v.regime_id,
    )


@router.get("/{rule_id}/at-timestamp", response_model=RuleVersionDetailResponse)
async def get_version_at_timestamp(
    rule_id: str, timestamp: str = Query(..., description="ISO 8601 timestamp")
) -> RuleVersionDetailResponse:
    """Get the version of a rule effective at a specific timestamp."""
    repo = get_version_repo()
    v = repo.get_version_at_timestamp(rule_id, timestamp)

    if not v:
        raise HTTPException(
            status_code=404,
            detail=f"No version found for rule {rule_id} at timestamp {timestamp}",
        )

    return RuleVersionDetailResponse(
        id=v.id,
        rule_id=v.rule_id,
        version=v.version,
        content_hash=v.content_hash,
        content_yaml=v.content_yaml,
        content_json=v.content_json,
        effective_from=v.effective_from,
        effective_to=v.effective_to,
        created_at=v.created_at,
        created_by=v.created_by,
        superseded_by=v.superseded_by,
        superseded_at=v.superseded_at,
        jurisdiction_code=v.jurisdiction_code,
        regime_id=v.regime_id,
    )


@router.get("/{rule_id}/events", response_model=RuleEventListResponse)
async def list_events(
    rule_id: str, limit: int = Query(100, ge=1, le=1000)
) -> RuleEventListResponse:
    """List all events for a rule."""
    repo = get_event_repo()
    events = repo.get_events_for_rule(rule_id)

    # Apply limit
    events = events[:limit]

    return RuleEventListResponse(
        rule_id=rule_id,
        events=[
            RuleEventResponse(
                id=e.id,
                sequence_number=e.sequence_number,
                rule_id=e.rule_id,
                version=e.version,
                event_type=e.event_type,
                event_data=json.loads(e.event_data) if e.event_data else {},
                timestamp=e.timestamp,
                actor=e.actor,
                reason=e.reason,
            )
            for e in events
        ],
        total=len(events),
    )
