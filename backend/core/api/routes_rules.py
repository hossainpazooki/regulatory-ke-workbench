"""Routes for inspecting rules."""

from fastapi import APIRouter, HTTPException

from backend.rule_service.app.services import RuleLoader
from backend.config import get_settings
from .models import RulesListResponse, RuleInfo, RuleDetailResponse

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
