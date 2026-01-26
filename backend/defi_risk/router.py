"""DeFi risk scoring API endpoints."""

from fastapi import APIRouter, HTTPException

from . import service
from .schemas import (
    DeFiCategory,
    DeFiRiskScore,
    DeFiScoreRequest,
    SmartContractRisk,
    EconomicRisk,
    OracleRisk,
    GovernanceRisk,
)
from .constants import DEFI_PROTOCOL_DEFAULTS

router = APIRouter(prefix="/defi-risk", tags=["defi-risk"])


@router.post("/score", response_model=DeFiRiskScore)
async def score_protocol(request: DeFiScoreRequest) -> DeFiRiskScore:
    """
    Score a DeFi protocol across risk dimensions.

    Returns letter grades (A-F) for smart contract, economic, oracle,
    and governance risks, along with detailed risk factors and
    regulatory flags.
    """
    return service.score_defi_protocol(
        protocol_id=request.protocol_id,
        category=request.category,
        smart_contract=request.smart_contract,
        economic=request.economic,
        oracle=request.oracle,
        governance=request.governance,
    )


@router.get("/protocols")
async def list_protocol_defaults() -> dict:
    """List available protocol default configurations."""
    return {"protocols": service.list_protocol_defaults()}


@router.get("/protocols/{protocol_id}")
async def get_protocol_config(protocol_id: str) -> dict:
    """Get default configuration for a known protocol."""
    config = service.get_protocol_defaults(protocol_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Protocol '{protocol_id}' not found. Available: {service.list_protocol_defaults()}"
        )
    return {"protocol_id": protocol_id, **config}


@router.post("/protocols/{protocol_id}/score", response_model=DeFiRiskScore)
async def score_known_protocol(protocol_id: str) -> DeFiRiskScore:
    """
    Score a known protocol using its default configuration.

    This endpoint provides a quick way to get risk scores for
    well-known protocols like Aave, Uniswap, Lido, and GMX.
    """
    config = service.get_protocol_defaults(protocol_id)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=f"Protocol '{protocol_id}' not found. Available: {service.list_protocol_defaults()}"
        )

    return service.score_defi_protocol(
        protocol_id=protocol_id,
        category=config["category"],
        smart_contract=SmartContractRisk(**config.get("smart_contract", {})),
        economic=EconomicRisk(**config.get("economic", {})),
        oracle=OracleRisk(**config.get("oracle", {})),
        governance=GovernanceRisk(**config.get("governance", {})),
    )


@router.get("/categories")
async def list_categories() -> dict:
    """List DeFi protocol categories."""
    return {"categories": [c.value for c in DeFiCategory]}
