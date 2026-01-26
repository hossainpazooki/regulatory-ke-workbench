"""DeFi risk scoring domain."""

from .router import router
from .service import score_defi_protocol, get_protocol_defaults, list_protocol_defaults
from .schemas import (
    DeFiCategory,
    GovernanceType,
    OracleProvider,
    RiskGrade,
    SmartContractRisk,
    EconomicRisk,
    OracleRisk,
    GovernanceRisk,
    DeFiRiskScore,
    DeFiScoreRequest,
)
from .constants import DEFI_PROTOCOL_DEFAULTS, REPUTABLE_AUDITORS

__all__ = [
    # Router
    "router",
    # Service functions
    "score_defi_protocol",
    "get_protocol_defaults",
    "list_protocol_defaults",
    # Schemas
    "DeFiCategory",
    "GovernanceType",
    "OracleProvider",
    "RiskGrade",
    "SmartContractRisk",
    "EconomicRisk",
    "OracleRisk",
    "GovernanceRisk",
    "DeFiRiskScore",
    "DeFiScoreRequest",
    # Constants
    "DEFI_PROTOCOL_DEFAULTS",
    "REPUTABLE_AUDITORS",
]
