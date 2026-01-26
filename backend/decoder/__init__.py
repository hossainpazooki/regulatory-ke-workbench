"""Decoder domain - tiered explanations and counterfactual analysis."""

from .router import router
from .service import (
    DecoderService,
    CounterfactualEngine,
    CitationInjector,
    TemplateRegistry,
    DeltaAnalyzer,
)
from .schemas import (
    # Enums
    ExplanationTier,
    ScenarioType,
    # Citation
    Citation,
    # Explanation
    Explanation,
    ExplanationSummary,
    AuditInfo,
    # Decoder Request/Response
    DecoderRequest,
    DecoderResponse,
    # Counterfactual
    Scenario,
    OutcomeSummary,
    DeltaAnalysis,
    CounterfactualExplanation,
    CounterfactualRequest,
    CounterfactualResponse,
    # Comparison
    ComparisonRequest,
    MatrixInsight,
    ComparisonMatrix,
    # Templates
    TemplateVariable,
    CitationSlot,
    TemplateSection,
    ExplanationTemplate,
    # Router Request Models
    TemplateInfo,
    ExplainByDecisionRequest,
    InlineDecisionRequest,
    ScenarioRequest,
    AnalyzeByIdRequest,
    InlineAnalyzeRequest,
    CompareByIdRequest,
    InlineCompareRequest,
)

__all__ = [
    # Router
    "router",
    # Services
    "DecoderService",
    "CounterfactualEngine",
    "CitationInjector",
    "TemplateRegistry",
    "DeltaAnalyzer",
    # Enums
    "ExplanationTier",
    "ScenarioType",
    # Citation
    "Citation",
    # Explanation
    "Explanation",
    "ExplanationSummary",
    "AuditInfo",
    # Decoder Request/Response
    "DecoderRequest",
    "DecoderResponse",
    # Counterfactual
    "Scenario",
    "OutcomeSummary",
    "DeltaAnalysis",
    "CounterfactualExplanation",
    "CounterfactualRequest",
    "CounterfactualResponse",
    # Comparison
    "ComparisonRequest",
    "MatrixInsight",
    "ComparisonMatrix",
    # Templates
    "TemplateVariable",
    "CitationSlot",
    "TemplateSection",
    "ExplanationTemplate",
    # Router Request Models
    "TemplateInfo",
    "ExplainByDecisionRequest",
    "InlineDecisionRequest",
    "ScenarioRequest",
    "AnalyzeByIdRequest",
    "InlineAnalyzeRequest",
    "CompareByIdRequest",
    "InlineCompareRequest",
]
