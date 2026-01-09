"""Visualization module for decision trees and analytics (canonical location)."""

from .tree_adapter import (
    TreeNode,
    TreeEdge,
    TreeGraph,
    TreeAdapter,
    NodeConsistencyInfo,
    rule_to_graph,
    render_dot,
    render_mermaid,
    extract_trace_path,
)

from .supertree_adapters import (
    build_rulebook_outline,
    build_decision_trace_tree,
    build_ontology_tree,
    build_corpus_rule_links,
    build_decision_tree_structure,
    build_legal_corpus_coverage,
)

from .supertree_utils import (
    SUPERTREE_AVAILABLE,
    is_supertree_available,
    render_rulebook_outline_html,
    render_decision_trace_html,
    render_ontology_tree_html,
    render_corpus_links_html,
    render_legal_corpus_html,
    SupertreeNotAvailableError,
)

__all__ = [
    # Existing tree adapter exports
    "TreeNode",
    "TreeEdge",
    "TreeGraph",
    "TreeAdapter",
    "NodeConsistencyInfo",
    "rule_to_graph",
    "render_dot",
    "render_mermaid",
    "extract_trace_path",
    # Supertree data adapters
    "build_rulebook_outline",
    "build_decision_trace_tree",
    "build_ontology_tree",
    "build_corpus_rule_links",
    "build_decision_tree_structure",
    "build_legal_corpus_coverage",
    # Supertree rendering (optional)
    "SUPERTREE_AVAILABLE",
    "is_supertree_available",
    "render_rulebook_outline_html",
    "render_decision_trace_html",
    "render_ontology_tree_html",
    "render_corpus_links_html",
    "render_legal_corpus_html",
    "SupertreeNotAvailableError",
]
