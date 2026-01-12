"""
UI shared modules for KE Workbench.

This package contains reusable UI components and helpers used across
the dashboard pages.
"""

from frontend.ui.review_helpers import (
    get_status_color,
    get_status_emoji,
    get_priority_score,
    get_rule_issues,
    submit_review,
)
from frontend.ui.insights import (
    render_tree_view,
    render_chart,
    render_tool_gallery,
    ToolCard,
)
from frontend.ui.worklist import (
    WorklistItem,
    build_worklist,
    render_worklist_item,
    filter_worklist,
)
from frontend.ui.embedding_viz import (
    render_embedding_type_selector,
    render_umap_controls,
    render_umap_scatter,
    render_color_by_selector,
    render_selected_rule_details,
    render_cluster_selector,
)
from frontend.ui.similarity_cards import (
    render_search_mode_selector,
    render_weight_sliders,
    render_search_params,
    render_similarity_result,
    render_similarity_results,
    render_score_bar,
    render_export_buttons,
    render_rule_selector,
    render_text_query_input,
    render_entity_selector,
)
from frontend.ui.graph_components import (
    render_graph_controls,
    render_pyvis_graph,
    render_graph_stats,
    compute_graph_stats,
    get_node_colors,
    get_node_color,
    render_rule_graph_selector,
    render_graph_comparison,
    build_rule_graph,
    build_rule_network,
)

__all__ = [
    # Review helpers
    "get_status_color",
    "get_status_emoji",
    "get_priority_score",
    "get_rule_issues",
    "submit_review",
    # Insights
    "render_tree_view",
    "render_chart",
    "render_tool_gallery",
    "ToolCard",
    # Worklist
    "WorklistItem",
    "build_worklist",
    "render_worklist_item",
    "filter_worklist",
    # Embedding Visualization
    "render_embedding_type_selector",
    "render_umap_controls",
    "render_umap_scatter",
    "render_color_by_selector",
    "render_selected_rule_details",
    "render_cluster_selector",
    # Similarity Cards
    "render_search_mode_selector",
    "render_weight_sliders",
    "render_search_params",
    "render_similarity_result",
    "render_similarity_results",
    "render_score_bar",
    "render_export_buttons",
    "render_rule_selector",
    "render_text_query_input",
    "render_entity_selector",
    # Graph Components
    "render_graph_controls",
    "render_pyvis_graph",
    "render_graph_stats",
    "compute_graph_stats",
    "get_node_colors",
    "get_node_color",
    "render_rule_graph_selector",
    "render_graph_comparison",
    "build_rule_graph",
    "build_rule_network",
]
