"""
Tree Visualization Utilities.

This module provides HTML-based visualization for regulatory charts.
Uses a custom collapsible tree renderer for hierarchical data display.

Note: Supertree (pip install supertree) is for ML decision tree visualization,
not arbitrary tree data. We use a custom HTML renderer instead.
"""

from __future__ import annotations

import html
import json
from typing import Any

# We always have visualization available (pure HTML/CSS/JS)
SUPERTREE_AVAILABLE = True


class SupertreeNotAvailableError(Exception):
    """Raised when visualization functionality is not available."""
    pass


def _escape(text: Any) -> str:
    """Escape HTML entities in text."""
    return html.escape(str(text)) if text is not None else ""


def _render_tree_node(node: dict, depth: int = 0) -> str:
    """Recursively render a tree node as HTML."""
    title = _escape(node.get("title", "Node"))
    children = node.get("children", [])
    node_id = f"node_{id(node)}_{depth}"

    # Collect metadata (non-title, non-children keys)
    metadata = []
    for key, value in node.items():
        if key not in ("title", "children"):
            if isinstance(value, (str, int, float, bool)):
                metadata.append(f'<span class="tree-meta"><strong>{_escape(key)}:</strong> {_escape(value)}</span>')
            elif isinstance(value, list) and all(isinstance(v, (str, int)) for v in value):
                metadata.append(f'<span class="tree-meta"><strong>{_escape(key)}:</strong> {_escape(", ".join(map(str, value)))}</span>')

    meta_html = " ".join(metadata) if metadata else ""

    if children:
        # Branch node with children
        children_html = "\n".join(_render_tree_node(child, depth + 1) for child in children)
        return f'''
        <details class="tree-node" {"open" if depth < 1 else ""}>
            <summary class="tree-branch">
                <span class="tree-icon">▶</span>
                <span class="tree-title">{title}</span>
                {meta_html}
            </summary>
            <div class="tree-children">
                {children_html}
            </div>
        </details>
        '''
    else:
        # Leaf node
        return f'''
        <div class="tree-leaf">
            <span class="tree-icon">•</span>
            <span class="tree-title">{title}</span>
            {meta_html}
        </div>
        '''


def _render_tree_html(tree_data: dict, chart_title: str) -> str:
    """Render a complete tree as interactive HTML."""
    tree_content = _render_tree_node(tree_data)

    return f'''
    <div class="tree-container">
        <style>
            .tree-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                line-height: 1.5;
                padding: 16px;
                background: #fafafa;
                border-radius: 8px;
                max-height: 600px;
                overflow: auto;
            }}
            .tree-header {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e0e0e0;
                color: #333;
            }}
            .tree-node {{
                margin: 4px 0;
            }}
            .tree-branch {{
                cursor: pointer;
                padding: 6px 8px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }}
            .tree-branch:hover {{
                background: #e8f4fc;
            }}
            .tree-leaf {{
                padding: 6px 8px 6px 24px;
                display: flex;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
            }}
            .tree-children {{
                margin-left: 20px;
                padding-left: 12px;
                border-left: 2px solid #ddd;
            }}
            .tree-icon {{
                color: #666;
                font-size: 10px;
                width: 12px;
                transition: transform 0.2s;
            }}
            details[open] > summary .tree-icon {{
                transform: rotate(90deg);
            }}
            .tree-title {{
                font-weight: 500;
                color: #1a73e8;
            }}
            .tree-leaf .tree-title {{
                color: #333;
                font-weight: 400;
            }}
            .tree-meta {{
                font-size: 12px;
                color: #666;
                background: #e8e8e8;
                padding: 2px 6px;
                border-radius: 3px;
                margin-left: 4px;
            }}
            details > summary {{
                list-style: none;
            }}
            details > summary::-webkit-details-marker {{
                display: none;
            }}
        </style>
        <div class="tree-header">{_escape(chart_title)}</div>
        {tree_content}
    </div>
    '''


def render_rulebook_outline_html(tree_data: dict) -> str:
    """Render the rulebook outline tree as HTML.

    Args:
        tree_data: Nested dict from build_rulebook_outline()

    Returns:
        HTML string with interactive tree visualization.
    """
    try:
        return _render_tree_html(tree_data, tree_data.get("title", "Rulebook Outline"))
    except Exception as e:
        return f'<div style="color: red; padding: 20px;">Error rendering chart: {_escape(str(e))}</div>'


def render_decision_trace_html(tree_data: dict) -> str:
    """Render the decision trace tree as HTML.

    Args:
        tree_data: Nested dict from build_decision_trace_tree()

    Returns:
        HTML string with interactive tree visualization.
    """
    try:
        return _render_tree_html(tree_data, tree_data.get("title", "Decision Trace"))
    except Exception as e:
        return f'<div style="color: red; padding: 20px;">Error rendering chart: {_escape(str(e))}</div>'


def render_ontology_tree_html(tree_data: dict) -> str:
    """Render the ontology browser tree as HTML.

    Args:
        tree_data: Nested dict from build_ontology_tree()

    Returns:
        HTML string with interactive tree visualization.
    """
    try:
        return _render_tree_html(tree_data, tree_data.get("title", "Regulatory Ontology"))
    except Exception as e:
        return f'<div style="color: red; padding: 20px;">Error rendering chart: {_escape(str(e))}</div>'


def render_corpus_links_html(tree_data: dict) -> str:
    """Render the corpus-to-rule links tree as HTML.

    Args:
        tree_data: Nested dict from build_corpus_rule_links()

    Returns:
        HTML string with interactive tree visualization.
    """
    try:
        return _render_tree_html(tree_data, tree_data.get("title", "Corpus-Rule Links"))
    except Exception as e:
        return f'<div style="color: red; padding: 20px;">Error rendering chart: {_escape(str(e))}</div>'


def _render_coverage_node(node: dict, depth: int = 0) -> str:
    """Render a coverage node with status-based styling."""
    title = _escape(node.get("title", "Node"))
    children = node.get("children", [])
    status = node.get("status", "")
    node_id = f"cov_{id(node)}_{depth}"

    # Determine status color
    if status == "covered":
        status_class = "coverage-covered"
        status_icon = "&#x2714;"  # Checkmark
    elif status == "gap":
        status_class = "coverage-gap"
        status_icon = "&#x26A0;"  # Warning
    else:
        status_class = ""
        status_icon = ""

    # Collect metadata
    metadata = []
    for key, value in node.items():
        if key not in ("title", "children", "status", "has_rules", "rules"):
            if isinstance(value, (str, int, float, bool)):
                metadata.append(f'<span class="tree-meta"><strong>{_escape(key)}:</strong> {_escape(value)}</span>')

    meta_html = " ".join(metadata) if metadata else ""

    # Status badge
    status_badge = f'<span class="coverage-badge {status_class}">{status_icon}</span>' if status else ""

    if children:
        # Branch node
        children_html = "\n".join(_render_coverage_node(child, depth + 1) for child in children)
        return f'''
        <details class="tree-node coverage-node" {"open" if depth < 1 else ""}>
            <summary class="tree-branch {status_class}">
                <span class="tree-icon">▶</span>
                {status_badge}
                <span class="tree-title">{title}</span>
                {meta_html}
            </summary>
            <div class="tree-children">
                {children_html}
            </div>
        </details>
        '''
    else:
        # Leaf node
        return f'''
        <div class="tree-leaf {status_class}">
            {status_badge}
            <span class="tree-title">{title}</span>
            {meta_html}
        </div>
        '''


def render_legal_corpus_html(tree_data: dict) -> str:
    """Render the legal corpus coverage tree as HTML.

    Args:
        tree_data: Nested dict from build_legal_corpus_coverage()

    Returns:
        HTML string with interactive tree visualization with coverage indicators.
    """
    try:
        tree_content = _render_coverage_node(tree_data)

        return f'''
        <div class="coverage-container">
            <style>
                .coverage-container {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-size: 14px;
                    line-height: 1.5;
                    padding: 16px;
                    background: #fafafa;
                    border-radius: 8px;
                    max-height: 600px;
                    overflow: auto;
                }}
                .coverage-header {{
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 16px;
                    padding-bottom: 8px;
                    border-bottom: 2px solid #e0e0e0;
                    color: #333;
                }}
                .coverage-legend {{
                    display: flex;
                    gap: 16px;
                    margin-bottom: 16px;
                    padding: 8px;
                    background: #f0f0f0;
                    border-radius: 4px;
                }}
                .coverage-legend-item {{
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-size: 12px;
                }}
                .coverage-node {{
                    margin: 4px 0;
                }}
                .tree-branch {{
                    cursor: pointer;
                    padding: 6px 8px;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                }}
                .tree-branch:hover {{
                    background: #e8f4fc;
                }}
                .tree-leaf {{
                    padding: 6px 8px 6px 24px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    flex-wrap: wrap;
                }}
                .tree-children {{
                    margin-left: 20px;
                    padding-left: 12px;
                    border-left: 2px solid #ddd;
                }}
                .tree-icon {{
                    color: #666;
                    font-size: 10px;
                    width: 12px;
                    transition: transform 0.2s;
                }}
                details[open] > summary .tree-icon {{
                    transform: rotate(90deg);
                }}
                .tree-title {{
                    font-weight: 500;
                    color: #1a73e8;
                }}
                .tree-leaf .tree-title {{
                    color: #333;
                    font-weight: 400;
                }}
                .tree-meta {{
                    font-size: 12px;
                    color: #666;
                    background: #e8e8e8;
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin-left: 4px;
                }}
                details > summary {{
                    list-style: none;
                }}
                details > summary::-webkit-details-marker {{
                    display: none;
                }}
                /* Coverage-specific styles */
                .coverage-badge {{
                    font-size: 14px;
                    margin-right: 4px;
                }}
                .coverage-covered .coverage-badge {{
                    color: #28a745;
                }}
                .coverage-gap .coverage-badge {{
                    color: #dc3545;
                }}
                .coverage-covered.tree-leaf {{
                    background: rgba(40, 167, 69, 0.1);
                    border-radius: 4px;
                }}
                .coverage-gap.tree-leaf {{
                    background: rgba(220, 53, 69, 0.1);
                    border-radius: 4px;
                }}
                .coverage-progress {{
                    width: 100px;
                    height: 8px;
                    background: #e0e0e0;
                    border-radius: 4px;
                    overflow: hidden;
                    margin-left: 8px;
                }}
                .coverage-progress-bar {{
                    height: 100%;
                    background: linear-gradient(90deg, #28a745, #20c997);
                    transition: width 0.3s;
                }}
            </style>
            <div class="coverage-header">{_escape(tree_data.get("title", "Legal Corpus Coverage"))}</div>
            <div class="coverage-legend">
                <div class="coverage-legend-item">
                    <span style="color: #28a745;">&#x2714;</span>
                    <span>Covered (has rules)</span>
                </div>
                <div class="coverage-legend-item">
                    <span style="color: #dc3545;">&#x26A0;</span>
                    <span>Gap (no rules)</span>
                </div>
            </div>
            {tree_content}
        </div>
        '''
    except Exception as e:
        return f'<div style="color: red; padding: 20px;">Error rendering chart: {_escape(str(e))}</div>'


def is_supertree_available() -> bool:
    """Check if tree visualization is available.

    Always returns True since we use a custom HTML renderer.
    """
    return SUPERTREE_AVAILABLE
