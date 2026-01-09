"""
Tree Adapter - Converts decision trees to visualization-ready format.

This module transforms the Rule's DecisionBranch/DecisionLeaf structure
into a flat graph representation suitable for rendering with Graphviz or
similar libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from backend.rule_service.app.services.schema import (
    Rule,
    DecisionBranch,
    DecisionLeaf,
    ConditionSpec,
    ConsistencyBlock,
    ConsistencyEvidence,
    ConsistencyStatus,
)


# =============================================================================
# Data Classes for Graph Representation
# =============================================================================


@dataclass
class NodeConsistencyInfo:
    """Consistency information for a single tree node."""

    status: str = "unverified"
    confidence: float = 0.0
    evidence: list[ConsistencyEvidence] = field(default_factory=list)
    pass_count: int = 0
    fail_count: int = 0
    warning_count: int = 0

    @property
    def color(self) -> str:
        """Get color for visualization based on status."""
        return {
            "verified": "#28a745",      # green
            "needs_review": "#ffc107",  # yellow/amber
            "inconsistent": "#dc3545",  # red
            "unverified": "#6c757d",    # gray
        }.get(self.status, "#6c757d")

    @property
    def emoji(self) -> str:
        """Get emoji indicator for status."""
        return {
            "verified": "✓",
            "needs_review": "?",
            "inconsistent": "✗",
            "unverified": "○",
        }.get(self.status, "○")

    @property
    def border_color(self) -> str:
        """Get border color (darker variant) for visualization."""
        return {
            "verified": "#1e7e34",
            "needs_review": "#d39e00",
            "inconsistent": "#bd2130",
            "unverified": "#545b62",
        }.get(self.status, "#545b62")


@dataclass
class TreeNode:
    """A node in the visualization graph."""

    id: str
    node_type: Literal["branch", "leaf", "root"]
    label: str
    description: str | None = None

    # Branch-specific fields
    condition_field: str | None = None
    condition_operator: str | None = None
    condition_value: str | None = None

    # Leaf-specific fields
    decision: str | None = None
    obligations: list[str] = field(default_factory=list)

    # Consistency info
    consistency: NodeConsistencyInfo = field(default_factory=NodeConsistencyInfo)

    # Layout hints
    depth: int = 0
    position: int = 0


@dataclass
class TreeEdge:
    """An edge connecting two nodes."""

    source_id: str
    target_id: str
    label: str  # "true" or "false"
    is_true_branch: bool = True

    @property
    def color(self) -> str:
        """Edge color based on branch type."""
        return "#28a745" if self.is_true_branch else "#dc3545"

    @property
    def style(self) -> str:
        """Edge line style."""
        return "solid"


@dataclass
class TreeGraph:
    """Complete graph representation of a decision tree."""

    rule_id: str
    nodes: list[TreeNode] = field(default_factory=list)
    edges: list[TreeEdge] = field(default_factory=list)

    # Aggregate consistency
    overall_status: str = "unverified"
    overall_confidence: float = 0.0
    total_pass: int = 0
    total_fail: int = 0
    total_warning: int = 0

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_root(self) -> TreeNode | None:
        """Get the root node."""
        for node in self.nodes:
            if node.node_type == "root" or node.depth == 0:
                return node
        return self.nodes[0] if self.nodes else None

    def get_children(self, node_id: str) -> list[tuple[TreeEdge, TreeNode]]:
        """Get child edges and nodes for a given node."""
        result = []
        for edge in self.edges:
            if edge.source_id == node_id:
                node = self.get_node(edge.target_id)
                if node:
                    result.append((edge, node))
        return result

    def to_dot(
        self,
        show_consistency: bool = True,
        highlight_nodes: set[str] | None = None,
        highlight_edges: set[tuple[str, str]] | None = None,
    ) -> str:
        """Generate Graphviz DOT format string.

        Args:
            show_consistency: Whether to color nodes by consistency status
            highlight_nodes: Set of node IDs to highlight (e.g., trace path)
            highlight_edges: Set of (source_id, target_id) tuples to highlight

        Returns:
            DOT format string for Graphviz rendering
        """
        highlight_nodes = highlight_nodes or set()
        highlight_edges = highlight_edges or set()

        lines = [
            "digraph DecisionTree {",
            '    rankdir=TB;',
            '    node [shape=box, style="rounded,filled", fontname="Arial"];',
            '    edge [fontname="Arial", fontsize=10];',
            "",
        ]

        # Add nodes
        for node in self.nodes:
            is_highlighted = node.id in highlight_nodes

            if is_highlighted:
                # Highlighted nodes: yellow background, bold black border
                fill_color = "#fff3cd"  # Light yellow
                border_color = "#000000"  # Black
                penwidth = 4
            elif show_consistency:
                fill_color = node.consistency.color
                border_color = node.consistency.border_color
                penwidth = 2
            else:
                fill_color = "#e9ecef"
                border_color = "#495057"
                penwidth = 2

            # Build label
            if node.node_type == "leaf":
                label = f"{node.decision or 'Unknown'}"
                shape = "ellipse"
            else:
                if node.condition_field:
                    label = f"{node.condition_field}\\n{node.condition_operator} {node.condition_value}"
                else:
                    label = node.label
                shape = "box"

            # Add consistency indicator to label
            if show_consistency and node.consistency.status != "unverified":
                label = f"{node.consistency.emoji} {label}"

            # Add trace marker for highlighted nodes
            if is_highlighted:
                label = f"→ {label}"

            lines.append(
                f'    "{node.id}" ['
                f'label="{label}", '
                f'shape={shape}, '
                f'fillcolor="{fill_color}", '
                f'color="{border_color}", '
                f'penwidth={penwidth}'
                f'];'
            )

        lines.append("")

        # Add edges
        for edge in self.edges:
            is_highlighted = (edge.source_id, edge.target_id) in highlight_edges

            if is_highlighted:
                # Highlighted edges: black, thicker
                color = "#000000"
                penwidth = 3
                style = "bold"
            else:
                color = edge.color
                penwidth = 1
                style = "solid"

            label = "T" if edge.is_true_branch else "F"
            lines.append(
                f'    "{edge.source_id}" -> "{edge.target_id}" ['
                f'label="{label}", '
                f'color="{color}", '
                f'fontcolor="{color}", '
                f'penwidth={penwidth}, '
                f'style={style}'
                f'];'
            )

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, show_consistency: bool = True) -> str:
        """Generate Mermaid flowchart format string."""
        lines = ["flowchart TD"]

        # Node definitions with styling
        for node in self.nodes:
            if node.node_type == "leaf":
                # Rounded rectangle for leaves
                if node.decision:
                    label = node.decision
                else:
                    label = "Unknown"
                lines.append(f'    {node.id}(("{label}"))')
            else:
                # Rectangle for branches
                if node.condition_field:
                    label = f"{node.condition_field} {node.condition_operator} {node.condition_value}"
                else:
                    label = node.label
                lines.append(f'    {node.id}["{label}"]')

        # Add edges
        for edge in self.edges:
            label = "Yes" if edge.is_true_branch else "No"
            lines.append(f'    {edge.source_id} -->|{label}| {edge.target_id}')

        # Add styling
        if show_consistency:
            lines.append("")
            for node in self.nodes:
                color = node.consistency.color.replace("#", "")
                lines.append(f'    style {node.id} fill:#{color}')

        return "\n".join(lines)


# =============================================================================
# Tree Adapter
# =============================================================================


class TreeAdapter:
    """Converts Rule decision trees to visualization graphs."""

    def __init__(self) -> None:
        self._node_counter = 0

    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _reset_counter(self) -> None:
        """Reset the node counter for a new tree."""
        self._node_counter = 0

    def convert(
        self,
        rule: Rule,
        node_consistency_map: dict[str, NodeConsistencyInfo] | None = None,
    ) -> TreeGraph:
        """Convert a Rule's decision tree to a TreeGraph.

        Args:
            rule: The rule to convert
            node_consistency_map: Optional mapping of node IDs to consistency info

        Returns:
            TreeGraph representation of the decision tree
        """
        self._reset_counter()
        node_consistency_map = node_consistency_map or {}

        graph = TreeGraph(rule_id=rule.rule_id)

        if rule.decision_tree is None:
            # Empty tree - create a single node
            empty_node = TreeNode(
                id="empty",
                node_type="leaf",
                label="No decision tree defined",
                decision="undefined",
            )
            graph.nodes.append(empty_node)
            return graph

        # Recursively build the tree
        self._build_tree(
            node=rule.decision_tree,
            graph=graph,
            node_consistency_map=node_consistency_map,
            depth=0,
            position=0,
        )

        # Set overall consistency from rule
        if rule.consistency:
            graph.overall_status = rule.consistency.summary.status.value
            graph.overall_confidence = rule.consistency.summary.confidence
            graph.total_pass = sum(
                1 for e in rule.consistency.evidence if e.label == "pass"
            )
            graph.total_fail = sum(
                1 for e in rule.consistency.evidence if e.label == "fail"
            )
            graph.total_warning = sum(
                1 for e in rule.consistency.evidence if e.label == "warning"
            )

        return graph

    def _build_tree(
        self,
        node: DecisionBranch | DecisionLeaf,
        graph: TreeGraph,
        node_consistency_map: dict[str, NodeConsistencyInfo],
        depth: int,
        position: int,
        parent_id: str | None = None,
        is_true_branch: bool = True,
    ) -> str:
        """Recursively build tree nodes and edges.

        Returns the ID of the created node.
        """
        # Check for leaf by class name (Pydantic v2 intercepts hasattr/isinstance)
        node_class = type(node).__name__
        is_leaf = node_class == 'DecisionLeaf' or 'result' in type(node).model_fields

        if is_leaf:
            return self._build_leaf(
                node, graph, node_consistency_map, depth, position, parent_id, is_true_branch
            )
        else:
            return self._build_branch(
                node, graph, node_consistency_map, depth, position, parent_id, is_true_branch
            )

    def _build_leaf(
        self,
        node: DecisionLeaf,
        graph: TreeGraph,
        node_consistency_map: dict[str, NodeConsistencyInfo],
        depth: int,
        position: int,
        parent_id: str | None,
        is_true_branch: bool,
    ) -> str:
        """Build a leaf node."""
        node_id = self._generate_node_id("leaf")

        # Get consistency info if available
        consistency = node_consistency_map.get(node_id, NodeConsistencyInfo())

        tree_node = TreeNode(
            id=node_id,
            node_type="leaf",
            label=node.result,
            description=node.notes,
            decision=node.result,
            obligations=[obl.id for obl in node.obligations],
            consistency=consistency,
            depth=depth,
            position=position,
        )
        graph.nodes.append(tree_node)

        # Add edge from parent
        if parent_id:
            edge = TreeEdge(
                source_id=parent_id,
                target_id=node_id,
                label="true" if is_true_branch else "false",
                is_true_branch=is_true_branch,
            )
            graph.edges.append(edge)

        return node_id

    def _build_branch(
        self,
        node: DecisionBranch,
        graph: TreeGraph,
        node_consistency_map: dict[str, NodeConsistencyInfo],
        depth: int,
        position: int,
        parent_id: str | None,
        is_true_branch: bool,
    ) -> str:
        """Build a branch node."""
        # Use the node_id from the branch if available
        node_id = node.node_id or self._generate_node_id("branch")

        # Get consistency info if available
        consistency = node_consistency_map.get(node_id, NodeConsistencyInfo())

        # Build condition label
        condition_field = None
        condition_operator = None
        condition_value = None
        label = node_id

        if node.condition:
            condition_field = node.condition.field
            condition_operator = node.condition.operator
            condition_value = str(node.condition.value) if node.condition.value is not None else "null"
            label = f"{condition_field} {condition_operator} {condition_value}"

        # Safely get description - Pydantic v2 raises AttributeError for missing optional fields
        condition_description = None
        if node.condition:
            try:
                condition_description = node.condition.description
            except AttributeError:
                pass

        tree_node = TreeNode(
            id=node_id,
            node_type="root" if depth == 0 else "branch",
            label=label,
            description=condition_description,
            condition_field=condition_field,
            condition_operator=condition_operator,
            condition_value=condition_value,
            consistency=consistency,
            depth=depth,
            position=position,
        )
        graph.nodes.append(tree_node)

        # Add edge from parent
        if parent_id:
            edge = TreeEdge(
                source_id=parent_id,
                target_id=node_id,
                label="true" if is_true_branch else "false",
                is_true_branch=is_true_branch,
            )
            graph.edges.append(edge)

        # Recursively build children
        next_position = 0

        if node.true_branch:
            self._build_tree(
                node=node.true_branch,
                graph=graph,
                node_consistency_map=node_consistency_map,
                depth=depth + 1,
                position=next_position,
                parent_id=node_id,
                is_true_branch=True,
            )
            next_position += 1

        if node.false_branch:
            self._build_tree(
                node=node.false_branch,
                graph=graph,
                node_consistency_map=node_consistency_map,
                depth=depth + 1,
                position=next_position,
                parent_id=node_id,
                is_true_branch=False,
            )

        return node_id

    def build_node_consistency_map(
        self,
        rule: Rule,
        consistency_block: ConsistencyBlock | None = None,
    ) -> dict[str, NodeConsistencyInfo]:
        """Build a mapping of node IDs to consistency information.

        This aggregates evidence by rule_element field to associate
        consistency checks with specific tree nodes.
        """
        result: dict[str, NodeConsistencyInfo] = {}

        if consistency_block is None:
            consistency_block = rule.consistency

        if consistency_block is None:
            return result

        # Group evidence by rule_element (node ID)
        for evidence in consistency_block.evidence:
            if evidence.rule_element:
                node_id = evidence.rule_element
            else:
                # Associate with root if no specific element
                node_id = "__rule__"

            if node_id not in result:
                result[node_id] = NodeConsistencyInfo()

            result[node_id].evidence.append(evidence)

            if evidence.label == "pass":
                result[node_id].pass_count += 1
            elif evidence.label == "fail":
                result[node_id].fail_count += 1
            elif evidence.label == "warning":
                result[node_id].warning_count += 1

        # Compute status for each node
        for node_id, info in result.items():
            if info.fail_count > 0:
                info.status = "inconsistent"
            elif info.warning_count > 0:
                info.status = "needs_review"
            elif info.pass_count > 0:
                info.status = "verified"
            else:
                info.status = "unverified"

            # Compute confidence as pass ratio
            total = info.pass_count + info.fail_count + info.warning_count
            if total > 0:
                info.confidence = info.pass_count / total

        return result


# =============================================================================
# Utility Functions
# =============================================================================


def rule_to_graph(
    rule: Rule,
    include_consistency: bool = True,
) -> TreeGraph:
    """Convenience function to convert a rule to a tree graph.

    Args:
        rule: The rule to convert
        include_consistency: Whether to include consistency overlay

    Returns:
        TreeGraph representation
    """
    adapter = TreeAdapter()

    node_map = {}
    if include_consistency and rule.consistency:
        node_map = adapter.build_node_consistency_map(rule)

    return adapter.convert(rule, node_map)


def render_dot(
    graph: TreeGraph,
    show_consistency: bool = True,
    highlight_nodes: set[str] | None = None,
    highlight_edges: set[tuple[str, str]] | None = None,
) -> str:
    """Render a tree graph as Graphviz DOT format."""
    return graph.to_dot(
        show_consistency=show_consistency,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
    )


def render_mermaid(graph: TreeGraph, show_consistency: bool = True) -> str:
    """Render a tree graph as Mermaid flowchart format."""
    return graph.to_mermaid(show_consistency=show_consistency)


def extract_trace_path(trace: list) -> tuple[set[str], set[tuple[str, str]]]:
    """Extract highlighted nodes and edges from a decision trace.

    Args:
        trace: List of TraceStep objects from DecisionResult

    Returns:
        Tuple of (highlight_nodes set, highlight_edges set)
    """
    highlight_nodes: set[str] = set()
    highlight_edges: set[tuple[str, str]] = set()

    prev_node_id = None

    for step in trace:
        # Extract node ID from trace step
        # TraceStep has node_path like "check_exemption" or "applicability.all[0]"
        node_id = getattr(step, "node_path", None) or getattr(step, "node", None)

        if node_id:
            # Clean up node_path to get actual node ID
            # Handle paths like "decision_tree.check_exemption" -> "check_exemption"
            if "." in node_id:
                parts = node_id.split(".")
                # Take the last part that looks like a node ID
                for part in reversed(parts):
                    if not part.startswith("all[") and not part.startswith("any["):
                        node_id = part
                        break

            highlight_nodes.add(node_id)

            # Add edge from previous node
            if prev_node_id and prev_node_id != node_id:
                highlight_edges.add((prev_node_id, node_id))

            prev_node_id = node_id

    return highlight_nodes, highlight_edges
