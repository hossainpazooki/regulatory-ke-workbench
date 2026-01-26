"""Tests for the tree adapter visualization module."""

import pytest

from backend.rules import (
    Rule,
    DecisionBranch,
    DecisionLeaf,
    ConditionSpec,
    ObligationSpec,
    ConsistencyBlock,
    ConsistencySummary,
    ConsistencyEvidence,
    ConsistencyStatus,
)
from backend.core.visualization.tree_adapter import (
    TreeAdapter,
    TreeNode,
    TreeEdge,
    TreeGraph,
    NodeConsistencyInfo,
    rule_to_graph,
    render_dot,
    render_mermaid,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_rule() -> Rule:
    """Create a simple rule with a basic decision tree."""
    return Rule(
        rule_id="test_simple_rule",
        description="A simple test rule",
        decision_tree=DecisionBranch(
            node_id="root",
            condition=ConditionSpec(
                field="actor.type",
                operator="==",
                value="legal_person",
                description="Check if actor is legal person",
            ),
            true_branch=DecisionLeaf(
                result="authorized",
                obligations=[
                    ObligationSpec(id="OBL001", description="File annual report")
                ],
                notes="Legal persons are authorized",
            ),
            false_branch=DecisionLeaf(
                result="not_authorized",
                notes="Natural persons need additional checks",
            ),
        ),
    )


@pytest.fixture
def nested_rule() -> Rule:
    """Create a rule with nested decision branches."""
    return Rule(
        rule_id="test_nested_rule",
        description="A rule with nested branches",
        decision_tree=DecisionBranch(
            node_id="root",
            condition=ConditionSpec(
                field="actor.type",
                operator="==",
                value="legal_person",
            ),
            true_branch=DecisionBranch(
                node_id="branch_1",
                condition=ConditionSpec(
                    field="actor.jurisdiction",
                    operator="in",
                    value=["EU", "EEA"],
                ),
                true_branch=DecisionLeaf(result="authorized"),
                false_branch=DecisionLeaf(result="restricted"),
            ),
            false_branch=DecisionLeaf(result="not_authorized"),
        ),
    )


@pytest.fixture
def rule_with_consistency() -> Rule:
    """Create a rule with consistency metadata."""
    return Rule(
        rule_id="test_consistency_rule",
        description="A rule with consistency checks",
        decision_tree=DecisionBranch(
            node_id="root",
            condition=ConditionSpec(
                field="instrument.type",
                operator="==",
                value="e_money_token",
            ),
            true_branch=DecisionLeaf(result="regulated"),
            false_branch=DecisionLeaf(result="unregulated"),
        ),
        consistency=ConsistencyBlock(
            summary=ConsistencySummary(
                status=ConsistencyStatus.NEEDS_REVIEW,
                confidence=0.75,
                last_verified="2024-12-10T10:00:00Z",
                verified_by="system",
            ),
            evidence=[
                ConsistencyEvidence(
                    tier=0,
                    category="schema_valid",
                    label="pass",
                    score=1.0,
                    details="All required fields present",
                    rule_element="root",
                ),
                ConsistencyEvidence(
                    tier=1,
                    category="deontic_alignment",
                    label="warning",
                    score=0.6,
                    details="Deontic verb mismatch",
                    rule_element="root",
                ),
                ConsistencyEvidence(
                    tier=1,
                    category="keyword_overlap",
                    label="pass",
                    score=0.8,
                    details="Good keyword coverage",
                ),
            ],
        ),
    )


@pytest.fixture
def empty_rule() -> Rule:
    """Create a rule with no decision tree."""
    return Rule(
        rule_id="test_empty_rule",
        description="A rule with no decision tree",
        decision_tree=None,
    )


# =============================================================================
# TreeAdapter Tests
# =============================================================================


class TestTreeAdapter:
    """Tests for TreeAdapter class."""

    def test_convert_simple_rule(self, simple_rule: Rule) -> None:
        """Test converting a simple rule to a tree graph."""
        adapter = TreeAdapter()
        graph = adapter.convert(simple_rule)

        assert graph.rule_id == "test_simple_rule"
        assert len(graph.nodes) == 3  # 1 root + 2 leaves
        assert len(graph.edges) == 2  # 2 edges from root

        # Check root node
        root = graph.get_root()
        assert root is not None
        assert root.node_type == "root"
        assert root.condition_field == "actor.type"
        assert root.condition_operator == "=="
        assert root.condition_value == "legal_person"

        # Check edges
        children = graph.get_children(root.id)
        assert len(children) == 2

        # Find true and false branches
        true_edges = [e for e, n in children if e.is_true_branch]
        false_edges = [e for e, n in children if not e.is_true_branch]
        assert len(true_edges) == 1
        assert len(false_edges) == 1

    def test_convert_nested_rule(self, nested_rule: Rule) -> None:
        """Test converting a nested rule to a tree graph."""
        adapter = TreeAdapter()
        graph = adapter.convert(nested_rule)

        assert graph.rule_id == "test_nested_rule"
        assert len(graph.nodes) == 5  # 2 branches + 3 leaves
        assert len(graph.edges) == 4

        # Verify depth levels
        depths = {n.depth for n in graph.nodes}
        assert 0 in depths  # root
        assert 1 in depths  # first level children
        assert 2 in depths  # second level children

    def test_convert_empty_rule(self, empty_rule: Rule) -> None:
        """Test converting a rule with no decision tree."""
        adapter = TreeAdapter()
        graph = adapter.convert(empty_rule)

        assert graph.rule_id == "test_empty_rule"
        assert len(graph.nodes) == 1
        assert graph.nodes[0].decision == "undefined"

    def test_convert_with_consistency(self, rule_with_consistency: Rule) -> None:
        """Test converting a rule with consistency metadata."""
        adapter = TreeAdapter()

        # Build node consistency map
        node_map = adapter.build_node_consistency_map(rule_with_consistency)

        # Convert with consistency overlay
        graph = adapter.convert(rule_with_consistency, node_map)

        assert graph.overall_status == "needs_review"
        assert graph.overall_confidence == 0.75
        assert graph.total_pass == 2
        assert graph.total_fail == 0
        assert graph.total_warning == 1

    def test_build_node_consistency_map(self, rule_with_consistency: Rule) -> None:
        """Test building node consistency map from evidence."""
        adapter = TreeAdapter()
        node_map = adapter.build_node_consistency_map(rule_with_consistency)

        # Should have entries for 'root' and '__rule__'
        assert "root" in node_map
        assert "__rule__" in node_map

        # Root node should have mixed evidence
        root_info = node_map["root"]
        assert root_info.pass_count == 1
        assert root_info.warning_count == 1
        assert root_info.status == "needs_review"  # warning present


class TestTreeGraph:
    """Tests for TreeGraph class."""

    def test_get_node(self, simple_rule: Rule) -> None:
        """Test getting a node by ID."""
        graph = rule_to_graph(simple_rule)

        root = graph.get_root()
        assert root is not None

        found = graph.get_node(root.id)
        assert found == root

        not_found = graph.get_node("nonexistent")
        assert not_found is None

    def test_get_children(self, simple_rule: Rule) -> None:
        """Test getting child nodes."""
        graph = rule_to_graph(simple_rule)
        root = graph.get_root()

        children = graph.get_children(root.id)
        assert len(children) == 2

        # Children should be leaves
        for edge, node in children:
            assert node.node_type == "leaf"

    def test_to_dot(self, simple_rule: Rule) -> None:
        """Test DOT format generation."""
        graph = rule_to_graph(simple_rule)
        dot = graph.to_dot()

        assert "digraph DecisionTree" in dot
        assert "rankdir=TB" in dot
        assert "->" in dot  # edges
        assert "actor.type" in dot  # condition

    def test_to_dot_without_consistency(self, simple_rule: Rule) -> None:
        """Test DOT format without consistency overlay."""
        graph = rule_to_graph(simple_rule)
        dot = graph.to_dot(show_consistency=False)

        assert "digraph DecisionTree" in dot
        # Should not have status emojis
        assert "✓" not in dot
        assert "✗" not in dot

    def test_to_mermaid(self, simple_rule: Rule) -> None:
        """Test Mermaid format generation."""
        graph = rule_to_graph(simple_rule)
        mermaid = graph.to_mermaid()

        assert "flowchart TD" in mermaid
        assert "-->|Yes|" in mermaid or "-->|No|" in mermaid


class TestNodeConsistencyInfo:
    """Tests for NodeConsistencyInfo class."""

    def test_color_mapping(self) -> None:
        """Test status to color mapping."""
        info = NodeConsistencyInfo(status="verified")
        assert info.color == "#28a745"

        info.status = "needs_review"
        assert info.color == "#ffc107"

        info.status = "inconsistent"
        assert info.color == "#dc3545"

        info.status = "unverified"
        assert info.color == "#6c757d"

    def test_emoji_mapping(self) -> None:
        """Test status to emoji mapping."""
        info = NodeConsistencyInfo(status="verified")
        assert info.emoji == "✓"

        info.status = "inconsistent"
        assert info.emoji == "✗"

    def test_border_color(self) -> None:
        """Test border color is darker variant."""
        info = NodeConsistencyInfo(status="verified")
        assert info.border_color == "#1e7e34"


class TestTreeNode:
    """Tests for TreeNode class."""

    def test_branch_node_creation(self) -> None:
        """Test creating a branch node."""
        node = TreeNode(
            id="test_branch",
            node_type="branch",
            label="test condition",
            condition_field="actor.type",
            condition_operator="==",
            condition_value="legal_person",
            depth=1,
            position=0,
        )

        assert node.id == "test_branch"
        assert node.node_type == "branch"
        assert node.condition_field == "actor.type"
        assert node.decision is None

    def test_leaf_node_creation(self) -> None:
        """Test creating a leaf node."""
        node = TreeNode(
            id="test_leaf",
            node_type="leaf",
            label="authorized",
            decision="authorized",
            obligations=["OBL001", "OBL002"],
            depth=2,
            position=1,
        )

        assert node.id == "test_leaf"
        assert node.node_type == "leaf"
        assert node.decision == "authorized"
        assert len(node.obligations) == 2


class TestTreeEdge:
    """Tests for TreeEdge class."""

    def test_true_branch_edge(self) -> None:
        """Test true branch edge properties."""
        edge = TreeEdge(
            source_id="parent",
            target_id="child",
            label="true",
            is_true_branch=True,
        )

        assert edge.color == "#28a745"  # green
        assert edge.style == "solid"

    def test_false_branch_edge(self) -> None:
        """Test false branch edge properties."""
        edge = TreeEdge(
            source_id="parent",
            target_id="child",
            label="false",
            is_true_branch=False,
        )

        assert edge.color == "#dc3545"  # red


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_rule_to_graph(self, simple_rule: Rule) -> None:
        """Test rule_to_graph convenience function."""
        graph = rule_to_graph(simple_rule)

        assert isinstance(graph, TreeGraph)
        assert graph.rule_id == "test_simple_rule"

    def test_rule_to_graph_with_consistency(self, rule_with_consistency: Rule) -> None:
        """Test rule_to_graph with consistency enabled."""
        graph = rule_to_graph(rule_with_consistency, include_consistency=True)

        assert graph.overall_status == "needs_review"

    def test_rule_to_graph_without_consistency(self, rule_with_consistency: Rule) -> None:
        """Test rule_to_graph with consistency disabled."""
        graph = rule_to_graph(rule_with_consistency, include_consistency=False)

        # Overall status should still be set from rule
        assert graph.overall_status == "needs_review"

    def test_render_dot(self, simple_rule: Rule) -> None:
        """Test render_dot function."""
        graph = rule_to_graph(simple_rule)
        dot = render_dot(graph)

        assert isinstance(dot, str)
        assert "digraph" in dot

    def test_render_mermaid(self, simple_rule: Rule) -> None:
        """Test render_mermaid function."""
        graph = rule_to_graph(simple_rule)
        mermaid = render_mermaid(graph)

        assert isinstance(mermaid, str)
        assert "flowchart" in mermaid


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_leaf_only_tree(self) -> None:
        """Test rule with only a leaf node (no branches)."""
        rule = Rule(
            rule_id="leaf_only",
            decision_tree=DecisionLeaf(
                result="always_authorized",
                notes="Unconditional authorization",
            ),
        )

        graph = rule_to_graph(rule)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].node_type == "leaf"
        assert graph.nodes[0].decision == "always_authorized"
        assert len(graph.edges) == 0

    def test_branch_without_condition(self) -> None:
        """Test branch node without a condition."""
        rule = Rule(
            rule_id="no_condition",
            decision_tree=DecisionBranch(
                node_id="root",
                condition=None,
                true_branch=DecisionLeaf(result="yes"),
                false_branch=DecisionLeaf(result="no"),
            ),
        )

        graph = rule_to_graph(rule)
        root = graph.get_root()

        assert root is not None
        assert root.condition_field is None
        assert root.label == "root"

    def test_single_branch_path(self) -> None:
        """Test tree with only true branch."""
        rule = Rule(
            rule_id="single_branch",
            decision_tree=DecisionBranch(
                node_id="root",
                condition=ConditionSpec(
                    field="active",
                    operator="==",
                    value=True,
                ),
                true_branch=DecisionLeaf(result="active"),
                false_branch=None,
            ),
        )

        graph = rule_to_graph(rule)

        assert len(graph.nodes) == 2  # root + 1 leaf
        assert len(graph.edges) == 1  # only true edge

    def test_deeply_nested_tree(self) -> None:
        """Test deeply nested decision tree."""

        def make_nested(depth: int) -> DecisionBranch | DecisionLeaf:
            if depth == 0:
                return DecisionLeaf(result=f"depth_{depth}")
            return DecisionBranch(
                node_id=f"branch_{depth}",
                condition=ConditionSpec(
                    field=f"level_{depth}",
                    operator="==",
                    value=True,
                ),
                true_branch=make_nested(depth - 1),
                false_branch=DecisionLeaf(result=f"exit_{depth}"),
            )

        rule = Rule(
            rule_id="deep_tree",
            decision_tree=make_nested(5),
        )

        graph = rule_to_graph(rule)

        # 5 branches + 6 leaves = 11 nodes
        assert len(graph.nodes) == 11

        # Verify max depth
        max_depth = max(n.depth for n in graph.nodes)
        assert max_depth == 5

    def test_consistency_without_rule_element(self) -> None:
        """Test evidence without rule_element goes to __rule__."""
        rule = Rule(
            rule_id="no_element",
            decision_tree=DecisionLeaf(result="test"),
            consistency=ConsistencyBlock(
                summary=ConsistencySummary(
                    status=ConsistencyStatus.VERIFIED,
                    confidence=1.0,
                ),
                evidence=[
                    ConsistencyEvidence(
                        tier=0,
                        category="test",
                        label="pass",
                        score=1.0,
                        details="No rule element specified",
                        rule_element=None,
                    ),
                ],
            ),
        )

        adapter = TreeAdapter()
        node_map = adapter.build_node_consistency_map(rule)

        assert "__rule__" in node_map
        assert node_map["__rule__"].pass_count == 1
