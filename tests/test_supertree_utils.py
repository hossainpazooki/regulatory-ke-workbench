"""Tests for tree visualization utilities.

These tests verify that the custom HTML tree rendering functions work correctly.
The visualization uses a custom HTML/CSS/JS renderer with collapsible <details> elements.
"""

import pytest

from backend.core.visualization.supertree_utils import (
    SUPERTREE_AVAILABLE,
    is_supertree_available,
    render_rulebook_outline_html,
    render_decision_trace_html,
    render_ontology_tree_html,
    render_corpus_links_html,
    _escape,
    _render_tree_node,
    _render_tree_html,
)


class TestAvailability:
    """Tests for visualization availability checks."""

    def test_is_supertree_available_returns_bool(self):
        """Test that is_supertree_available returns boolean."""
        result = is_supertree_available()
        assert isinstance(result, bool)

    def test_constant_matches_function(self):
        """Test that SUPERTREE_AVAILABLE matches function result."""
        assert SUPERTREE_AVAILABLE == is_supertree_available()

    def test_visualization_always_available(self):
        """Test that visualization is always available (no external deps)."""
        assert is_supertree_available() is True
        assert SUPERTREE_AVAILABLE is True


class TestEscapeFunction:
    """Tests for HTML escaping."""

    def test_escapes_html_entities(self):
        """Test that HTML entities are escaped."""
        assert _escape("<script>") == "&lt;script&gt;"
        assert _escape("&") == "&amp;"
        assert _escape('"') == "&quot;"

    def test_handles_none(self):
        """Test that None returns empty string."""
        assert _escape(None) == ""

    def test_converts_non_strings(self):
        """Test that non-strings are converted."""
        assert _escape(123) == "123"
        assert _escape(True) == "True"


class TestRenderTreeNode:
    """Tests for individual tree node rendering."""

    def test_renders_leaf_node(self):
        """Test rendering a leaf node."""
        node = {"title": "Leaf"}
        result = _render_tree_node(node)
        assert "Leaf" in result
        assert "tree-leaf" in result

    def test_renders_branch_node(self):
        """Test rendering a branch node with children."""
        node = {
            "title": "Branch",
            "children": [{"title": "Child"}]
        }
        result = _render_tree_node(node)
        assert "Branch" in result
        assert "Child" in result
        assert "<details" in result

    def test_renders_metadata(self):
        """Test that metadata is included in output."""
        node = {"title": "Node", "status": "active", "count": 5}
        result = _render_tree_node(node)
        assert "status" in result
        assert "active" in result
        assert "count" in result
        assert "5" in result


class TestRenderTreeHtml:
    """Tests for complete tree rendering."""

    def test_includes_styles(self):
        """Test that CSS styles are included."""
        tree = {"title": "Root"}
        result = _render_tree_html(tree, "Test Chart")
        assert "<style>" in result
        assert ".tree-container" in result

    def test_includes_chart_title(self):
        """Test that chart title is included."""
        tree = {"title": "Root"}
        result = _render_tree_html(tree, "My Chart Title")
        assert "My Chart Title" in result
        assert "tree-header" in result


class TestRenderFunctions:
    """Tests for high-level render functions."""

    @pytest.fixture
    def sample_tree_data(self):
        """Create sample tree data for testing."""
        return {
            "title": "Test Tree",
            "children": [
                {"title": "Child 1"},
                {"title": "Child 2", "children": [{"title": "Grandchild"}]},
            ]
        }

    def test_render_rulebook_outline_returns_html(self, sample_tree_data):
        """Test render_rulebook_outline_html returns HTML."""
        result = render_rulebook_outline_html(sample_tree_data)
        assert isinstance(result, str)
        assert "<div" in result
        assert "Test Tree" in result

    def test_render_decision_trace_returns_html(self, sample_tree_data):
        """Test render_decision_trace_html returns HTML."""
        result = render_decision_trace_html(sample_tree_data)
        assert isinstance(result, str)
        assert "<div" in result

    def test_render_ontology_tree_returns_html(self, sample_tree_data):
        """Test render_ontology_tree_html returns HTML."""
        result = render_ontology_tree_html(sample_tree_data)
        assert isinstance(result, str)
        assert "<div" in result

    def test_render_corpus_links_returns_html(self, sample_tree_data):
        """Test render_corpus_links_html returns HTML."""
        result = render_corpus_links_html(sample_tree_data)
        assert isinstance(result, str)
        assert "<div" in result


class TestRenderFunctionsWithEmptyData:
    """Tests for render functions with empty data."""

    def test_render_rulebook_with_empty_dict(self):
        """Test rendering with empty tree data."""
        result = render_rulebook_outline_html({})
        assert isinstance(result, str)
        assert "<div" in result

    def test_render_decision_trace_with_empty_dict(self):
        """Test rendering decision trace with empty data."""
        result = render_decision_trace_html({})
        assert isinstance(result, str)

    def test_render_ontology_with_empty_dict(self):
        """Test rendering ontology with empty data."""
        result = render_ontology_tree_html({})
        assert isinstance(result, str)

    def test_render_corpus_links_with_empty_dict(self):
        """Test rendering corpus links with empty data."""
        result = render_corpus_links_html({})
        assert isinstance(result, str)


class TestErrorHandling:
    """Tests for error handling in render functions."""

    def test_handles_invalid_data_gracefully(self):
        """Test that invalid data produces error message rather than crash."""
        # This should not raise, but produce an error message
        result = render_rulebook_outline_html(None)
        # Will either render or show error div
        assert isinstance(result, str)

    def test_handles_deeply_nested_tree(self):
        """Test rendering deeply nested tree structures."""
        tree = {"title": "Level 0"}
        current = tree
        for i in range(1, 20):
            child = {"title": f"Level {i}"}
            current["children"] = [child]
            current = child

        result = render_rulebook_outline_html(tree)
        assert isinstance(result, str)
        assert "Level 0" in result
        assert "Level 19" in result
