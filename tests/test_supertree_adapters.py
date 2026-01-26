"""Tests for Supertree data adapters.

These tests verify that the adapters produce valid tree structures
without depending on Supertree being installed.
"""

import pytest
from pathlib import Path

from backend.rules import (
    Rule,
    SourceRef,
    ConditionGroupSpec,
    ConditionSpec,
    DecisionBranch,
    DecisionLeaf,
    ObligationSpec,
    TraceStep,
)
from backend.core.visualization.supertree_adapters import (
    build_rulebook_outline,
    build_decision_trace_tree,
    build_ontology_tree,
    build_corpus_rule_links,
    build_decision_tree_structure,
)


@pytest.fixture
def sample_rules() -> list[Rule]:
    """Create sample rules for testing."""
    return [
        Rule(
            rule_id="test_rule_1",
            version="1.0",
            description="Test rule for authorization",
            tags=["authorization", "test"],
            source=SourceRef(
                document_id="test_doc",
                article="1(1)",
                pages=[1, 2],
            ),
        ),
        Rule(
            rule_id="test_rule_2",
            version="1.0",
            description="Test rule for disclosure",
            tags=["disclosure", "test"],
            source=SourceRef(
                document_id="test_doc",
                article="2(1)",
                pages=[3, 4],
            ),
        ),
        Rule(
            rule_id="other_rule",
            version="1.0",
            description="Rule from other document",
            tags=["custody"],
            source=SourceRef(
                document_id="other_doc",
                article="10",
            ),
        ),
    ]


@pytest.fixture
def sample_trace() -> list[TraceStep]:
    """Create sample trace steps for testing."""
    return [
        TraceStep(
            node="applicability.all[0]",
            condition="instrument_type in ['art', 'stablecoin']",
            result=True,
            value_checked="art",
        ),
        TraceStep(
            node="applicability.all[1]",
            condition="jurisdiction == EU",
            result=True,
            value_checked="EU",
        ),
        TraceStep(
            node="check_authorization",
            condition="authorized == true",
            result=False,
            value_checked=False,
        ),
    ]


@pytest.fixture
def sample_decision_tree() -> DecisionBranch:
    """Create a sample decision tree for testing."""
    return DecisionBranch(
        node_id="check_exemption",
        condition=ConditionSpec(
            field="is_credit_institution",
            operator="==",
            value=True,
        ),
        true_branch=DecisionLeaf(
            result="exempt",
            notes="Credit institutions are exempt",
        ),
        false_branch=DecisionBranch(
            node_id="check_authorization",
            condition=ConditionSpec(
                field="authorized",
                operator="==",
                value=True,
            ),
            true_branch=DecisionLeaf(
                result="authorized",
                notes="Issuer is authorized",
            ),
            false_branch=DecisionLeaf(
                result="not_authorized",
                obligations=[
                    ObligationSpec(
                        id="obtain_authorization",
                        description="Must obtain authorization",
                    ),
                ],
            ),
        ),
    )


class TestBuildRulebookOutline:
    """Tests for build_rulebook_outline function."""

    def test_empty_rules(self):
        """Test with empty rule list."""
        result = build_rulebook_outline([])
        assert result["title"] == "Legal Corpus & Rulebook"
        # May include legal corpus documents even with no rules
        assert isinstance(result["children"], list)

    def test_groups_by_document(self, sample_rules):
        """Test that rules are grouped by source document."""
        result = build_rulebook_outline(sample_rules)

        assert result["title"] == "Legal Corpus & Rulebook"
        assert result["total_rules"] == 3
        # Should include test_doc and other_doc (plus any legal corpus docs)
        doc_titles = [d.get("title", "") for d in result["children"]]
        assert any("test_doc" in t.lower() or "test doc" in t.lower() for t in doc_titles)

    def test_document_contains_rules(self, sample_rules):
        """Test that documents contain their rules."""
        result = build_rulebook_outline(sample_rules)

        # Find test_doc (may be titled differently due to formatting)
        test_doc = next(
            (d for d in result["children"] if "test_doc" in str(d).lower()),
            None
        )
        assert test_doc is not None
        # Check it has rules (in children or nested under articles)
        assert "children" in test_doc

    def test_rule_node_structure(self, sample_rules):
        """Test that rule nodes have expected structure."""
        result = build_rulebook_outline(sample_rules)

        # Find test_doc
        test_doc = next(
            (d for d in result["children"] if "test_doc" in str(d).lower()),
            None
        )
        assert test_doc is not None
        # Rules may be directly in children or nested under articles
        assert "children" in test_doc
        assert len(test_doc["children"]) > 0

    def test_unlinked_rules(self):
        """Test handling of rules without source."""
        rules = [
            Rule(
                rule_id="unlinked_rule",
                description="Rule without source",
            )
        ]
        result = build_rulebook_outline(rules)

        # Should have unlinked rules section
        unlinked = next(
            (d for d in result["children"] if "unlinked" in d.get("title", "").lower()),
            None
        )
        assert unlinked is not None
        assert unlinked["count"] == 1


class TestBuildDecisionTraceTree:
    """Tests for build_decision_trace_tree function."""

    def test_empty_trace(self):
        """Test with empty trace list."""
        result = build_decision_trace_tree([])

        assert result["title"] == "Decision Trace"
        assert result["children"] == []

    def test_trace_with_steps(self, sample_trace):
        """Test trace tree with steps."""
        result = build_decision_trace_tree(
            sample_trace,
            decision="not_authorized",
            rule_id="test_rule",
        )

        assert result["title"] == "Decision Trace"
        assert result["decision"] == "not_authorized"
        assert result["rule_id"] == "test_rule"
        assert result["steps"] == 3
        assert len(result["children"]) == 3

    def test_trace_step_structure(self, sample_trace):
        """Test that trace steps have expected structure."""
        result = build_decision_trace_tree(sample_trace)
        step = result["children"][0]

        assert "title" in step
        assert "condition" in step
        assert "result" in step
        assert "result_label" in step
        assert step["result_label"] == "TRUE"

    def test_false_result_label(self, sample_trace):
        """Test that false results have correct label."""
        result = build_decision_trace_tree(sample_trace)

        # Find the false step
        false_step = next(s for s in result["children"] if not s["result"])
        assert false_step["result_label"] == "FALSE"

    def test_value_checked_included(self, sample_trace):
        """Test that value_checked is included when present."""
        result = build_decision_trace_tree(sample_trace)
        step = result["children"][0]

        assert "value_checked" in step
        assert step["value_checked"] == "art"


class TestBuildOntologyTree:
    """Tests for build_ontology_tree function."""

    def test_returns_tree_structure(self):
        """Test that ontology tree has valid structure."""
        result = build_ontology_tree()

        assert result["title"] == "Regulatory Ontology"
        assert "children" in result
        assert len(result["children"]) > 0

    def test_contains_actor_types(self):
        """Test that actor types are included."""
        result = build_ontology_tree()

        actor_node = next(
            (c for c in result["children"] if "Actor" in c["title"]),
            None
        )
        assert actor_node is not None
        assert len(actor_node["children"]) > 0

    def test_contains_instrument_types(self):
        """Test that instrument types are included."""
        result = build_ontology_tree()

        instrument_node = next(
            (c for c in result["children"] if "Instrument" in c["title"]),
            None
        )
        assert instrument_node is not None
        assert len(instrument_node["children"]) > 0

    def test_contains_activity_types(self):
        """Test that activity types are included."""
        result = build_ontology_tree()

        activity_node = next(
            (c for c in result["children"] if "Activity" in c["title"]),
            None
        )
        assert activity_node is not None
        assert len(activity_node["children"]) > 0

    def test_enum_values_as_children(self):
        """Test that enum values become children with title and name."""
        result = build_ontology_tree()

        actor_node = next(c for c in result["children"] if "Actor" in c["title"])
        child = actor_node["children"][0]

        assert "title" in child
        assert "name" in child


class TestBuildCorpusRuleLinks:
    """Tests for build_corpus_rule_links function."""

    def test_empty_rules(self):
        """Test with empty rule list."""
        result = build_corpus_rule_links([])

        assert result["title"] == "Corpus-Rule Links"
        assert result["children"] == []

    def test_groups_by_document_and_article(self, sample_rules):
        """Test grouping by document and article."""
        result = build_corpus_rule_links(sample_rules)

        assert result["title"] == "Corpus-Rule Links"
        assert result["documents"] == 2
        assert result["total_rules"] == 3

    def test_document_structure(self, sample_rules):
        """Test document node structure."""
        result = build_corpus_rule_links(sample_rules)

        # Find test_doc
        test_doc = next(
            (d for d in result["children"] if d["title"] == "test_doc"),
            None
        )
        assert test_doc is not None
        assert "articles" in test_doc
        assert "rules" in test_doc
        assert test_doc["rules"] == 2

    def test_article_contains_rules(self, sample_rules):
        """Test that articles contain their rules."""
        result = build_corpus_rule_links(sample_rules)

        test_doc = next(d for d in result["children"] if d["title"] == "test_doc")
        article = test_doc["children"][0]

        assert "Art." in article["title"]
        assert article["count"] == 1
        assert len(article["children"]) == 1


class TestBuildDecisionTreeStructure:
    """Tests for build_decision_tree_structure function."""

    def test_none_input(self):
        """Test with None input."""
        result = build_decision_tree_structure(None)
        assert result is None

    def test_leaf_node(self):
        """Test rendering of leaf node."""
        leaf = DecisionLeaf(
            result="authorized",
            notes="Test note",
            obligations=[
                ObligationSpec(id="test_obl", description="Test obligation")
            ],
        )
        result = build_decision_tree_structure(leaf)

        assert result["type"] == "leaf"
        assert result["result"] == "authorized"
        assert result["notes"] == "Test note"
        assert len(result["obligations"]) == 1

    def test_branch_node(self, sample_decision_tree):
        """Test rendering of branch node."""
        result = build_decision_tree_structure(sample_decision_tree)

        assert result["title"] == "check_exemption"
        assert result["type"] == "branch"
        assert "condition" in result
        assert len(result["children"]) == 2

    def test_full_tree_structure(self, sample_decision_tree):
        """Test full tree structure is preserved."""
        result = build_decision_tree_structure(sample_decision_tree)

        # Check true branch
        true_branch = next(c for c in result["children"] if c["branch"] == "true")
        assert true_branch["type"] == "leaf"
        assert true_branch["result"] == "exempt"

        # Check false branch (nested branch)
        false_branch = next(c for c in result["children"] if c["branch"] == "false")
        assert false_branch["type"] == "branch"
        assert false_branch["title"] == "check_authorization"

    def test_condition_string_format(self, sample_decision_tree):
        """Test condition is formatted as string."""
        result = build_decision_tree_structure(sample_decision_tree)

        assert "is_credit_institution == True" in result["condition"]
