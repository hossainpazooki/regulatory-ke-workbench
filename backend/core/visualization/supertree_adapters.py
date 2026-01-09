"""Tree Data Adapters for Visualization.

This module provides pure-Python adapters that convert regulatory data structures
into nested dict/list format suitable for tree visualization (e.g., Supertree).

These adapters are independent of any visualization library and can be used
for testing or with any tree renderer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.rule_service.app.services.schema import Rule, DecisionBranch, DecisionLeaf
    from backend.rule_service.app.services.engine import TraceStep


def build_rulebook_outline(rules: list[Rule]) -> dict:
    """Build a tree structure representing the rulebook outline.

    Shows legal corpus documents with their articles, organized hierarchically.
    Each article shows the rules that implement it.

    Args:
        rules: List of Rule objects from the rule loader.

    Returns:
        Nested dict with legal corpus structure and rule mappings.
    """
    import re

    # Try to load legal corpus for rich document info
    legal_docs = []
    try:
        from backend.rag_service.app.services.corpus_loader import load_all_legal_documents
        legal_docs = list(load_all_legal_documents())
    except Exception:
        pass

    # Build rule coverage map: document_id -> article -> rules
    rule_coverage: dict[str, dict[str, list[Rule]]] = defaultdict(lambda: defaultdict(list))
    unlinked_rules: list[Rule] = []

    for rule in rules:
        if rule.source and rule.source.document_id:
            doc_id = rule.source.document_id
            article = rule.source.article or "General"
            # Normalize article number
            match = re.search(r"(\d+)", str(article))
            if match:
                article = match.group(1)
            rule_coverage[doc_id][article].append(rule)
        else:
            unlinked_rules.append(rule)

    doc_children = []

    # Process legal corpus documents first (with full metadata)
    for doc in legal_docs:
        doc_articles = _extract_articles_from_text(doc.text)
        doc_rules_map = rule_coverage.get(doc.document_id, {})

        article_children = []
        total_doc_rules = 0

        for article_num in sorted(doc_articles, key=lambda x: int(x) if x.isdigit() else 0):
            article_rules = doc_rules_map.get(article_num, [])
            total_doc_rules += len(article_rules)

            rule_nodes = [
                {
                    "title": r.rule_id,
                    "description": r.description or "",
                    "tags": r.tags,
                    "version": r.version,
                }
                for r in article_rules
            ]

            article_node = {
                "title": f"Article {article_num}",
                "count": len(article_rules),
                "status": "covered" if article_rules else "gap",
            }
            if rule_nodes:
                article_node["children"] = rule_nodes

            article_children.append(article_node)

        # Also add rules for articles not in the extracted list
        for article_num, article_rules in doc_rules_map.items():
            if article_num not in doc_articles and article_num != "General":
                total_doc_rules += len(article_rules)
                rule_nodes = [
                    {
                        "title": r.rule_id,
                        "description": r.description or "",
                        "tags": r.tags,
                    }
                    for r in article_rules
                ]
                article_children.append({
                    "title": f"Article {article_num}",
                    "count": len(article_rules),
                    "status": "covered",
                    "children": rule_nodes,
                })

        doc_children.append({
            "title": doc.title or doc.document_id,
            "document_id": doc.document_id,
            "citation": doc.citation,
            "jurisdiction": doc.jurisdiction,
            "articles": len(doc_articles),
            "rules": total_doc_rules,
            "children": article_children,
        })

    # Add any documents with rules but not in legal corpus
    for doc_id, articles_map in rule_coverage.items():
        if not any(d.document_id == doc_id for d in legal_docs):
            article_children = []
            total_rules = 0
            for article_num, article_rules in sorted(articles_map.items()):
                total_rules += len(article_rules)
                rule_nodes = [
                    {
                        "title": r.rule_id,
                        "description": r.description or "",
                        "tags": r.tags,
                    }
                    for r in article_rules
                ]
                article_children.append({
                    "title": f"Article {article_num}" if article_num != "General" else "General",
                    "count": len(article_rules),
                    "children": rule_nodes,
                })

            doc_children.append({
                "title": doc_id.replace("_", " ").title(),
                "document_id": doc_id,
                "rules": total_rules,
                "children": article_children,
            })

    # Add unlinked rules if any
    if unlinked_rules:
        unlinked_nodes = [
            {
                "title": r.rule_id,
                "description": r.description or "",
                "tags": r.tags,
            }
            for r in unlinked_rules
        ]
        doc_children.append({
            "title": "Unlinked Rules",
            "count": len(unlinked_rules),
            "children": unlinked_nodes,
        })

    return {
        "title": "Legal Corpus & Rulebook",
        "total_rules": len(rules),
        "documents": len(doc_children),
        "children": doc_children,
    }


def build_decision_trace_tree(
    trace: list[TraceStep],
    decision: str | None = None,
    rule_id: str | None = None,
) -> dict:
    """Build a tree structure from a decision trace.

    Converts the flat trace list into a hierarchical visualization
    showing the decision path taken through the rule.

    Args:
        trace: List of TraceStep objects from engine evaluation.
        decision: Final decision outcome (optional).
        rule_id: Rule ID that was evaluated (optional).

    Returns:
        Nested dict with structure:
        {
            "title": "Decision Trace",
            "decision": "authorized",
            "children": [
                {
                    "title": "node_id",
                    "condition": "field == value",
                    "result": true,
                    "value": "actual_value"
                }
            ]
        }
    """
    if not trace:
        return {
            "title": "Decision Trace",
            "rule_id": rule_id,
            "decision": decision,
            "children": [],
        }

    # Build trace nodes
    trace_nodes = []
    for step in trace:
        node = {
            "title": step.node,
            "condition": step.condition,
            "result": step.result,
            "result_label": "TRUE" if step.result else "FALSE",
        }
        if step.value_checked is not None:
            node["value_checked"] = step.value_checked
        trace_nodes.append(node)

    return {
        "title": "Decision Trace",
        "rule_id": rule_id,
        "decision": decision,
        "steps": len(trace),
        "children": trace_nodes,
    }


def build_ontology_tree() -> dict:
    """Build a tree structure representing the regulatory ontology.

    Shows the hierarchy of ontology types (actors, instruments, activities)
    with their enumeration values.

    Returns:
        Nested dict with structure:
        {
            "title": "Regulatory Ontology",
            "children": [
                {
                    "title": "Actor Types",
                    "children": [{"title": "issuer"}, ...]
                },
                ...
            ]
        }
    """
    # Import here to avoid circular dependencies
    from backend.core.ontology.types import ActorType, InstrumentType, ActivityType, ProvisionType

    def enum_to_children(enum_class) -> list[dict]:
        """Convert enum values to tree children."""
        return [{"title": e.value, "name": e.name} for e in enum_class]

    return {
        "title": "Regulatory Ontology",
        "children": [
            {
                "title": "Actor Types",
                "description": "Types of regulated entities",
                "children": enum_to_children(ActorType),
            },
            {
                "title": "Instrument Types",
                "description": "Types of crypto-assets and financial instruments",
                "children": enum_to_children(InstrumentType),
            },
            {
                "title": "Activity Types",
                "description": "Types of regulated activities",
                "children": enum_to_children(ActivityType),
            },
            {
                "title": "Provision Types",
                "description": "Types of legal provisions",
                "children": enum_to_children(ProvisionType),
            },
        ],
    }


def build_corpus_rule_links(rules: list[Rule]) -> dict:
    """Build a tree showing corpus-to-rule mappings.

    Maps source documents and articles to the rules that reference them,
    useful for traceability analysis. Includes legal corpus metadata.

    Args:
        rules: List of Rule objects from the rule loader.

    Returns:
        Nested dict with structure:
        {
            "title": "Corpus-Rule Links",
            "children": [
                {
                    "title": "mica_2023",
                    "document_title": "Markets in Crypto-Assets...",
                    "jurisdiction": "EU",
                    "citation": "Regulation (EU) 2023/1114",
                    "source_url": "https://...",
                    "children": [
                        {
                            "title": "Art. 36(1)",
                            "children": [{"title": "rule_id", ...}]
                        }
                    ]
                }
            ]
        }
    """
    if not rules:
        return {"title": "Corpus-Rule Links", "children": []}

    # Try to load legal corpus metadata
    legal_docs_metadata: dict[str, dict] = {}
    try:
        from backend.rag_service.app.services.corpus_loader import load_all_legal_documents
        for doc in load_all_legal_documents():
            legal_docs_metadata[doc.document_id] = {
                "document_title": doc.title,
                "citation": doc.citation,
                "jurisdiction": doc.jurisdiction,
                "source_url": doc.source_url,
            }
    except Exception:
        pass  # Legal corpus not available

    # Group rules by document -> article
    corpus_map: dict[str, dict[str, list[Rule]]] = defaultdict(lambda: defaultdict(list))

    for rule in rules:
        if rule.source:
            doc_id = rule.source.document_id
            article = rule.source.article or "General"
            corpus_map[doc_id][article].append(rule)
        else:
            corpus_map["unlinked"]["No Source"].append(rule)

    # Build tree structure
    doc_children = []
    for doc_id, articles in sorted(corpus_map.items()):
        article_children = []
        for article, article_rules in sorted(articles.items()):
            rule_nodes = [
                {
                    "title": r.rule_id,
                    "description": r.description or "",
                    "tags": r.tags,
                }
                for r in article_rules
            ]
            article_children.append({
                "title": f"Art. {article}" if article != "General" and article != "No Source" else article,
                "count": len(article_rules),
                "children": rule_nodes,
            })

        # Get legal corpus metadata for this document
        doc_meta = legal_docs_metadata.get(doc_id, {})
        doc_node = {
            "title": doc_id,
            "articles": len(articles),
            "rules": sum(len(ar) for ar in articles.values()),
            "children": article_children,
        }
        # Add legal corpus metadata if available
        if doc_meta:
            doc_node["document_title"] = doc_meta.get("document_title")
            doc_node["citation"] = doc_meta.get("citation")
            doc_node["jurisdiction"] = doc_meta.get("jurisdiction")
            doc_node["source_url"] = doc_meta.get("source_url")

        doc_children.append(doc_node)

    return {
        "title": "Corpus-Rule Links",
        "documents": len(corpus_map),
        "total_rules": len(rules),
        "children": doc_children,
    }


def build_legal_corpus_coverage(rules: list[Rule]) -> dict:
    """Build a tree showing legal corpus coverage status.

    Shows all legal documents in the corpus with their articles,
    indicating which articles have rules and which are gaps.

    Args:
        rules: List of Rule objects from the rule loader.

    Returns:
        Nested dict with coverage structure:
        {
            "title": "Legal Corpus Coverage",
            "children": [
                {
                    "title": "MiCA 2023",
                    "document_id": "mica_2023",
                    "jurisdiction": "EU",
                    "coverage": 0.75,
                    "covered_articles": 12,
                    "total_articles": 16,
                    "children": [
                        {
                            "title": "Art. 36",
                            "has_rules": true,
                            "rule_count": 3,
                            "status": "covered"
                        },
                        {
                            "title": "Art. 37",
                            "has_rules": false,
                            "rule_count": 0,
                            "status": "gap"
                        }
                    ]
                }
            ]
        }
    """
    import re

    # Try to load legal corpus
    legal_docs = []
    try:
        from backend.rag_service.app.services.corpus_loader import load_all_legal_documents
        legal_docs = list(load_all_legal_documents())
    except Exception:
        pass

    if not legal_docs:
        return {
            "title": "Legal Corpus Coverage",
            "children": [],
            "message": "No legal corpus available",
        }

    # Build rule coverage map: document_id -> article -> rule_ids
    rule_coverage: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for rule in rules:
        if rule.source and rule.source.document_id:
            doc_id = rule.source.document_id
            # Normalize article reference
            article = rule.source.article or "General"
            # Extract just the article number
            match = re.search(r"(\d+)", str(article))
            if match:
                article = match.group(1)
            rule_coverage[doc_id][article].append(rule.rule_id)

    # Build coverage tree
    doc_children = []
    total_covered = 0
    total_gaps = 0

    for doc in legal_docs:
        # Extract articles from document text
        doc_articles = _extract_articles_from_text(doc.text)
        doc_coverage = rule_coverage.get(doc.document_id, {})

        article_children = []
        covered_count = 0

        for article_num in sorted(doc_articles, key=lambda x: int(x) if x.isdigit() else 0):
            rules_for_article = doc_coverage.get(article_num, [])
            has_rules = len(rules_for_article) > 0

            if has_rules:
                covered_count += 1
                total_covered += 1
                status = "covered"
            else:
                total_gaps += 1
                status = "gap"

            article_children.append({
                "title": f"Art. {article_num}",
                "article": article_num,
                "has_rules": has_rules,
                "rule_count": len(rules_for_article),
                "rules": rules_for_article,
                "status": status,
            })

        # Calculate coverage percentage
        total_articles = len(doc_articles)
        coverage_pct = (covered_count / total_articles * 100) if total_articles > 0 else 0

        doc_children.append({
            "title": doc.title or doc.document_id,
            "document_id": doc.document_id,
            "jurisdiction": doc.jurisdiction,
            "citation": doc.citation,
            "source_url": doc.source_url,
            "coverage": round(coverage_pct, 1),
            "covered_articles": covered_count,
            "total_articles": total_articles,
            "gap_articles": total_articles - covered_count,
            "children": article_children,
        })

    return {
        "title": "Legal Corpus Coverage",
        "documents": len(legal_docs),
        "total_covered": total_covered,
        "total_gaps": total_gaps,
        "children": doc_children,
    }


def _extract_articles_from_text(text: str) -> list[str]:
    """Extract article numbers from legal document text.

    Args:
        text: Full document text.

    Returns:
        List of article numbers (as strings).
    """
    import re

    articles = set()

    # Match "Article N" or "Art. N" patterns
    for match in re.finditer(r"Article\s+(\d+)", text, re.IGNORECASE):
        articles.add(match.group(1))

    # Also match "SECTION N" for US-style documents
    for match in re.finditer(r"Section\s+(\d+)", text, re.IGNORECASE):
        articles.add(match.group(1))

    return list(articles)


def build_decision_tree_structure(node) -> dict | None:
    """Build a tree structure from a rule's decision tree.

    Converts the decision tree definition into a visualization-friendly format.
    Handles both schema.py types (DecisionBranch, DecisionLeaf) and
    loader.py types (DecisionNode, DecisionLeaf).

    Args:
        node: Root node of the decision tree.

    Returns:
        Nested dict representing the tree structure, or None if no tree.
    """
    if node is None:
        return None

    # Import here to avoid circular dependencies
    from backend.rule_service.app.services.schema import DecisionLeaf as SchemaLeaf, DecisionBranch
    from backend.rule_service.app.services.loader import DecisionLeaf as LoaderLeaf, DecisionNode

    # Handle leaf nodes (both schema and loader types)
    if isinstance(node, (SchemaLeaf, LoaderLeaf)) or hasattr(node, "result") and not hasattr(node, "node_id"):
        result = {
            "title": f"Result: {node.result}",
            "type": "leaf",
            "result": node.result,
        }
        if hasattr(node, "notes") and node.notes:
            result["notes"] = node.notes
        if hasattr(node, "obligations") and node.obligations:
            result["obligations"] = [
                {"id": o.id, "description": getattr(o, "description", None)}
                for o in node.obligations
            ]
        return result

    # Handle branch nodes (DecisionBranch from schema, DecisionNode from loader)
    if isinstance(node, (DecisionBranch, DecisionNode)) or hasattr(node, "node_id"):
        condition_str = ""
        if hasattr(node, "condition") and node.condition:
            condition_str = f"{node.condition.field} {node.condition.operator} {node.condition.value}"

        children = []
        if hasattr(node, "true_branch") and node.true_branch:
            true_child = build_decision_tree_structure(node.true_branch)
            if true_child:
                true_child["branch"] = "true"
                children.append(true_child)

        if hasattr(node, "false_branch") and node.false_branch:
            false_child = build_decision_tree_structure(node.false_branch)
            if false_child:
                false_child["branch"] = "false"
                children.append(false_child)

        return {
            "title": node.node_id,
            "type": "branch",
            "condition": condition_str,
            "children": children,
        }

    return None
