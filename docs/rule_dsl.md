# Rule DSL Specification

This document specifies the YAML-based Domain Specific Language (DSL) for encoding regulatory rules.

## Design Principles

1. **Human-readable**: Non-engineers (lawyers, compliance officers) can review rules
2. **Machine-executable**: Rules evaluate deterministically
3. **Traceable**: Every decision links to source legal text
4. **Versionable**: Rules have explicit versions and effective dates

## Rule Structure

```yaml
# Required
rule_id: string              # Unique identifier (snake_case)

# Optional metadata
version: string              # Semantic version (default: "1.0")
description: string          # Human-readable description
effective_from: date         # YYYY-MM-DD when rule becomes active
effective_to: date           # YYYY-MM-DD when rule expires
tags: [string]               # Classification tags

# Applicability (optional)
applies_if:
  all: [conditions]          # All must be true (AND)
  any: [conditions]          # Any must be true (OR)

# Decision logic (optional)
decision_tree:
  # Tree structure (see below)

# Source citation (required for production rules)
source:
  document_id: string
  article: string
  section: string
  pages: [int]

# Documentation
interpretation_notes: string  # Explanation of modeling choices
```

## Condition Syntax

### Simple Condition

```yaml
field: instrument_type       # Field name to check
operator: "=="               # Comparison operator
value: art                   # Expected value
```

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equals | `field: x, operator: "==", value: 5` |
| `!=` | Not equals | `field: x, operator: "!=", value: null` |
| `in` | Value in list | `field: type, operator: in, value: [a, b, c]` |
| `not_in` | Value not in list | `field: type, operator: not_in, value: [x, y]` |
| `>` | Greater than | `field: amount, operator: ">", value: 1000` |
| `<` | Less than | `field: amount, operator: "<", value: 100` |
| `>=` | Greater or equal | `field: count, operator: ">=", value: 10` |
| `<=` | Less or equal | `field: count, operator: "<=", value: 5` |
| `exists` | Field is not null | `field: email, operator: exists, value: true` |

### Condition Groups

**AND (all must be true):**

```yaml
applies_if:
  all:
    - field: instrument_type
      operator: in
      value: [art, stablecoin]
    - field: activity
      operator: "=="
      value: public_offer
    - field: jurisdiction
      operator: "=="
      value: EU
```

**OR (any must be true):**

```yaml
applies_if:
  any:
    - field: is_credit_institution
      operator: "=="
      value: true
    - field: is_e_money_institution
      operator: "=="
      value: true
```

**Nested (complex logic):**

```yaml
applies_if:
  all:
    - field: instrument_type
      operator: "=="
      value: emt
    - any:
        - field: is_credit_institution
          operator: "=="
          value: true
        - field: is_e_money_institution
          operator: "=="
          value: true
```

This represents: `instrument_type == emt AND (is_credit_institution OR is_e_money_institution)`

## Decision Tree

The decision tree encodes the logic for reaching a conclusion.

### Branch Node

```yaml
decision_tree:
  node_id: check_authorization    # Identifier for tracing
  condition:
    field: authorized
    operator: "=="
    value: true
  true_branch:                    # If condition is true
    # Another node or leaf
  false_branch:                   # If condition is false
    # Another node or leaf
```

### Leaf Node (Terminal)

```yaml
true_branch:
  result: authorized              # Decision outcome
  obligations: []                 # Triggered obligations (optional)
  notes: "Issuer is authorized"   # Explanation (optional)
```

### Obligations

Obligations are requirements triggered by a decision:

```yaml
false_branch:
  result: not_authorized
  obligations:
    - id: obtain_authorization_art21
      description: "Obtain authorization from competent authority per Article 21"
      deadline: "Before public offer"
    - id: submit_whitepaper_art6
      description: "Prepare and submit crypto-asset white paper per Article 6"
```

## Complete Example

```yaml
rule_id: mica_art36_public_offer_authorization
version: "1.0"
description: |
  Authorization requirement for public offers of asset-referenced tokens (ARTs)
  under MiCA Article 36.
effective_from: 2024-06-30
tags: [mica, authorization, art, public_offer]

applies_if:
  all:
    - field: instrument_type
      operator: in
      value: [art, stablecoin]
    - field: activity
      operator: "=="
      value: public_offer
    - field: jurisdiction
      operator: "=="
      value: EU

decision_tree:
  node_id: check_exemption
  condition:
    field: is_credit_institution
    operator: "=="
    value: true
  true_branch:
    result: exempt
    notes: "Credit institutions exempt per Art. 36(2)"
  false_branch:
    node_id: check_authorization
    condition:
      field: authorized
      operator: "=="
      value: true
    true_branch:
      result: authorized
      notes: "Issuer holds valid authorization"
    false_branch:
      result: not_authorized
      obligations:
        - id: obtain_authorization_art21
          description: "Obtain authorization from competent authority per Article 21"
        - id: submit_whitepaper_art6
          description: "Prepare and submit crypto-asset white paper per Article 6"

source:
  document_id: mica_2023
  article: "36(1)"
  pages: [65, 66]

interpretation_notes: |
  Article 36(1) requires authorization for public offers of ARTs in the EU.
  Credit institutions under Directive 2013/36/EU are exempt per Art. 36(2).
```

## Evaluation Semantics

### Applicability Check

1. If `applies_if` is absent, rule applies to all scenarios
2. If `applies_if` is present, evaluate conditions:
   - `all`: Short-circuit AND — stop on first false
   - `any`: Short-circuit OR — stop on first true
3. If applicability fails, return `{applicable: false, decision: "not_applicable"}`

### Decision Tree Traversal

1. Start at root node
2. At each branch node:
   - Evaluate condition against scenario
   - Follow `true_branch` or `false_branch`
3. At leaf node:
   - Return `result` as decision
   - Collect `obligations`
   - Record full trace

### Trace Generation

Every evaluation step is recorded:

```json
{
  "node": "check_exemption",
  "condition": "is_credit_institution == true",
  "result": false,
  "value_checked": false
}
```

## File Organization

Rules are stored in `backend/rules/`:

```
backend/rules/
├── schema.yaml              # This specification (reference)
├── mica_authorization.yaml  # Authorization rules
├── mica_stablecoin.yaml     # Stablecoin/reserve rules
└── ...
```

### Conventions

- One file per regulatory topic or article cluster
- Use descriptive `rule_id` values: `{regulation}_{article}_{topic}`
- Include `source` for all production rules
- Document interpretation choices in `interpretation_notes`

## Testing Rules

Rules can be tested by providing scenarios:

```python
from backend.core.ontology import Scenario
from backend.rule_service.app.services import RuleLoader, DecisionEngine

loader = RuleLoader("backend/rules")
loader.load_directory()
engine = DecisionEngine(loader)

scenario = Scenario(
    instrument_type="art",
    activity="public_offer",
    jurisdiction="EU",
    authorized=False,
    is_credit_institution=False,
)

result = engine.evaluate(scenario, "mica_art36_public_offer_authorization")
assert result.decision == "not_authorized"
assert len(result.obligations) == 2
```

## Limitations

- No arithmetic expressions (conditions only compare values)
- No cross-rule references (rules are independent)
- No temporal operators (use `effective_from`/`to` for versioning)
- Single-pass evaluation (no loops or recursion)

These limitations are intentional — they keep rules auditable and deterministic.




Changes Made
1. Enhanced "Corpus-Rule Links" Chart
Now displays legal corpus metadata (title, citation, jurisdiction, source URL)
Shows jurisdiction badges (e.g., [EU], [US])
Links to source documents when available
2. New "Legal Corpus Coverage" Chart
Added as a new chart type in the dropdown
Shows coverage analysis for each legal document:
Coverage percentage with progress bar
Covered vs total articles count
Lists of covered articles with their associated rules
Lists of gap articles (provisions with no rules)
Summary metrics at the top:
Total Legal Documents
Total Covered Articles
Total Coverage Gaps
3. Updated Visualization Functions
build_corpus_rule_links() - enhanced to include legal corpus metadata
build_legal_corpus_coverage() - new function for coverage analysis
render_legal_corpus_html() - new HTML renderer with coverage-specific styling (green checkmarks for covered, red warnings for gaps)
Files Modified
supertree_adapters.py - Added build_legal_corpus_coverage() and _extract_articles_from_text()
supertree_utils.py - Added render_legal_corpus_html() with coverage styling
visualization/init.py - Updated exports
charts.py - Added Legal Corpus Coverage chart and enhanced Corpus-Rule Links
You can test the new charts by running: streamlit run frontend/ke_dashboard.py and navigating to the Charts page.