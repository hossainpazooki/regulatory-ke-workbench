"""
Tests for the persistence layer.

Tests database operations, rule repository, and verification repository.
"""

import pytest
import tempfile
from pathlib import Path

from backend.database_service.app.services.database import (
    init_db,
    get_db,
    set_db_path,
    get_table_stats,
    reset_db,
)
from backend.database_service.app.services.repositories.rule_repo import RuleRepository
from backend.database_service.app.services.repositories.verification_repo import VerificationRepository
from backend.database_service.app.services.migration import (
    migrate_yaml_rules,
    extract_premise_keys,
    load_rules_from_db,
    sync_rule_to_db,
    get_migration_status,
)
from backend.rule_service.app.services.loader import RuleLoader, ConditionGroupSpec, ConditionSpec


@pytest.fixture(autouse=True)
def temp_database():
    """Use a temporary database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)

    set_db_path(temp_path)
    init_db()
    yield temp_path

    # Cleanup
    try:
        temp_path.unlink()
    except Exception:
        pass


class TestDatabase:
    """Test database initialization and utilities."""

    def test_init_db_creates_tables(self, temp_database):
        """Test that init_db creates all required tables."""
        stats = get_table_stats()

        assert "rules" in stats
        assert "rule_premise_index" in stats
        assert "verification_results" in stats
        assert "verification_evidence" in stats
        assert "reviews" in stats

    def test_init_db_is_idempotent(self, temp_database):
        """Test that init_db can be called multiple times safely."""
        init_db()
        init_db()
        stats = get_table_stats()
        assert "rules" in stats


class TestRuleRepository:
    """Test rule repository operations."""

    def test_save_and_get_rule(self, temp_database):
        """Test saving and retrieving a rule."""
        repo = RuleRepository()

        yaml_content = """
rule_id: test_rule
applies_if:
  all:
    - field: instrument_type
      operator: "=="
      value: art
decision_tree:
  result: authorized
"""

        # Save
        record = repo.save_rule(
            rule_id="test_rule",
            content_yaml=yaml_content,
            source_document_id="test_doc",
            source_article="1(1)",
        )

        assert record.rule_id == "test_rule"
        assert record.source_document_id == "test_doc"

        # Retrieve
        retrieved = repo.get_rule("test_rule")
        assert retrieved is not None
        assert retrieved.rule_id == "test_rule"
        assert retrieved.content_yaml == yaml_content
        assert retrieved.content_json is not None  # Should be parsed

    def test_update_existing_rule(self, temp_database):
        """Test updating an existing rule."""
        repo = RuleRepository()

        # Create initial
        repo.save_rule(rule_id="update_test", content_yaml="rule_id: update_test")

        # Update
        new_yaml = "rule_id: update_test\nversion: 2"
        repo.save_rule(
            rule_id="update_test",
            content_yaml=new_yaml,
            source_article="2(1)",
        )

        # Verify update
        retrieved = repo.get_rule("update_test")
        assert retrieved is not None
        assert retrieved.content_yaml == new_yaml
        assert retrieved.source_article == "2(1)"

    def test_get_all_rules(self, temp_database):
        """Test getting all rules."""
        repo = RuleRepository()

        # Add multiple rules
        repo.save_rule(rule_id="rule_a", content_yaml="rule_id: rule_a")
        repo.save_rule(rule_id="rule_b", content_yaml="rule_id: rule_b")
        repo.save_rule(rule_id="rule_c", content_yaml="rule_id: rule_c")

        rules = repo.get_all_rules()
        assert len(rules) == 3
        rule_ids = [r.rule_id for r in rules]
        assert "rule_a" in rule_ids
        assert "rule_b" in rule_ids
        assert "rule_c" in rule_ids

    def test_delete_rule_soft(self, temp_database):
        """Test soft deletion of a rule."""
        repo = RuleRepository()

        repo.save_rule(rule_id="delete_test", content_yaml="rule_id: delete_test")
        assert repo.get_rule("delete_test") is not None

        # Soft delete
        result = repo.delete_rule("delete_test", soft=True)
        assert result is True

        # Should not be retrievable (active_only=True by default)
        assert repo.get_rule("delete_test") is None

        # Should still exist in DB with is_active=0
        all_rules = repo.get_all_rules(active_only=False)
        assert any(r.rule_id == "delete_test" for r in all_rules)

    def test_update_rule_ir(self, temp_database):
        """Test updating compiled IR for a rule."""
        repo = RuleRepository()

        repo.save_rule(rule_id="ir_test", content_yaml="rule_id: ir_test")

        # Update IR
        ir_json = '{"rule_id": "ir_test", "premise_keys": ["instrument_type:art"]}'
        result = repo.update_rule_ir("ir_test", ir_json)
        assert result is True

        # Retrieve IR
        retrieved_ir = repo.get_rule_ir("ir_test")
        assert retrieved_ir == ir_json

        # Full record should have compiled_at set
        record = repo.get_rule("ir_test")
        assert record.compiled_at is not None

    def test_premise_index_operations(self, temp_database):
        """Test premise index CRUD operations."""
        repo = RuleRepository()

        # Create rules
        repo.save_rule(rule_id="premise_rule_1", content_yaml="rule_id: premise_rule_1")
        repo.save_rule(rule_id="premise_rule_2", content_yaml="rule_id: premise_rule_2")
        repo.save_rule(rule_id="premise_rule_3", content_yaml="rule_id: premise_rule_3")

        # Add premise index entries
        repo.update_premise_index(
            "premise_rule_1",
            ["instrument_type:art", "activity:public_offer"],
        )
        repo.update_premise_index(
            "premise_rule_2",
            ["instrument_type:art", "jurisdiction:EU"],
        )
        repo.update_premise_index(
            "premise_rule_3",
            ["instrument_type:emt"],
        )

        # Query by single premise
        rules = repo.get_rules_by_premise("instrument_type:art")
        assert len(rules) == 2
        assert "premise_rule_1" in rules
        assert "premise_rule_2" in rules

        # Query by multiple premises (intersection)
        rules = repo.get_rules_by_premises(["instrument_type:art", "jurisdiction:EU"])
        assert len(rules) == 1
        assert "premise_rule_2" in rules

        # Get all premise keys
        all_keys = repo.get_all_premise_keys()
        assert "instrument_type:art" in all_keys
        assert "activity:public_offer" in all_keys
        assert "jurisdiction:EU" in all_keys
        assert "instrument_type:emt" in all_keys

    def test_get_rules_by_document(self, temp_database):
        """Test getting rules by document ID."""
        repo = RuleRepository()

        repo.save_rule(
            rule_id="mica_1",
            content_yaml="rule_id: mica_1",
            source_document_id="mica_2023",
        )
        repo.save_rule(
            rule_id="mica_2",
            content_yaml="rule_id: mica_2",
            source_document_id="mica_2023",
        )
        repo.save_rule(
            rule_id="rwa_1",
            content_yaml="rule_id: rwa_1",
            source_document_id="rwa_2025",
        )

        mica_rules = repo.get_rules_by_document("mica_2023")
        assert len(mica_rules) == 2
        assert all(r.source_document_id == "mica_2023" for r in mica_rules)

    def test_count_operations(self, temp_database):
        """Test count operations."""
        repo = RuleRepository()

        # Initially empty
        assert repo.count_rules() == 0
        assert repo.count_compiled_rules() == 0

        # Add rules
        repo.save_rule(rule_id="count_1", content_yaml="rule_id: count_1")
        repo.save_rule(rule_id="count_2", content_yaml="rule_id: count_2")

        assert repo.count_rules() == 2
        assert repo.count_compiled_rules() == 0

        # Compile one
        repo.update_rule_ir("count_1", '{"test": true}')
        assert repo.count_compiled_rules() == 1


class TestVerificationRepository:
    """Test verification repository operations."""

    def test_save_and_get_verification_result(self, temp_database):
        """Test saving and retrieving verification results."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        # Create a rule first
        rule_repo.save_rule(rule_id="verify_test", content_yaml="rule_id: verify_test")

        # Save verification result
        evidence = [
            {
                "tier": 0,
                "category": "schema_valid",
                "label": "pass",
                "score": 1.0,
                "details": "Rule schema is valid",
            },
            {
                "tier": 1,
                "category": "deontic_alignment",
                "label": "warning",
                "score": 0.7,
                "details": "Deontic modality unclear",
            },
        ]

        record = verify_repo.save_verification_result(
            rule_id="verify_test",
            status="needs_review",
            confidence=0.85,
            verified_by="system",
            evidence=evidence,
        )

        assert record.rule_id == "verify_test"
        assert record.status == "needs_review"
        assert record.confidence == 0.85

        # Retrieve
        result, ev_list = verify_repo.get_verification_result("verify_test")
        assert result is not None
        assert result.status == "needs_review"
        assert len(ev_list) == 2
        assert any(e.category == "schema_valid" for e in ev_list)
        assert any(e.category == "deontic_alignment" for e in ev_list)

    def test_update_verification_result(self, temp_database):
        """Test updating an existing verification result."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        rule_repo.save_rule(rule_id="update_verify", content_yaml="rule_id: update_verify")

        # Initial save
        verify_repo.save_verification_result(
            rule_id="update_verify",
            status="needs_review",
            confidence=0.5,
            evidence=[{"tier": 0, "category": "test", "label": "warning", "score": 0.5}],
        )

        # Update with new status
        verify_repo.save_verification_result(
            rule_id="update_verify",
            status="verified",
            confidence=0.95,
            evidence=[{"tier": 0, "category": "test", "label": "pass", "score": 1.0}],
        )

        # Verify update
        result, evidence = verify_repo.get_verification_result("update_verify")
        assert result.status == "verified"
        assert result.confidence == 0.95
        assert len(evidence) == 1  # Old evidence replaced
        assert evidence[0].label == "pass"

    def test_get_all_verification_results(self, temp_database):
        """Test getting all verification results."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        # Create rules and verifications
        for i in range(3):
            rule_id = f"all_verify_{i}"
            rule_repo.save_rule(rule_id=rule_id, content_yaml=f"rule_id: {rule_id}")
            verify_repo.save_verification_result(
                rule_id=rule_id,
                status="verified" if i % 2 == 0 else "needs_review",
                confidence=0.8 + i * 0.05,
            )

        all_results = verify_repo.get_all_verification_results()
        assert len(all_results) == 3

    def test_verification_stats(self, temp_database):
        """Test verification statistics."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        # Create rules and verifications with different statuses
        for status, count in [("verified", 3), ("needs_review", 2), ("inconsistent", 1)]:
            for i in range(count):
                rule_id = f"stats_{status}_{i}"
                rule_repo.save_rule(rule_id=rule_id, content_yaml=f"rule_id: {rule_id}")
                verify_repo.save_verification_result(rule_id=rule_id, status=status)

        stats = verify_repo.get_verification_stats()
        assert stats["verified"] == 3
        assert stats["needs_review"] == 2
        assert stats["inconsistent"] == 1

    def test_save_and_get_reviews(self, temp_database):
        """Test human review operations."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        rule_repo.save_rule(rule_id="review_test", content_yaml="rule_id: review_test")

        # Save reviews
        verify_repo.save_review(
            rule_id="review_test",
            reviewer_id="alice",
            decision="consistent",
            notes="Looks good",
        )
        verify_repo.save_review(
            rule_id="review_test",
            reviewer_id="bob",
            decision="inconsistent",
            notes="Needs work on deontic alignment",
        )

        # Get reviews for rule
        reviews = verify_repo.get_reviews_for_rule("review_test")
        assert len(reviews) == 2

        # Get latest review
        latest = verify_repo.get_latest_review("review_test")
        assert latest is not None
        assert latest.reviewer_id == "bob"  # Most recent

        # Get reviews by reviewer
        alice_reviews = verify_repo.get_reviews_by_reviewer("alice")
        assert len(alice_reviews) == 1
        assert alice_reviews[0].decision == "consistent"


class TestIntegration:
    """Integration tests for persistence layer."""

    def test_full_rule_lifecycle(self, temp_database):
        """Test complete rule lifecycle: create, verify, review, compile."""
        rule_repo = RuleRepository()
        verify_repo = VerificationRepository()

        # 1. Create rule
        yaml_content = """
rule_id: lifecycle_test
source:
  document_id: mica_2023
  article: "36"
applies_if:
  all:
    - field: instrument_type
      operator: in
      value: [art, stablecoin]
decision_tree:
  node_id: check_auth
  condition:
    field: authorized
    operator: "=="
    value: true
  true_branch:
    result: authorized
  false_branch:
    result: not_authorized
"""

        rule_repo.save_rule(
            rule_id="lifecycle_test",
            content_yaml=yaml_content,
            source_document_id="mica_2023",
            source_article="36",
        )

        # 2. Add premise index
        rule_repo.update_premise_index(
            "lifecycle_test",
            ["instrument_type:art", "instrument_type:stablecoin"],
        )

        # 3. Verify rule
        verify_repo.save_verification_result(
            rule_id="lifecycle_test",
            status="needs_review",
            confidence=0.75,
            verified_by="system",
            evidence=[
                {
                    "tier": 0,
                    "category": "schema_valid",
                    "label": "pass",
                    "score": 1.0,
                },
                {
                    "tier": 1,
                    "category": "deontic_alignment",
                    "label": "warning",
                    "score": 0.5,
                },
            ],
        )

        # 4. Human review
        verify_repo.save_review(
            rule_id="lifecycle_test",
            reviewer_id="expert_1",
            decision="consistent",
            notes="Verified against source text",
        )

        # 5. Update verification after review
        verify_repo.save_verification_result(
            rule_id="lifecycle_test",
            status="verified",
            confidence=0.95,
            verified_by="human:expert_1",
        )

        # 6. Compile to IR
        ir_json = """
{
  "rule_id": "lifecycle_test",
  "version": 1,
  "premise_keys": ["instrument_type:art", "instrument_type:stablecoin"],
  "applicability_checks": [
    {"index": 0, "field": "instrument_type", "op": "in", "value": ["art", "stablecoin"]}
  ],
  "decision_table": [
    {"entry_id": 0, "condition_mask": [0], "result": "authorized"},
    {"entry_id": 1, "condition_mask": [-1], "result": "not_authorized"}
  ]
}
"""
        rule_repo.update_rule_ir("lifecycle_test", ir_json)

        # Verify final state
        rule = rule_repo.get_rule("lifecycle_test")
        assert rule is not None
        assert rule.rule_ir is not None
        assert rule.compiled_at is not None

        result, evidence = verify_repo.get_verification_result("lifecycle_test")
        assert result.status == "verified"
        assert result.confidence == 0.95

        reviews = verify_repo.get_reviews_for_rule("lifecycle_test")
        assert len(reviews) == 1

        # O(1) lookup should work
        matching = rule_repo.get_rules_by_premise("instrument_type:art")
        assert "lifecycle_test" in matching


class TestMigration:
    """Test migration utilities."""

    def test_extract_premise_keys_simple(self, temp_database):
        """Test premise key extraction from simple conditions."""
        group = ConditionGroupSpec(
            all=[
                ConditionSpec(field="instrument_type", operator="==", value="art"),
                ConditionSpec(field="jurisdiction", operator="==", value="EU"),
            ]
        )

        keys = extract_premise_keys(group)
        assert "instrument_type:art" in keys
        assert "jurisdiction:EU" in keys
        assert len(keys) == 2

    def test_extract_premise_keys_with_in_operator(self, temp_database):
        """Test premise key extraction with 'in' operator."""
        group = ConditionGroupSpec(
            all=[
                ConditionSpec(
                    field="instrument_type", operator="in", value=["art", "emt", "stablecoin"]
                ),
            ]
        )

        keys = extract_premise_keys(group)
        assert "instrument_type:art" in keys
        assert "instrument_type:emt" in keys
        assert "instrument_type:stablecoin" in keys
        assert len(keys) == 3

    def test_extract_premise_keys_nested(self, temp_database):
        """Test premise key extraction from nested condition groups."""
        group = ConditionGroupSpec(
            all=[
                ConditionSpec(field="instrument_type", operator="==", value="art"),
                ConditionGroupSpec(
                    any=[
                        ConditionSpec(field="activity", operator="==", value="public_offer"),
                        ConditionSpec(field="activity", operator="==", value="admission"),
                    ]
                ),
            ]
        )

        keys = extract_premise_keys(group)
        assert "instrument_type:art" in keys
        assert "activity:public_offer" in keys
        assert "activity:admission" in keys

    def test_migrate_yaml_rules(self, temp_database):
        """Test migrating YAML rules to database."""
        rules_dir = Path(__file__).parent.parent / "backend" / "rule_service" / "data"

        result = migrate_yaml_rules(rules_dir, clear_existing=True)

        assert result["success"] is True
        assert result["rules_migrated"] > 0
        assert result["premise_keys_indexed"] > 0

        # Verify rules are in database
        repo = RuleRepository()
        all_rules = repo.get_all_rules()
        assert len(all_rules) >= result["rules_migrated"]

    def test_load_rules_from_db(self, temp_database):
        """Test loading rules from database as Rule objects."""
        # First migrate some rules
        rules_dir = Path(__file__).parent.parent / "backend" / "rule_service" / "data"
        migrate_yaml_rules(rules_dir, clear_existing=True)

        # Load back from DB
        rules = load_rules_from_db()

        assert len(rules) > 0
        for rule_id, rule in rules.items():
            assert rule.rule_id == rule_id
            assert rule.applies_if is not None or rule.decision_tree is not None

    def test_sync_rule_to_db(self, temp_database):
        """Test syncing a single rule to database."""
        loader = RuleLoader()
        rules_dir = Path(__file__).parent.parent / "backend" / "rule_service" / "data"
        rules = loader.load_directory(rules_dir)

        if rules:
            rule = rules[0]
            result = sync_rule_to_db(rule)
            assert result is True

            # Verify in database
            repo = RuleRepository()
            record = repo.get_rule(rule.rule_id)
            assert record is not None
            assert record.rule_id == rule.rule_id

    def test_get_migration_status(self, temp_database):
        """Test getting migration status."""
        # Migrate first
        rules_dir = Path(__file__).parent.parent / "backend" / "rule_service" / "data"
        migrate_yaml_rules(rules_dir, clear_existing=True)

        status = get_migration_status()

        assert "rules_count" in status
        assert "compiled_rules_count" in status
        assert "verification_stats" in status
        assert "premise_keys" in status
        assert status["rules_count"] > 0
