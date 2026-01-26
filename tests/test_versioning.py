"""
Tests for the temporal versioning system.

Tests version repository, event repository, and versioned rule service.
"""

import pytest
import tempfile
from pathlib import Path

from backend.storage.database import (
    init_db,
    set_db_path,
)
from backend.storage.temporal.version_repo import RuleVersionRepository
from backend.storage.temporal.event_repo import RuleEventRepository
from backend.core.models import RuleEventType


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


class TestRuleVersionRepository:
    """Test rule version repository operations."""

    def test_create_first_version(self, temp_database):
        """Test creating the first version of a rule."""
        repo = RuleVersionRepository()

        version = repo.create_version(
            rule_id="test_rule",
            content_yaml="rule_id: test_rule\ndescription: Test",
            created_by="test_user",
            jurisdiction_code="EU",
        )

        assert version.rule_id == "test_rule"
        assert version.version == 1
        assert version.created_by == "test_user"
        assert version.jurisdiction_code == "EU"
        assert version.content_hash is not None
        assert len(version.content_hash) == 16  # SHA256 truncated

    def test_create_subsequent_versions(self, temp_database):
        """Test creating multiple versions of a rule."""
        repo = RuleVersionRepository()

        # Create first version
        v1 = repo.create_version(
            rule_id="multi_version",
            content_yaml="version: 1",
        )
        assert v1.version == 1

        # Create second version
        v2 = repo.create_version(
            rule_id="multi_version",
            content_yaml="version: 2",
        )
        assert v2.version == 2

        # Create third version
        v3 = repo.create_version(
            rule_id="multi_version",
            content_yaml="version: 3",
        )
        assert v3.version == 3

        # Verify supersession chain
        v1_updated = repo.get_version("multi_version", 1)
        assert v1_updated.superseded_by == 2

        v2_updated = repo.get_version("multi_version", 2)
        assert v2_updated.superseded_by == 3

    def test_get_specific_version(self, temp_database):
        """Test retrieving a specific version."""
        repo = RuleVersionRepository()

        repo.create_version(rule_id="specific", content_yaml="v1")
        repo.create_version(rule_id="specific", content_yaml="v2")
        repo.create_version(rule_id="specific", content_yaml="v3")

        v2 = repo.get_version("specific", 2)
        assert v2 is not None
        assert v2.content_yaml == "v2"

    def test_get_latest_version(self, temp_database):
        """Test retrieving the latest version."""
        repo = RuleVersionRepository()

        repo.create_version(rule_id="latest", content_yaml="v1")
        repo.create_version(rule_id="latest", content_yaml="v2")
        repo.create_version(rule_id="latest", content_yaml="v3")

        latest = repo.get_latest_version("latest")
        assert latest is not None
        assert latest.version == 3
        assert latest.content_yaml == "v3"

    def test_get_version_history(self, temp_database):
        """Test retrieving version history."""
        repo = RuleVersionRepository()

        for i in range(5):
            repo.create_version(
                rule_id="history",
                content_yaml=f"version: {i+1}",
            )

        history = repo.get_version_history("history", limit=3)
        assert len(history) == 3
        # Should be newest first
        assert history[0].version == 5
        assert history[1].version == 4
        assert history[2].version == 3

    def test_get_version_nonexistent(self, temp_database):
        """Test getting a nonexistent version returns None."""
        repo = RuleVersionRepository()

        result = repo.get_version("nonexistent", 1)
        assert result is None

    def test_content_hash_uniqueness(self, temp_database):
        """Test that same content produces same hash."""
        repo = RuleVersionRepository()

        content = "identical: content"

        v1 = repo.create_version(rule_id="hash_test_1", content_yaml=content)
        v2 = repo.create_version(rule_id="hash_test_2", content_yaml=content)

        assert v1.content_hash == v2.content_hash

    def test_get_versions_by_hash(self, temp_database):
        """Test finding versions with same content hash."""
        repo = RuleVersionRepository()

        content = "shared: content"

        repo.create_version(rule_id="rule_a", content_yaml=content)
        repo.create_version(rule_id="rule_b", content_yaml=content)
        repo.create_version(rule_id="rule_c", content_yaml="different: content")

        # Get hash from first version
        v1 = repo.get_version("rule_a", 1)
        same_hash = repo.get_versions_by_hash(v1.content_hash)

        assert len(same_hash) == 2
        rule_ids = [v.rule_id for v in same_hash]
        assert "rule_a" in rule_ids
        assert "rule_b" in rule_ids

    def test_count_versions(self, temp_database):
        """Test counting versions."""
        repo = RuleVersionRepository()

        assert repo.count_versions() == 0

        repo.create_version(rule_id="count_a", content_yaml="v1")
        repo.create_version(rule_id="count_a", content_yaml="v2")
        repo.create_version(rule_id="count_b", content_yaml="v1")

        assert repo.count_versions() == 3
        assert repo.count_versions("count_a") == 2
        assert repo.count_versions("count_b") == 1


class TestRuleEventRepository:
    """Test rule event repository operations."""

    def test_append_event(self, temp_database):
        """Test appending an event."""
        repo = RuleEventRepository()

        event = repo.append_event(
            rule_id="test_rule",
            version=1,
            event_type=RuleEventType.RULE_CREATED,
            event_data={"content_hash": "abc123"},
            actor="test_user",
            reason="Initial creation",
        )

        assert event.rule_id == "test_rule"
        assert event.version == 1
        assert event.event_type == "RuleCreated"
        assert event.actor == "test_user"
        assert event.sequence_number == 1

    def test_sequence_numbers_increment(self, temp_database):
        """Test that sequence numbers auto-increment."""
        repo = RuleEventRepository()

        e1 = repo.append_event("rule_a", 1, RuleEventType.RULE_CREATED, {})
        e2 = repo.append_event("rule_b", 1, RuleEventType.RULE_CREATED, {})
        e3 = repo.append_event("rule_a", 2, RuleEventType.RULE_UPDATED, {})

        assert e1.sequence_number == 1
        assert e2.sequence_number == 2
        assert e3.sequence_number == 3

    def test_get_events_for_rule(self, temp_database):
        """Test getting events for a specific rule."""
        repo = RuleEventRepository()

        repo.append_event("rule_a", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("rule_a", 2, RuleEventType.RULE_UPDATED, {})
        repo.append_event("rule_b", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("rule_a", 3, RuleEventType.RULE_UPDATED, {})

        events = repo.get_events_for_rule("rule_a")
        assert len(events) == 3
        # Should be newest first
        assert events[0].version == 3
        assert events[1].version == 2

    def test_get_events_by_type(self, temp_database):
        """Test filtering events by type."""
        repo = RuleEventRepository()

        repo.append_event("rule_1", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("rule_2", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("rule_1", 2, RuleEventType.RULE_UPDATED, {})
        repo.append_event("rule_1", 2, RuleEventType.RULE_DEPRECATED, {})

        created_events = repo.get_events_by_type(RuleEventType.RULE_CREATED)
        assert len(created_events) == 2

        deprecated_events = repo.get_events_by_type(RuleEventType.RULE_DEPRECATED)
        assert len(deprecated_events) == 1

    def test_get_events_after_sequence(self, temp_database):
        """Test getting events after a sequence number for replay."""
        repo = RuleEventRepository()

        for i in range(5):
            repo.append_event(f"rule_{i}", 1, RuleEventType.RULE_CREATED, {})

        # Get events after sequence 2
        events = repo.get_events_after_sequence(2)
        assert len(events) == 3
        # Should be oldest first for replay
        assert events[0].sequence_number == 3
        assert events[1].sequence_number == 4
        assert events[2].sequence_number == 5

    def test_get_latest_event(self, temp_database):
        """Test getting the latest event for a rule."""
        repo = RuleEventRepository()

        repo.append_event("rule_x", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("rule_x", 2, RuleEventType.RULE_UPDATED, {})
        repo.append_event("rule_x", 3, RuleEventType.RULE_UPDATED, {})

        latest = repo.get_latest_event("rule_x")
        assert latest is not None
        assert latest.version == 3
        assert latest.event_type == "RuleUpdated"

    def test_event_summary(self, temp_database):
        """Test getting event summary by type."""
        repo = RuleEventRepository()

        repo.append_event("r1", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("r2", 1, RuleEventType.RULE_CREATED, {})
        repo.append_event("r1", 2, RuleEventType.RULE_UPDATED, {})
        repo.append_event("r1", 3, RuleEventType.RULE_UPDATED, {})
        repo.append_event("r1", 3, RuleEventType.RULE_DEPRECATED, {})

        summary = repo.get_event_summary()
        assert summary["RuleCreated"] == 2
        assert summary["RuleUpdated"] == 2
        assert summary["RuleDeprecated"] == 1


class TestVersioningIntegration:
    """Integration tests for versioning system."""

    def test_full_version_lifecycle(self, temp_database):
        """Test complete version lifecycle with events."""
        version_repo = RuleVersionRepository()
        event_repo = RuleEventRepository()

        # 1. Create initial version
        v1 = version_repo.create_version(
            rule_id="lifecycle",
            content_yaml="version: 1\nstatus: draft",
            created_by="author",
        )

        event_repo.append_event(
            rule_id="lifecycle",
            version=v1.version,
            event_type=RuleEventType.RULE_CREATED,
            event_data={"hash": v1.content_hash},
            actor="author",
            reason="Initial draft",
        )

        # 2. Update rule
        v2 = version_repo.create_version(
            rule_id="lifecycle",
            content_yaml="version: 2\nstatus: review",
            created_by="author",
        )

        event_repo.append_event(
            rule_id="lifecycle",
            version=v2.version,
            event_type=RuleEventType.RULE_UPDATED,
            event_data={"prev_hash": v1.content_hash, "new_hash": v2.content_hash},
            actor="author",
            reason="Ready for review",
        )

        # 3. Final version
        v3 = version_repo.create_version(
            rule_id="lifecycle",
            content_yaml="version: 3\nstatus: approved",
            created_by="reviewer",
        )

        event_repo.append_event(
            rule_id="lifecycle",
            version=v3.version,
            event_type=RuleEventType.RULE_UPDATED,
            event_data={"prev_hash": v2.content_hash, "new_hash": v3.content_hash},
            actor="reviewer",
            reason="Approved after review",
        )

        # Verify version history
        history = version_repo.get_version_history("lifecycle")
        assert len(history) == 3
        assert history[0].version == 3  # Latest first

        # Verify event log
        events = event_repo.get_events_for_rule("lifecycle")
        assert len(events) == 3
        assert events[0].actor == "reviewer"  # Most recent first

        # Verify supersession chain
        v1_check = version_repo.get_version("lifecycle", 1)
        assert v1_check.superseded_by == 2

        v2_check = version_repo.get_version("lifecycle", 2)
        assert v2_check.superseded_by == 3

        v3_check = version_repo.get_version("lifecycle", 3)
        assert v3_check.superseded_by is None  # Latest version

    def test_point_in_time_query(self, temp_database):
        """Test querying versions at specific timestamps."""
        version_repo = RuleVersionRepository()

        # Create versions at different "times" using effective_from
        v1 = version_repo.create_version(
            rule_id="time_test",
            content_yaml="v1",
            effective_from="2024-01-01",
        )

        v2 = version_repo.create_version(
            rule_id="time_test",
            content_yaml="v2",
            effective_from="2024-06-01",
        )

        # Query before first version
        before = version_repo.get_version_at_timestamp("time_test", "2023-12-01")
        assert before is None

        # Query during v1
        during_v1 = version_repo.get_version_at_timestamp("time_test", "2024-03-15")
        assert during_v1 is not None
        assert during_v1.version == 1

        # Query during v2
        during_v2 = version_repo.get_version_at_timestamp("time_test", "2024-08-01")
        assert during_v2 is not None
        assert during_v2.version == 2
