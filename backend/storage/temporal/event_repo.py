"""
Rule event repository for event sourcing operations.

Provides append-only operations for rule lifecycle events.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from backend.storage.database import get_db
from backend.core.models import (
    RuleEventRecord,
    RuleEventType,
    generate_uuid,
    now_iso,
)


class RuleEventRepository:
    """Repository for rule event persistence operations.

    Events are append-only and form an audit log of all rule changes.
    """

    # =========================================================================
    # Event Operations
    # =========================================================================

    def append_event(
        self,
        rule_id: str,
        version: int,
        event_type: str | RuleEventType,
        event_data: dict[str, Any],
        actor: str | None = None,
        reason: str | None = None,
    ) -> RuleEventRecord:
        """Append a new event to the event log.

        Args:
            rule_id: The rule identifier
            version: The rule version this event relates to
            event_type: Type of event (RuleCreated, RuleUpdated, RuleDeprecated)
            event_data: Event payload as dictionary
            actor: Who triggered the event
            reason: Why the change was made

        Returns:
            The created RuleEventRecord
        """
        # Convert enum to string if needed
        if isinstance(event_type, RuleEventType):
            event_type = event_type.value

        # Get next sequence number
        sequence_number = self.get_next_sequence_number()

        # Serialize event data
        event_data_json = json.dumps(event_data)

        record = RuleEventRecord(
            rule_id=rule_id,
            version=version,
            event_type=event_type,
            event_data=event_data_json,
            sequence_number=sequence_number,
            actor=actor,
            reason=reason,
        )

        with get_db() as conn:
            conn.execute(
                text("""
                INSERT INTO rule_events (
                    id, sequence_number, rule_id, version,
                    event_type, event_data, timestamp, actor, reason
                ) VALUES (:id, :sequence_number, :rule_id, :version,
                          :event_type, :event_data, :timestamp, :actor, :reason)
                """),
                {
                    "id": record.id,
                    "sequence_number": record.sequence_number,
                    "rule_id": record.rule_id,
                    "version": record.version,
                    "event_type": record.event_type,
                    "event_data": record.event_data,
                    "timestamp": record.timestamp,
                    "actor": record.actor,
                    "reason": record.reason,
                },
            )
            conn.commit()

        return record

    def get_events_for_rule(
        self,
        rule_id: str,
        limit: int | None = None,
    ) -> list[RuleEventRecord]:
        """Get all events for a rule.

        Args:
            rule_id: The rule identifier
            limit: Maximum number of events to return (newest first)

        Returns:
            List of RuleEventRecord objects, newest first
        """
        with get_db() as conn:
            if limit:
                result = conn.execute(
                    text("""
                    SELECT * FROM rule_events
                    WHERE rule_id = :rule_id
                    ORDER BY sequence_number DESC
                    LIMIT :limit
                    """),
                    {"rule_id": rule_id, "limit": limit},
                )
            else:
                result = conn.execute(
                    text("""
                    SELECT * FROM rule_events
                    WHERE rule_id = :rule_id
                    ORDER BY sequence_number DESC
                    """),
                    {"rule_id": rule_id},
                )

            return [RuleEventRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_events_by_type(
        self,
        event_type: str | RuleEventType,
        limit: int = 100,
    ) -> list[RuleEventRecord]:
        """Get all events of a specific type.

        Args:
            event_type: The event type to filter by
            limit: Maximum number of events to return

        Returns:
            List of RuleEventRecord objects, newest first
        """
        if isinstance(event_type, RuleEventType):
            event_type = event_type.value

        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_events
                WHERE event_type = :event_type
                ORDER BY sequence_number DESC
                LIMIT :limit
                """),
                {"event_type": event_type, "limit": limit},
            )
            return [RuleEventRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_events_by_actor(
        self,
        actor: str,
        limit: int = 100,
    ) -> list[RuleEventRecord]:
        """Get all events by a specific actor.

        Args:
            actor: The actor to filter by
            limit: Maximum number of events to return

        Returns:
            List of RuleEventRecord objects, newest first
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_events
                WHERE actor = :actor
                ORDER BY sequence_number DESC
                LIMIT :limit
                """),
                {"actor": actor, "limit": limit},
            )
            return [RuleEventRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_events_after_sequence(
        self,
        sequence_number: int,
        limit: int = 100,
    ) -> list[RuleEventRecord]:
        """Get all events after a specific sequence number.

        Useful for event replay and synchronization.

        Args:
            sequence_number: The sequence number to start from (exclusive)
            limit: Maximum number of events to return

        Returns:
            List of RuleEventRecord objects, oldest first
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_events
                WHERE sequence_number > :sequence_number
                ORDER BY sequence_number ASC
                LIMIT :limit
                """),
                {"sequence_number": sequence_number, "limit": limit},
            )
            return [RuleEventRecord.from_row(row._mapping) for row in result.fetchall()]

    def get_next_sequence_number(self) -> int:
        """Get the next sequence number for events.

        Returns:
            The next sequence number (max + 1)
        """
        with get_db() as conn:
            result = conn.execute(
                text("SELECT MAX(sequence_number) as max_seq FROM rule_events")
            )
            row = result.fetchone()
            return (row[0] or 0) + 1

    def get_latest_event(self, rule_id: str) -> RuleEventRecord | None:
        """Get the most recent event for a rule.

        Args:
            rule_id: The rule identifier

        Returns:
            RuleEventRecord if found, None otherwise
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT * FROM rule_events
                WHERE rule_id = :rule_id
                ORDER BY sequence_number DESC
                LIMIT 1
                """),
                {"rule_id": rule_id},
            )
            row = result.fetchone()

            if row:
                return RuleEventRecord.from_row(row._mapping)
            return None

    def count_events(self, rule_id: str | None = None) -> int:
        """Count events.

        Args:
            rule_id: If provided, count only for this rule

        Returns:
            Number of events
        """
        with get_db() as conn:
            if rule_id:
                result = conn.execute(
                    text("SELECT COUNT(*) as count FROM rule_events WHERE rule_id = :rule_id"),
                    {"rule_id": rule_id},
                )
            else:
                result = conn.execute(text("SELECT COUNT(*) as count FROM rule_events"))

            return result.fetchone()[0]

    def get_event_summary(self) -> dict[str, int]:
        """Get a summary of events by type.

        Returns:
            Dictionary mapping event type to count
        """
        with get_db() as conn:
            result = conn.execute(
                text("""
                SELECT event_type, COUNT(*) as count
                FROM rule_events
                GROUP BY event_type
                ORDER BY event_type
                """)
            )
            return {row[0]: row[1] for row in result.fetchall()}
