"""Business logic for embedding rules."""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select

from .models import EmbeddingRule, EmbeddingCondition, EmbeddingDecision, EmbeddingLegalSource
from .schemas import RuleCreate, RuleUpdate


class EmbeddingRuleService:
    """Service for embedding rule CRUD operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_rule(self, rule_data: RuleCreate) -> EmbeddingRule:
        rule = EmbeddingRule(
            rule_id=rule_data.rule_id,
            name=rule_data.name,
            description=rule_data.description,
        )
        self.session.add(rule)
        self.session.flush()

        for cond_data in rule_data.conditions:
            condition = EmbeddingCondition(
                field=cond_data.field,
                operator=cond_data.operator,
                value=cond_data.value,
                description=cond_data.description,
                rule_id=rule.id,
            )
            self.session.add(condition)

        if rule_data.decision:
            decision = EmbeddingDecision(
                outcome=rule_data.decision.outcome,
                confidence=rule_data.decision.confidence,
                explanation=rule_data.decision.explanation,
                rule_id=rule.id,
            )
            self.session.add(decision)

        for source_data in rule_data.legal_sources:
            source = EmbeddingLegalSource(
                citation=source_data.citation,
                document_id=source_data.document_id,
                url=source_data.url,
                rule_id=rule.id,
            )
            self.session.add(source)

        self.session.commit()
        self.session.refresh(rule)
        return rule

    def get_rule_by_rule_id(self, rule_id: str) -> Optional[EmbeddingRule]:
        statement = select(EmbeddingRule).where(EmbeddingRule.rule_id == rule_id)
        return self.session.exec(statement).first()

    def get_rules(self, skip: int = 0, limit: int = 100) -> list[EmbeddingRule]:
        statement = select(EmbeddingRule).offset(skip).limit(limit).order_by(EmbeddingRule.created_at.desc())
        return list(self.session.exec(statement).all())

    def update_rule(self, rule_id: str, rule_data: RuleUpdate) -> Optional[EmbeddingRule]:
        rule = self.get_rule_by_rule_id(rule_id)
        if not rule:
            return None

        if rule_data.name is not None:
            rule.name = rule_data.name
        if rule_data.description is not None:
            rule.description = rule_data.description
        if rule_data.is_active is not None:
            rule.is_active = rule_data.is_active

        if rule_data.conditions is not None:
            for cond in rule.conditions:
                self.session.delete(cond)
            for cond_data in rule_data.conditions:
                condition = EmbeddingCondition(
                    field=cond_data.field,
                    operator=cond_data.operator,
                    value=cond_data.value,
                    description=cond_data.description,
                    rule_id=rule.id,
                )
                self.session.add(condition)

        if rule_data.decision is not None:
            if rule.decision:
                self.session.delete(rule.decision)
                self.session.flush()
            decision = EmbeddingDecision(
                outcome=rule_data.decision.outcome,
                confidence=rule_data.decision.confidence,
                explanation=rule_data.decision.explanation,
                rule_id=rule.id,
            )
            self.session.add(decision)

        if rule_data.legal_sources is not None:
            for source in rule.legal_sources:
                self.session.delete(source)
            for source_data in rule_data.legal_sources:
                source = EmbeddingLegalSource(
                    citation=source_data.citation,
                    document_id=source_data.document_id,
                    url=source_data.url,
                    rule_id=rule.id,
                )
                self.session.add(source)

        rule.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(rule)
        return rule

    def soft_delete_rule(self, rule_id: str) -> Optional[EmbeddingRule]:
        rule = self.get_rule_by_rule_id(rule_id)
        if not rule:
            return None

        rule.is_active = False
        rule.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(rule)
        return rule
