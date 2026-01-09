"""Pydantic schemas for embedding rule API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ConditionCreate(BaseModel):
    field: str
    operator: str
    value: str
    description: Optional[str] = None


class ConditionRead(BaseModel):
    id: int
    field: str
    operator: str
    value: str
    description: Optional[str]

    model_config = {"from_attributes": True}


class DecisionCreate(BaseModel):
    outcome: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: Optional[str] = None


class DecisionRead(BaseModel):
    id: int
    outcome: str
    confidence: float
    explanation: Optional[str]

    model_config = {"from_attributes": True}


class LegalSourceCreate(BaseModel):
    citation: str
    document_id: Optional[str] = None
    url: Optional[str] = None


class LegalSourceRead(BaseModel):
    id: int
    citation: str
    document_id: Optional[str]
    url: Optional[str]

    model_config = {"from_attributes": True}


class RuleCreate(BaseModel):
    rule_id: str
    name: str
    description: Optional[str] = None
    conditions: list[ConditionCreate] = Field(default_factory=list)
    decision: Optional[DecisionCreate] = None
    legal_sources: list[LegalSourceCreate] = Field(default_factory=list)


class RuleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    conditions: Optional[list[ConditionCreate]] = None
    decision: Optional[DecisionCreate] = None
    legal_sources: Optional[list[LegalSourceCreate]] = None
    is_active: Optional[bool] = None


class RuleRead(BaseModel):
    id: int
    rule_id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    conditions: list[ConditionRead]
    decision: Optional[DecisionRead]
    legal_sources: list[LegalSourceRead]

    model_config = {"from_attributes": True}


class RuleList(BaseModel):
    id: int
    rule_id: str
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
