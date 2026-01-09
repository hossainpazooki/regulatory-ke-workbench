"""Verification service - Consistency engine for rule verification."""

from .app.services.consistency_engine import ConsistencyEngine, verify_rule

__all__ = ["ConsistencyEngine", "verify_rule"]
