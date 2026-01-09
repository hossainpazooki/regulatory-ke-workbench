"""Verification service business logic."""

from .consistency_engine import ConsistencyEngine, verify_rule

__all__ = ["ConsistencyEngine", "verify_rule"]
