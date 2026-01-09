"""Database service - centralized data access middleware."""

from .database import (
    get_db,
    get_db_path,
    set_db_path,
    init_db,
    reset_db,
    get_table_stats,
    seed_jurisdictions,
    init_db_with_seed,
)

from .migration import (
    migrate_yaml_rules,
    sync_rule_to_db,
    load_rules_from_db,
    extract_premise_keys,
    get_migration_status,
)

from .repositories import RuleRepository, VerificationRepository

from .compiler import (
    RuleCompiler,
    compile_rule,
    compile_rules,
    PremiseIndexBuilder,
    get_premise_index,
    RuleIR,
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
)

from .runtime import (
    RuleRuntime,
    execute_rule,
    IRCache,
    get_ir_cache,
    reset_ir_cache,
    ExecutionTrace,
    TraceStep,
    DecisionResult,
)

__all__ = [
    # Database
    "get_db",
    "get_db_path",
    "set_db_path",
    "init_db",
    "reset_db",
    "get_table_stats",
    "seed_jurisdictions",
    "init_db_with_seed",
    # Migration
    "migrate_yaml_rules",
    "sync_rule_to_db",
    "load_rules_from_db",
    "extract_premise_keys",
    "get_migration_status",
    # Repositories
    "RuleRepository",
    "VerificationRepository",
    # Compiler
    "RuleCompiler",
    "compile_rule",
    "compile_rules",
    "PremiseIndexBuilder",
    "get_premise_index",
    "RuleIR",
    "CompiledCheck",
    "DecisionEntry",
    "ObligationSpec",
    # Runtime
    "RuleRuntime",
    "execute_rule",
    "IRCache",
    "get_ir_cache",
    "reset_ir_cache",
    "ExecutionTrace",
    "TraceStep",
    "DecisionResult",
]
