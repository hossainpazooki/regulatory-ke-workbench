"""Storage domain - database, repositories, and temporal stores."""

# Database
from backend.storage.database import (
    get_db,
    get_db_path,
    set_db_path,
    get_engine,
    reset_engine,
    init_db,
    reset_db,
    get_table_stats,
    seed_jurisdictions,
    init_db_with_seed,
)

# Migration
from backend.storage.migration import (
    migrate_yaml_rules,
    sync_rule_to_db,
    load_rules_from_db,
    extract_premise_keys,
    get_migration_status,
)

# Repositories
from backend.storage.repositories import (
    RuleRepository,
    VerificationRepository,
)

# Temporal Engine
from backend.storage.temporal import (
    RuleVersionRepository,
    RuleEventRepository,
    RuleVersionRecord,
    RuleEventRecord,
    RuleEventType,
)

# Retrieval Engine
from backend.storage.retrieval import (
    # Compiler - IR Types
    CompiledCheck,
    DecisionEntry,
    ObligationSpec,
    RuleIR,
    # Compiler - Functions
    RuleCompiler,
    compile_rule,
    compile_rules,
    # Compiler - Optimizer
    RuleOptimizer,
    optimize_rule,
    # Compiler - Index
    PremiseIndexBuilder,
    get_premise_index,
    # Runtime - Cache
    IRCache,
    get_ir_cache,
    reset_ir_cache,
    # Runtime - Trace
    TraceStep,
    ExecutionTrace,
    DecisionResult,
    # Runtime - Executor
    RuleRuntime,
    execute_rule,
)

# Stores
from backend.storage.stores import (
    EmbeddingStore,
    GraphStore,
    JurisdictionConfigRepository,
    EmbeddingRecord,
    EmbeddingType,
    GraphNode,
    GraphEdge,
    GraphQuery,
    GraphQueryResult,
)

__all__ = [
    # Database
    "get_db",
    "get_db_path",
    "set_db_path",
    "get_engine",
    "reset_engine",
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
    # Temporal Engine
    "RuleVersionRepository",
    "RuleEventRepository",
    "RuleVersionRecord",
    "RuleEventRecord",
    "RuleEventType",
    # Retrieval Engine - Compiler
    "RuleCompiler",
    "compile_rule",
    "compile_rules",
    "RuleOptimizer",
    "optimize_rule",
    "PremiseIndexBuilder",
    "get_premise_index",
    "RuleIR",
    "CompiledCheck",
    "DecisionEntry",
    "ObligationSpec",
    # Retrieval Engine - Runtime
    "RuleRuntime",
    "execute_rule",
    "IRCache",
    "get_ir_cache",
    "reset_ir_cache",
    "ExecutionTrace",
    "TraceStep",
    "DecisionResult",
    # Stores
    "EmbeddingStore",
    "GraphStore",
    "JurisdictionConfigRepository",
    "EmbeddingRecord",
    "EmbeddingType",
    "GraphNode",
    "GraphEdge",
    "GraphQuery",
    "GraphQueryResult",
]
