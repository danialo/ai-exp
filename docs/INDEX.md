# Documentation Index

Complete documentation for Astra AI experience system.

## Quick Start

- **[Main README](../README.md)** - Project overview and setup
- **[MCP Quick Start](MCP_QUICKSTART.md)** - Get MCP server running in 5 minutes
- **[Operations Guide](OPERATIONS.md)** - Running Astra in production

## MCP Server (New)

Model Context Protocol server for autonomous operation and introspection.

- **[MCP Quick Start](MCP_QUICKSTART.md)** - Setup and basic usage
- **[MCP Tools Reference](MCP_TOOLS_REFERENCE.md)** - Complete tool documentation with examples
- **[MCP Architecture](MCP_ARCHITECTURE.md)** - Design philosophy, persistence, safety model
- **[MCP Implementation](MCP_AUTONOMOUS_SCHEDULING_IMPLEMENTATION.md)** - Full implementation details
- **[Schedule Safety Tiers](SCHEDULE_SAFETY_TIERS.md)** - Safety and budget system

**Status**: 67/67 tests passing, production-ready for Tier 0/1

## Core Systems

### Memory & RAW Store
- **[Experience Schema](experience_schema.md)** - Data model and architecture
- **[MVP Build Plan](mvp_build_plan.md)** - Original memory system implementation

### Awareness & Introspection
- **[Awareness Loop](AWARENESS_LOOP_IMPLEMENTATION.md)** - Four-tier continuous presence system
- **[Introspection System](INTROSPECTION_SYSTEM.md)** - Context-rich self-reflection
- **[Awareness Test Report](AWARENESS_TEST_REPORT.md)** - Performance benchmarks

### Beliefs & Identity
- **[Belief Memory System](BELIEF_MEMORY_SYSTEM_IMPLEMENTATION.md)** - Belief vector store and consistency
- **[Belief System Status](BELIEF_SYSTEM_STATUS.md)** - Implementation status
- **[Belief Gardener](AUTONOMOUS_AGENT_ARCHITECTURE_ANALYSIS.md#belief-gardener)** - Autonomous pattern detection

### Goal Execution & Planning
- **[Goal Store Usage](GOAL_STORE_USAGE.md)** - Goal management and prioritization
- **[Task Graph Usage](TASK_GRAPH_USAGE.md)** - Dependency tracking and execution
- **[Scheduled Tasks](SCHEDULED_TASKS.md)** - Autonomous task scheduling

### Multi-Agent Architecture
- **[Coder Agent Plan](CODER_AGENT_IMPLEMENTATION_PLAN.md)** - Specialized coding agent
- **[System Architecture](SYSTEM_ARCHITECTURE.md)** - Overall system design

## Advanced Topics

### Adaptive Systems
- **[Adaptive Decision Framework](ADAPTIVE_DECISION_FRAMEWORK.md)** - Learning from outcomes
- **[Adaptive Framework Integration](ADAPTIVE_FRAMEWORK_INTEGRATION.md)** - Integration guide
- **[Outcome-Driven Trust](OUTCOME_DRIVEN_TRUST_SYSTEM.md)** - Provenance weighting

### Implementation Plans
- **[Phase 0 Plan](PHASE0_IMPLEMENTATION_PLAN.md)** - Foundation implementation
- **[Phase 0 Completion](PHASE0_GROUP1-3_COMPLETION.md)** - Completion status
- **[Phase 2 Plan](PHASE2_IMPLEMENTATION_PLAN.md)** - Task graph design
- **[Phase 2 Task Graph](PHASE2_TASKGRAPH_DESIGN.md)** - Detailed design

### Operations
- **[Database Migration](DATABASE_MIGRATION_PHASE1.md)** - Schema changes
- **[HTTPS Setup](HTTPS_SETUP.md)** - SSL configuration
- **[Regression Test Checklist](REGRESSION_TEST_CHECKLIST.md)** - Testing guide

## By Feature Area

### For Users
- [Main README](../README.md) - Getting started
- [Operations Guide](OPERATIONS.md) - Running the system
- [MCP Quick Start](MCP_QUICKSTART.md) - Using MCP tools

### For Developers
- [System Architecture](SYSTEM_ARCHITECTURE.md) - Overall design
- [MCP Architecture](MCP_ARCHITECTURE.md) - MCP server design
- [Experience Schema](experience_schema.md) - Data model
- [Adaptive Framework](ADAPTIVE_DECISION_FRAMEWORK.md) - Learning systems

### For Operations
- [HTTPS Setup](HTTPS_SETUP.md) - Production configuration
- [Database Migration](DATABASE_MIGRATION_PHASE1.md) - Schema management
- [Awareness Test Report](AWARENESS_TEST_REPORT.md) - Performance benchmarks

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| MCP Quick Start | ✅ Complete | 2025-11-12 |
| MCP Tools Reference | ✅ Complete | 2025-11-12 |
| MCP Architecture | ✅ Complete | 2025-11-12 |
| MCP Implementation | ✅ Complete | 2025-11-12 |
| Safety Tiers | ✅ Complete | 2025-11-12 |
| Awareness Loop | ✅ Complete | 2025-11-10 |
| Introspection System | ✅ Complete | 2025-11-10 |
| Belief Memory System | ✅ Complete | 2025-11-09 |
| Goal Store Usage | ✅ Complete | 2025-11-08 |
| Task Graph Usage | ✅ Complete | 2025-11-08 |

## Contributing

When adding new documentation:
1. Add entry to this index
2. Follow existing format (Markdown, clear headers)
3. Include code examples where applicable
4. Update document status table
5. Cross-reference related docs

## Need Help?

- Check [Operations Guide](OPERATIONS.md) for common tasks
- See [MCP Quick Start](MCP_QUICKSTART.md) for MCP setup
- Review [System Architecture](SYSTEM_ARCHITECTURE.md) for design questions
