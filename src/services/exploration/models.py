"""Data models for autonomous exploration loop."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class OracleType(str, Enum):
    """Type of acceptance test oracle."""
    SCRIPT = "script"
    PYTHON = "python"
    LLM = "llm"


class JobState(str, Enum):
    """State of an exploration job."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class OracleConfig(BaseModel):
    """Configuration for the acceptance test oracle."""
    type: OracleType
    cmd: Optional[str] = None  # Required if type=script
    callable: Optional[str] = None  # Required if type=python (dotted path)
    rubric: Optional[List[str]] = None  # Required if type=llm
    max_tokens: Optional[int] = 1000  # For LLM oracle
    timeout_sec: int = Field(default=20, ge=1, le=300)


class OracleResult(BaseModel):
    """Result from running the oracle."""
    ok: bool
    detail: Dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    duration_ms: int = 0


class WorkspacePolicy(BaseModel):
    """File system permissions for the job workspace."""
    base_dir: str = "persona_space/workspaces"
    allow_write_outside: bool = False
    extra_write_allowlist: List[str] = Field(default_factory=list)


class JobCapabilities(BaseModel):
    """Capabilities enabled for the job."""
    allow_net: bool = False
    allow_subprocess: bool = True


class JobLimits(BaseModel):
    """Resource limits for the exploration job."""
    max_iterations: int = Field(default=12, ge=1, le=50)
    token_start: int = Field(default=3000, ge=1000, le=10000)
    token_growth: float = Field(default=1.6, ge=1.0, le=3.0)
    token_attempt_cap: int = Field(default=24000, ge=5000, le=100000)
    token_global_cap: int = Field(default=180000, ge=10000, le=500000)


class JobMetrics(BaseModel):
    """Performance metrics for a single iteration."""
    plan_time_ms: int = 0
    exec_time_ms: int = 0
    check_time_ms: int = 0
    reflect_time_ms: int = 0


class ExplorationJobCreate(BaseModel):
    """Request to create an exploration job."""
    task: str = Field(..., min_length=10, max_length=5000)
    oracle: OracleConfig
    limits: JobLimits = Field(default_factory=JobLimits)
    workspace_policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    capabilities: JobCapabilities = Field(default_factory=JobCapabilities)


class ExplorationJob(BaseModel):
    """A running or completed exploration job."""
    job_id: str
    task: str
    oracle: OracleConfig
    limits: JobLimits
    workspace_policy: WorkspacePolicy
    capabilities: JobCapabilities

    # Runtime state
    state: JobState = JobState.QUEUED
    iteration: int = 0
    tokens_spent: int = 0

    # Oracle tracking
    oracle_checks: int = 0
    last_oracle_result: Optional[OracleResult] = None

    # Artifacts and logs
    logs_tail: str = ""
    artifacts: List[str] = Field(default_factory=list)
    top_files_touched: List[str] = Field(default_factory=list)

    # Metrics
    metrics: JobMetrics = Field(default_factory=JobMetrics)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error tracking
    error_message: Optional[str] = None
    error_histogram: Dict[str, int] = Field(default_factory=dict)


class ExplorationJobStatus(BaseModel):
    """Status response for a job query."""
    job_id: str
    state: JobState
    iteration: int
    tokens_spent: int
    oracle: Dict[str, Any]
    logs_tail: str
    artifacts: List[str]
    metrics: JobMetrics
    acceptance_pass_at_iteration: Optional[int] = None
    budget_schedule_used: List[int] = Field(default_factory=list)
