# Token Usage Tracking Implementation - TODO

**Priority:** High
**Estimated Time:** 2-3 hours
**Status:** Not Started

---

## Overview

Implement comprehensive token usage tracking to move from estimates to real data. Currently spending ~$24/month based on estimates. Need actual tracking for accurate budgeting and optimization.

---

## Tasks

### 1. Create Database Schema ⏱️ 30 min

**File:** `src/db/migrations/add_api_usage_table.sql` (create migrations dir if needed)

```sql
CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model TEXT NOT NULL,
    feature TEXT NOT NULL,  -- 'chat', 'introspection', 'belief_gardener', 'decision', etc.
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    session_id TEXT,
    metadata JSON,

    UNIQUE(timestamp, feature, session_id)  -- Prevent duplicates
);

CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_feature ON api_usage(feature);
CREATE INDEX IF NOT EXISTS idx_usage_date ON api_usage(DATE(timestamp));
CREATE INDEX IF NOT EXISTS idx_usage_model ON api_usage(model);
```

**Location:** Add to `data/raw_store.db` or create separate `data/api_usage.db`

---

### 2. Add Tracking Helper Function ⏱️ 30 min

**File:** `src/services/llm.py`

Add after the `LLMService` class:

```python
def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD based on model pricing."""
    # Pricing as of 2024 (update as needed)
    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.0100},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    # Default to gpt-4o pricing if model not found
    rates = pricing.get(model, pricing["gpt-4o"])

    input_cost = (prompt_tokens / 1000) * rates["input"]
    output_cost = (completion_tokens / 1000) * rates["output"]

    return input_cost + output_cost

def _track_usage(
    self,
    model: str,
    usage: Any,  # OpenAI usage object
    feature: str,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Track API usage to database."""
    import sqlite3
    from pathlib import Path
    from config.settings import Settings

    # Calculate cost
    cost = self._calculate_cost(
        model,
        usage.prompt_tokens,
        usage.completion_tokens
    )

    # Get DB path
    db_path = Path(Settings.PROJECT_ROOT) / "data" / "api_usage.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize DB if needed
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                feature TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                session_id TEXT,
                metadata JSON
            )
        """)

        # Insert usage record
        conn.execute("""
            INSERT INTO api_usage
            (model, feature, prompt_tokens, completion_tokens, total_tokens, cost_usd, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model,
            feature,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            cost,
            session_id,
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()
```

---

### 3. Update `generate_with_tools()` ⏱️ 20 min

**File:** `src/services/llm.py`

In the `generate_with_tools` method, add tracking after the API call:

```python
def generate_with_tools(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: Dict[int, float] | None = None,
    feature: str = "unknown",  # ADD THIS PARAMETER
    session_id: str | None = None,  # ADD THIS PARAMETER
) -> Dict[str, Any]:
    """Generate response with OpenAI function calling / tools."""

    # ... existing code ...

    try:
        response = self.client.chat.completions.create(**kwargs)
    except Exception as e:
        # ... existing error handling ...
        raise

    # ADD THIS: Track usage
    if hasattr(response, 'usage') and response.usage:
        try:
            self._track_usage(
                model=self.model,
                usage=response.usage,
                feature=feature,
                session_id=session_id,
                metadata={
                    "tools_count": len(tools),
                    "messages_count": len(messages),
                }
            )
        except Exception as e:
            # Don't fail on tracking errors
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to track usage: {e}")

    choice = response.choices[0]

    # ... rest of existing code ...
```

---

### 4. Update Call Sites ⏱️ 30 min

Update all places that call `generate_with_tools()` to pass `feature` parameter:

**Common locations:**
- `src/services/persona_runtime.py` - feature="chat"
- `src/services/awareness_loop.py` - feature="introspection"
- `src/services/belief_gardener.py` - feature="belief_gardener"
- `src/services/decision_framework.py` - feature="decision"

Example change:
```python
# Before
result = llm.generate_with_tools(messages, tools)

# After
result = llm.generate_with_tools(
    messages,
    tools,
    feature="chat",
    session_id=session_id
)
```

---

### 5. Add Budget Checking ⏱️ 30 min

**File:** `src/services/budget_guard.py` (NEW)

```python
"""Budget guard to prevent overspending."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import Settings


class BudgetGuard:
    """Check if API usage is within budget limits."""

    # Budget limits (configurable via .env later)
    DAILY_TOKEN_LIMIT = 200_000
    DAILY_COST_LIMIT = 5.00  # $5/day
    MONTHLY_COST_LIMIT = 100.00  # $100/month

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = Path(Settings.PROJECT_ROOT) / "data" / "api_usage.db"
        self.db_path = db_path

    def check_daily_budget(self) -> dict:
        """Check if within daily budget."""
        today = datetime.now().date()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    SUM(total_tokens) as tokens,
                    SUM(cost_usd) as cost
                FROM api_usage
                WHERE DATE(timestamp) = ?
            """, (today,))

            row = cursor.fetchone()
            tokens = row[0] or 0
            cost = row[1] or 0.0

        return {
            "tokens_used": tokens,
            "tokens_limit": self.DAILY_TOKEN_LIMIT,
            "tokens_remaining": self.DAILY_TOKEN_LIMIT - tokens,
            "cost_used": cost,
            "cost_limit": self.DAILY_COST_LIMIT,
            "cost_remaining": self.DAILY_COST_LIMIT - cost,
            "over_budget": tokens >= self.DAILY_TOKEN_LIMIT or cost >= self.DAILY_COST_LIMIT
        }

    def check_monthly_budget(self) -> dict:
        """Check if within monthly budget."""
        # Similar to daily but for current month
        pass  # Implement similar logic


def check_budget_before_call(feature: str = "unknown") -> bool:
    """Check budget before making API call. Returns True if safe to proceed."""
    guard = BudgetGuard()
    status = guard.check_daily_budget()

    if status["over_budget"]:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Daily budget exceeded! Tokens: {status['tokens_used']}/{status['tokens_limit']}, "
            f"Cost: ${status['cost_used']:.2f}/${status['cost_limit']:.2f}"
        )
        return False

    return True
```

Then in `LLMService.generate_with_tools()`:

```python
# At the start of the method
from src.services.budget_guard import check_budget_before_call

if not check_budget_before_call(feature):
    raise RuntimeError("Daily budget exceeded. API calls disabled.")
```

---

### 6. Create Usage Dashboard Endpoint ⏱️ 20 min

**File:** `app.py`

Add new endpoint:

```python
@app.get("/api/usage/stats")
async def get_usage_stats(
    period: str = "today"  # "today", "week", "month"
):
    """Get token usage statistics."""
    from src.services.usage_stats import get_usage_stats

    stats = get_usage_stats(period)
    return stats
```

**File:** `src/services/usage_stats.py` (NEW)

```python
"""Usage statistics service."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import Settings


def get_usage_stats(period: str = "today") -> dict:
    """Get usage statistics for a time period."""
    db_path = Path(Settings.PROJECT_ROOT) / "data" / "api_usage.db"

    # Calculate date range
    if period == "today":
        start_date = datetime.now().date()
    elif period == "week":
        start_date = datetime.now().date() - timedelta(days=7)
    elif period == "month":
        start_date = datetime.now().date() - timedelta(days=30)
    else:
        start_date = datetime.now().date()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as calls,
                SUM(total_tokens) as tokens,
                SUM(cost_usd) as cost
            FROM api_usage
            WHERE DATE(timestamp) >= ?
        """, (start_date,))

        overall = cursor.fetchone()

        # By feature
        cursor.execute("""
            SELECT
                feature,
                COUNT(*) as calls,
                SUM(total_tokens) as tokens,
                SUM(cost_usd) as cost
            FROM api_usage
            WHERE DATE(timestamp) >= ?
            GROUP BY feature
            ORDER BY cost DESC
        """, (start_date,))

        by_feature = [
            {
                "feature": row[0],
                "calls": row[1],
                "tokens": row[2],
                "cost": row[3]
            }
            for row in cursor.fetchall()
        ]

    return {
        "period": period,
        "start_date": str(start_date),
        "total_calls": overall[0] or 0,
        "total_tokens": overall[1] or 0,
        "total_cost": overall[2] or 0.0,
        "by_feature": by_feature
    }
```

---

## Testing Checklist

- [ ] Database table created successfully
- [ ] Token tracking captures data on API calls
- [ ] Usage stats endpoint returns correct data
- [ ] Budget guard prevents calls when over budget
- [ ] All features (chat, introspection, beliefs) tracked separately
- [ ] Cost calculations match OpenAI pricing
- [ ] No performance degradation from tracking

---

## Rollout Plan

1. **Phase 1:** Create DB schema and tracking functions (no enforcement)
2. **Phase 2:** Add tracking to all call sites (monitor only)
3. **Phase 3:** Enable budget warnings (log only)
4. **Phase 4:** Enable budget enforcement (optional via config)

---

## Configuration (add to .env)

```bash
# Token usage tracking
USAGE_TRACKING_ENABLED=true
DAILY_TOKEN_BUDGET=200000
DAILY_COST_BUDGET=5.00
MONTHLY_COST_BUDGET=100.00
BUDGET_ENFORCEMENT=false  # Start with warnings only
```

---

## Success Metrics

After implementation:
- [ ] Can view daily/weekly/monthly token usage
- [ ] Can see cost breakdown by feature
- [ ] Can identify most expensive operations
- [ ] Can set budget alerts
- [ ] Move from $24/month estimate to actual data

---

**Estimated total time:** 2-3 hours
**Priority:** High (move from estimates to real data)
**Dependencies:** None (can be implemented independently)
