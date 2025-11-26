"""
Tests for Integration Layer Phase 4: Externalization & Observability.

Tests the FastAPI endpoints and Prometheus metrics for IL monitoring.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Import the app for testing
import sys
sys.path.insert(0, '/home/d/git/ai-exp')


class MockBudgetStatus:
    """Mock budget status for testing."""
    tokens_per_minute_limit = 2000
    tokens_used_this_minute = 500
    tokens_available = 1500
    beliefs_form_limit = 3
    beliefs_formed_today = 1
    beliefs_promote_limit = 5
    beliefs_promoted_today = 2
    beliefs_deprecate_limit = 3
    beliefs_deprecated_today = 0
    url_fetch_limit = 3
    url_fetches_this_session = 1
    min_introspection_interval_seconds = 30.0
    last_introspection = None
    cpu_usage = 0.25
    memory_usage = 0.40
    gpu_usage = 0.0
    dissonance_cooldown_map = {}


class MockFocusItem:
    """Mock focus item for testing."""
    def __init__(self, item_type, item_id, content, salience=0.8):
        self.item_type = MagicMock(value=item_type)
        self.item_id = item_id
        self.content = content
        self.salience = salience
        self.entered_focus = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 1
        self.decay_rate = 0.1
        self.min_salience_threshold = 0.2


class MockSelfModel:
    """Mock self model for testing."""
    snapshot_id = "test-snapshot-001"
    core_beliefs = ["belief1", "belief2"]
    peripheral_beliefs = ["belief3"]
    anchor_drift = 0.05


class MockExecutionMode:
    """Mock execution mode enum."""
    value = "interactive"


class MockAstraState:
    """Mock AstraState for testing."""
    def __init__(self):
        self.tick_id = 42
        self.timestamp = datetime.now()
        self.mode = MockExecutionMode()
        self.session_id = "test-session-001"
        self.self_model = MockSelfModel()
        self.focus_stack = [
            MockFocusItem("user_message", "msg-001", "Test message", 0.9),
            MockFocusItem("goal", "goal-001", "Test goal", 0.7),
        ]
        self.attention_capacity = 0.8
        self.active_goals = []
        self.goal_queue = None
        self.task_context = {}
        self.percept_buffer = []
        self.dissonance_alerts = []
        self.integration_events = []
        self.emotional_state = None
        self.arousal_level = 0.5
        self.cognitive_load = 0.3
        self.budget_status = MockBudgetStatus()
        self.last_introspection = None

    def to_snapshot(self):
        """Return a mock snapshot."""
        return MagicMock(
            tick_id=self.tick_id,
            timestamp=self.timestamp.isoformat(),
            mode=self.mode.value,
            session_id=self.session_id,
            self_model_id=self.self_model.snapshot_id if self.self_model else None,
            core_belief_count=len(self.self_model.core_beliefs) if self.self_model else 0,
            peripheral_belief_count=len(self.self_model.peripheral_beliefs) if self.self_model else 0,
            anchor_drift=self.self_model.anchor_drift if self.self_model else 0.0,
            focus_stack=[{"type": "user_message", "content": "Test", "salience": 0.9}],
            attention_capacity=self.attention_capacity,
            active_goal_ids=[],
            pending_goal_count=0,
            active_task_ids=[],
            percept_buffer_size=0,
            dissonance_alert_count=0,
            integration_event_count=0,
            emotional_state=None,
            arousal_level=self.arousal_level,
            cognitive_load=self.cognitive_load,
            tokens_available=self.budget_status.tokens_available,
            tokens_used_this_minute=self.budget_status.tokens_used_this_minute,
            beliefs_formed_today=self.budget_status.beliefs_formed_today,
            last_introspection=None,
        )


class MockIntegrationLayer:
    """Mock Integration Layer for testing."""
    def __init__(self):
        self.state = MockAstraState()
        self.tick_history = [
            {"tick_id": 40, "duration_ms": 15, "actions": 1},
            {"tick_id": 41, "duration_ms": 22, "actions": 2},
            {"tick_id": 42, "duration_ms": 18, "actions": 1},
        ]
        self.action_log = [
            {"tick_id": 40, "action_type": "USER_RESPONSE", "target": "msg-001"},
            {"tick_id": 41, "action_type": "INTROSPECTION", "target": None},
            {"tick_id": 42, "action_type": "BELIEF_GARDENING", "target": None},
        ]


@pytest.fixture
def mock_app():
    """Create a mock FastAPI app with integration layer."""
    from fastapi import FastAPI
    from src.api.integration_endpoints import router

    app = FastAPI()
    app.include_router(router)
    app.state.integration_layer = MockIntegrationLayer()

    return app


@pytest.fixture
def client(mock_app):
    """Create test client."""
    return TestClient(mock_app)


class TestIntegrationStateEndpoint:
    """Tests for GET /api/integration/state"""

    def test_get_state_returns_valid_json(self, client):
        """State endpoint returns valid JSON with expected fields."""
        response = client.get("/api/integration/state")
        assert response.status_code == 200

        data = response.json()
        assert "tick_id" in data
        assert "mode" in data
        assert "identity" in data
        assert "attention" in data
        assert "goals" in data
        assert "signals" in data
        assert "modulation" in data
        assert "budgets" in data

    def test_get_state_identity_fields(self, client):
        """State endpoint includes identity information."""
        response = client.get("/api/integration/state")
        data = response.json()

        identity = data["identity"]
        assert "self_model_id" in identity
        assert "core_belief_count" in identity
        assert "peripheral_belief_count" in identity
        assert "anchor_drift" in identity

    def test_get_state_without_il_returns_503(self, mock_app):
        """Returns 503 when Integration Layer is not initialized."""
        mock_app.state.integration_layer = None
        client = TestClient(mock_app)

        response = client.get("/api/integration/state")
        assert response.status_code == 503


class TestFocusEndpoint:
    """Tests for GET /api/integration/focus"""

    def test_get_focus_returns_stack(self, client):
        """Focus endpoint returns focus stack."""
        response = client.get("/api/integration/focus")
        assert response.status_code == 200

        data = response.json()
        assert "focus_stack" in data
        assert "stack_size" in data
        assert "attention_capacity" in data
        assert "cognitive_load" in data

    def test_focus_stack_has_items(self, client):
        """Focus stack contains expected items."""
        response = client.get("/api/integration/focus")
        data = response.json()

        assert data["stack_size"] == 2
        assert len(data["focus_stack"]) == 2

        # Check item structure
        item = data["focus_stack"][0]
        assert "item_type" in item
        assert "salience" in item
        assert "content" in item


class TestBudgetsEndpoint:
    """Tests for GET /api/integration/budgets"""

    def test_get_budgets_returns_all_categories(self, client):
        """Budgets endpoint returns all budget categories."""
        response = client.get("/api/integration/budgets")
        assert response.status_code == 200

        data = response.json()
        assert "tokens" in data
        assert "beliefs" in data
        assert "url_fetch" in data
        assert "introspection" in data
        assert "resources" in data

    def test_token_budget_calculations(self, client):
        """Token budget shows correct utilization."""
        response = client.get("/api/integration/budgets")
        data = response.json()

        tokens = data["tokens"]
        assert tokens["per_minute_limit"] == 2000
        assert tokens["used_this_minute"] == 500
        assert tokens["available"] == 1500
        assert tokens["utilization_pct"] == 25.0

    def test_belief_budget_remaining(self, client):
        """Belief budgets show remaining operations."""
        response = client.get("/api/integration/budgets")
        data = response.json()

        beliefs = data["beliefs"]
        assert beliefs["form_remaining"] == 2  # 3 - 1
        assert beliefs["promote_remaining"] == 3  # 5 - 2
        assert beliefs["deprecate_remaining"] == 3  # 3 - 0


class TestModeEndpoint:
    """Tests for POST /api/integration/mode"""

    def test_change_mode_to_autonomous(self, client):
        """Can change mode to autonomous."""
        response = client.post(
            "/api/integration/mode",
            json={"mode": "autonomous", "reason": "test"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["previous_mode"] == "interactive"
        assert data["new_mode"] == "autonomous"
        assert data["reason"] == "test"
        assert "changed_at" in data

    def test_change_mode_to_maintenance(self, client):
        """Can change mode to maintenance."""
        response = client.post(
            "/api/integration/mode",
            json={"mode": "maintenance", "reason": "scheduled"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["new_mode"] == "maintenance"

    def test_invalid_mode_returns_400(self, client):
        """Invalid mode returns 400 error."""
        response = client.post(
            "/api/integration/mode",
            json={"mode": "invalid_mode", "reason": "test"}
        )
        assert response.status_code == 400
        assert "Invalid mode" in response.json()["detail"]


class TestHistoryEndpoint:
    """Tests for GET /api/integration/history"""

    def test_get_history_returns_ticks(self, client):
        """History endpoint returns tick history."""
        response = client.get("/api/integration/history")
        assert response.status_code == 200

        data = response.json()
        assert "current_tick" in data
        assert "mode" in data
        assert "tick_history" in data
        assert "recent_actions" in data
        assert "history_size" in data

    def test_history_limit_parameter(self, client):
        """History respects limit parameter."""
        response = client.get("/api/integration/history?limit=2")
        data = response.json()

        assert len(data["tick_history"]) <= 2
        assert len(data["recent_actions"]) <= 2

    def test_history_contains_action_log(self, client):
        """History includes action log entries."""
        response = client.get("/api/integration/history")
        data = response.json()

        assert data["action_log_size"] == 3
        assert len(data["recent_actions"]) > 0


class TestMetricsEndpoint:
    """Tests for GET /api/integration/metrics"""

    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint returns Prometheus format."""
        response = client.get("/api/integration/metrics")
        assert response.status_code == 200

        # Check content type
        assert "text/plain" in response.headers["content-type"]

        # Check for expected metrics in response
        content = response.text
        assert "astra_focus_stack_size" in content
        assert "astra_cognitive_load" in content
        assert "astra_budget_tokens_available" in content

    def test_metrics_contains_gauges(self, client):
        """Metrics contains gauge values."""
        response = client.get("/api/integration/metrics")
        content = response.text

        # Should have HELP and TYPE lines
        assert "# HELP astra_focus_stack_size" in content
        assert "# TYPE astra_focus_stack_size gauge" in content

    def test_metrics_contains_counters(self, client):
        """Metrics contains counter values."""
        response = client.get("/api/integration/metrics")
        content = response.text

        assert "astra_tick_count_total" in content


class TestEndpointErrors:
    """Tests for error handling across endpoints."""

    def test_all_endpoints_handle_missing_il(self, mock_app):
        """All endpoints return 503 when IL is missing."""
        mock_app.state.integration_layer = None
        client = TestClient(mock_app)

        endpoints = [
            "/api/integration/state",
            "/api/integration/focus",
            "/api/integration/budgets",
            "/api/integration/history",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 503, f"Endpoint {endpoint} should return 503"

        # Mode change should also fail
        response = client.post(
            "/api/integration/mode",
            json={"mode": "autonomous", "reason": "test"}
        )
        assert response.status_code == 503
