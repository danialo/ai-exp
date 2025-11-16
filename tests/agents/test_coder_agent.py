"""Unit tests for CoderAgent."""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from src.agents.coder_agent import CoderAgent


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    llm = Mock()
    llm.generate_with_tools = AsyncMock()
    return llm


@pytest.fixture
def coder_agent(mock_llm_service):
    """Create CoderAgent instance with mock LLM."""
    return CoderAgent(llm_service=mock_llm_service)


@pytest.fixture
def valid_coder_response():
    """Valid JSON response from CoderAgent."""
    return {
        "plan": [
            "Create parse_kv() function",
            "Add error handling",
            "Write comprehensive tests"
        ],
        "artifacts": [
            {
                "filename": "src/utils/kv_parser.py",
                "language": "python",
                "code": '''"""Key-value parser."""

def parse_kv(line: str) -> dict:
    """Parse key=value line."""
    if "=" not in line:
        raise ValueError("Invalid format")
    key, value = line.split("=", 1)
    return {key.strip(): value.strip()}
'''
            },
            {
                "filename": "tests/test_kv_parser.py",
                "language": "python",
                "code": '''"""Tests for KV parser."""

import pytest
from src.utils.kv_parser import parse_kv

def test_simple_kv():
    assert parse_kv("foo=bar") == {"foo": "bar"}

def test_invalid_format():
    with pytest.raises(ValueError):
        parse_kv("no_equals")
'''
            }
        ],
        "checks": {
            "ruff_black_clean": True,
            "mypy_clean": True,
            "forbidden_apis_used": [],
            "size_ok": True,
            "idempotent_key": "abc123def456"
        },
        "assumptions": [
            "Input is always a string",
            "Empty values are valid"
        ]
    }


class TestCoderAgentBasics:
    """Test basic CoderAgent functionality."""

    def test_init(self, mock_llm_service):
        """Test CoderAgent initialization."""
        agent = CoderAgent(llm_service=mock_llm_service)
        assert agent.llm == mock_llm_service
        assert agent.system_prompt is not None
        assert len(agent.system_prompt) > 0

    def test_load_prompt(self, coder_agent):
        """Test prompt loading from file."""
        prompt = coder_agent._load_prompt()
        assert "JSON-ONLY OUTPUT" in prompt
        assert "FORBIDDEN APIS" in prompt
        assert "SIZE LIMITS" in prompt

    def test_load_prompt_missing_file(self, mock_llm_service):
        """Test error handling when prompt file missing."""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                CoderAgent(llm_service=mock_llm_service)

    def test_get_capabilities(self, coder_agent):
        """Test capabilities reporting."""
        caps = coder_agent.get_capabilities()
        assert caps["name"] == "CoderAgent"
        assert "code generation" in caps["description"].lower()
        assert "features" in caps
        assert "constraints" in caps


class TestProcessMethod:
    """Test the main process() method."""

    @pytest.mark.asyncio
    async def test_simple_request(self, coder_agent, mock_llm_service, valid_coder_response):
        """Test processing a simple code generation request."""
        # Mock LLM response
        mock_llm_service.generate_with_tools.return_value = {
            "message": Mock(content=json.dumps(valid_coder_response))
        }

        request = {
            "goal_text": "implement_feature",
            "context": {
                "requirements": "Create parse_kv function"
            }
        }

        result = await coder_agent.process(request)

        # Verify LLM was called correctly
        assert mock_llm_service.generate_with_tools.called
        call_kwargs = mock_llm_service.generate_with_tools.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 8000
        assert call_kwargs["tools"] is None

        # Verify output structure
        assert "plan" in result
        assert "artifacts" in result
        assert "checks" in result
        assert "assumptions" in result

    @pytest.mark.asyncio
    async def test_request_with_all_context(self, coder_agent, mock_llm_service, valid_coder_response):
        """Test request with full context."""
        mock_llm_service.generate_with_tools.return_value = {
            "message": Mock(content=json.dumps(valid_coder_response))
        }

        request = {
            "goal_text": "fix_bug",
            "context": {
                "requirements": "Fix the parser",
                "existing_files": ["src/utils/parser.py"],
                "constraints": ["no network", "pure stdlib"]
            },
            "timeout_ms": 60000
        }

        result = await coder_agent.process(request)

        # Check user prompt was built correctly
        call_args = mock_llm_service.generate_with_tools.call_args
        messages = call_args[1]["messages"]
        user_prompt = messages[1]["content"]

        assert "GOAL: fix_bug" in user_prompt
        assert "REQUIREMENTS:" in user_prompt
        assert "Fix the parser" in user_prompt
        assert "EXISTING FILES" in user_prompt
        assert "src/utils/parser.py" in user_prompt
        assert "CONSTRAINTS:" in user_prompt


class TestJSONParsing:
    """Test JSON parsing from LLM output."""

    def test_parse_clean_json(self, coder_agent):
        """Test parsing clean JSON."""
        json_str = '{"plan": [], "artifacts": []}'
        result = coder_agent._parse_json_output(json_str)
        assert result == {"plan": [], "artifacts": []}

    def test_parse_json_with_markdown_fences(self, coder_agent):
        """Test parsing JSON wrapped in markdown."""
        json_str = '```json\n{"plan": [], "artifacts": []}\n```'
        result = coder_agent._parse_json_output(json_str)
        assert result == {"plan": [], "artifacts": []}

    def test_parse_json_with_generic_fences(self, coder_agent):
        """Test parsing JSON with generic fences."""
        json_str = '```\n{"plan": [], "artifacts": []}\n```'
        result = coder_agent._parse_json_output(json_str)
        assert result == {"plan": [], "artifacts": []}

    def test_parse_json_with_whitespace(self, coder_agent):
        """Test parsing JSON with extra whitespace."""
        json_str = '\n\n  {"plan": [], "artifacts": []}  \n\n'
        result = coder_agent._parse_json_output(json_str)
        assert result == {"plan": [], "artifacts": []}

    def test_parse_invalid_json(self, coder_agent):
        """Test error handling for invalid JSON."""
        json_str = '{plan: [], not valid json}'
        with pytest.raises(json.JSONDecodeError):
            coder_agent._parse_json_output(json_str)


class TestSafetyValidation:
    """Test safety validation (forbidden APIs, size limits)."""

    def test_detect_forbidden_eval(self, coder_agent):
        """Test detection of eval() in code."""
        output = {
            "artifacts": [{
                "filename": "bad.py",
                "code": "result = eval(user_input)"
            }]
        }
        with pytest.raises(ValueError, match="Forbidden API"):
            coder_agent._validate_safety(output)

    def test_detect_forbidden_subprocess(self, coder_agent):
        """Test detection of subprocess in code."""
        output = {
            "artifacts": [{
                "filename": "bad.py",
                "code": "import subprocess\nsubprocess.run(['ls'])"
            }]
        }
        with pytest.raises(ValueError, match="Forbidden API"):
            coder_agent._validate_safety(output)

    def test_allow_forbidden_in_comments(self, coder_agent):
        """Test that forbidden APIs in comments are allowed."""
        output = {
            "artifacts": [{
                "filename": "good.py",
                "code": "# Don't use eval() - it's dangerous\ndef safe_func():\n    pass"
            }]
        }
        # Should not raise
        coder_agent._validate_safety(output)

    def test_impl_file_exceeds_line_limit(self, coder_agent):
        """Test detection of implementation file exceeding line limit."""
        large_code = "\n".join([f"line_{i} = {i}" for i in range(500)])
        output = {
            "artifacts": [{
                "filename": "src/large.py",
                "code": large_code
            }]
        }
        with pytest.raises(ValueError, match="exceeds line limit"):
            coder_agent._validate_safety(output)

    def test_test_file_exceeds_line_limit(self, coder_agent):
        """Test detection of test file exceeding line limit."""
        large_code = "\n".join([f"def test_{i}(): pass" for i in range(300)])
        output = {
            "artifacts": [{
                "filename": "tests/test_large.py",
                "code": large_code
            }]
        }
        with pytest.raises(ValueError, match="exceeds line limit"):
            coder_agent._validate_safety(output)

    def test_impl_file_exceeds_byte_limit(self, coder_agent):
        """Test detection of file exceeding byte limit."""
        # Create file with many long strings to exceed 40KB
        large_code = "\n".join([f"x{i} = '{' ' * 200}'" for i in range(300)])
        output = {
            "artifacts": [{
                "filename": "src/large.py",
                "code": large_code
            }]
        }
        with pytest.raises(ValueError, match="exceeds byte limit"):
            coder_agent._validate_safety(output)

    def test_valid_code_passes_safety(self, coder_agent):
        """Test that valid code passes all safety checks."""
        output = {
            "artifacts": [
                {
                    "filename": "src/valid.py",
                    "code": "def add(a: int, b: int) -> int:\n    return a + b"
                },
                {
                    "filename": "tests/test_valid.py",
                    "code": "def test_add():\n    assert add(1, 2) == 3"
                }
            ]
        }
        # Should not raise
        coder_agent._validate_safety(output)


class TestSchemaValidation:
    """Test output schema validation."""

    def test_valid_schema(self, coder_agent, valid_coder_response):
        """Test validation of correct schema."""
        assert coder_agent.validate_output(valid_coder_response) is True

    def test_missing_plan(self, coder_agent, valid_coder_response):
        """Test detection of missing 'plan' key."""
        del valid_coder_response["plan"]
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_missing_artifacts(self, coder_agent, valid_coder_response):
        """Test detection of missing 'artifacts' key."""
        del valid_coder_response["artifacts"]
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_missing_checks(self, coder_agent, valid_coder_response):
        """Test detection of missing 'checks' key."""
        del valid_coder_response["checks"]
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_missing_assumptions(self, coder_agent, valid_coder_response):
        """Test detection of missing 'assumptions' key."""
        del valid_coder_response["assumptions"]
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_invalid_plan_type(self, coder_agent, valid_coder_response):
        """Test detection of wrong type for 'plan'."""
        valid_coder_response["plan"] = "not a list"
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_invalid_artifacts_type(self, coder_agent, valid_coder_response):
        """Test detection of wrong type for 'artifacts'."""
        valid_coder_response["artifacts"] = "not a list"
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_artifact_missing_filename(self, coder_agent, valid_coder_response):
        """Test detection of artifact missing 'filename'."""
        valid_coder_response["artifacts"][0] = {"language": "python", "code": "pass"}
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_artifact_missing_code(self, coder_agent, valid_coder_response):
        """Test detection of artifact missing 'code'."""
        valid_coder_response["artifacts"][0] = {"filename": "test.py", "language": "python"}
        assert coder_agent.validate_output(valid_coder_response) is False

    def test_checks_missing_key(self, coder_agent, valid_coder_response):
        """Test detection of missing key in 'checks'."""
        del valid_coder_response["checks"]["idempotent_key"]
        assert coder_agent.validate_output(valid_coder_response) is False


class TestUserPromptBuilding:
    """Test building user prompts from requests."""

    def test_minimal_request(self, coder_agent):
        """Test building prompt from minimal request."""
        request = {"goal_text": "implement_feature"}
        prompt = coder_agent._build_user_prompt(request)
        assert "GOAL: implement_feature" in prompt

    def test_request_with_requirements(self, coder_agent):
        """Test building prompt with requirements."""
        request = {
            "goal_text": "implement_feature",
            "context": {
                "requirements": "Build a calculator"
            }
        }
        prompt = coder_agent._build_user_prompt(request)
        assert "GOAL: implement_feature" in prompt
        assert "REQUIREMENTS:" in prompt
        assert "Build a calculator" in prompt

    def test_request_with_existing_files(self, coder_agent):
        """Test building prompt with existing files."""
        request = {
            "goal_text": "refactor_code",
            "context": {
                "existing_files": ["src/calc.py", "src/parser.py"]
            }
        }
        prompt = coder_agent._build_user_prompt(request)
        assert "EXISTING FILES TO CONSIDER:" in prompt
        assert "src/calc.py" in prompt
        assert "src/parser.py" in prompt

    def test_request_with_constraints(self, coder_agent):
        """Test building prompt with constraints."""
        request = {
            "goal_text": "implement_feature",
            "context": {
                "constraints": ["no network", "pure stdlib", "max 400 lines"]
            }
        }
        prompt = coder_agent._build_user_prompt(request)
        assert "CONSTRAINTS:" in prompt
        assert "no network" in prompt
        assert "pure stdlib" in prompt
        assert "max 400 lines" in prompt
