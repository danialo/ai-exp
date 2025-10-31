"""
PII redaction for awareness notes.

Masks sensitive information (emails, IPs, phone numbers, case IDs) before
persistence to protect privacy.
"""

import re
from typing import List, Pattern


class PIIRedactor:
    """
    Redacts personally identifiable information from text.

    Uses regex patterns for common PII types. Can be extended with NER models
    for more sophisticated detection.
    """

    # Regex patterns for common PII
    PATTERNS: List[tuple[str, Pattern, str]] = [
        # Email addresses
        (
            "email",
            re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            "[EMAIL]"
        ),
        # IPv4 addresses
        (
            "ipv4",
            re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
                r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            "[IP]"
        ),
        # IPv6 addresses (simplified pattern)
        (
            "ipv6",
            re.compile(
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
            ),
            "[IP]"
        ),
        # US phone numbers (various formats)
        (
            "phone_us",
            re.compile(
                r'\b(?:\+?1[-.\s]?)?'
                r'\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
            ),
            "[PHONE]"
        ),
        # Credit card numbers (basic pattern, not comprehensive)
        (
            "credit_card",
            re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            "[CARD]"
        ),
        # Social Security Numbers (US)
        (
            "ssn",
            re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
            "[SSN]"
        ),
        # Case/ticket IDs (common patterns like CASE-12345, TKT-999)
        (
            "case_id",
            re.compile(
                r'\b(?:CASE|TKT|TICKET|ID|REF)[-:#]?\d{3,}\b',
                re.IGNORECASE
            ),
            "[CASE_ID]"
        ),
        # Generic numeric IDs (6+ digits, preceded by common ID markers)
        (
            "numeric_id",
            re.compile(
                r'\b(?:ID|#|No\.?|Number)\s*:?\s*\d{6,}\b',
                re.IGNORECASE
            ),
            "[ID]"
        ),
        # API keys (generic pattern for long alphanumeric strings)
        (
            "api_key",
            re.compile(
                r'\b[A-Za-z0-9_-]{32,}\b'
            ),
            "[API_KEY]"
        ),
    ]

    def __init__(self, custom_patterns: List[tuple[str, str, str]] = None):
        """
        Initialize redactor.

        Args:
            custom_patterns: Additional patterns as (name, regex_str, replacement)
        """
        self.patterns = list(self.PATTERNS)

        if custom_patterns:
            for name, regex_str, replacement in custom_patterns:
                self.patterns.append((
                    name,
                    re.compile(regex_str),
                    replacement
                ))

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text potentially containing PII

        Returns:
            Text with PII replaced by placeholders
        """
        if not text:
            return text

        result = text

        for name, pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)

        return result

    def redact_batch(self, texts: List[str]) -> List[str]:
        """
        Redact PII from multiple texts.

        Args:
            texts: List of texts to redact

        Returns:
            List of redacted texts
        """
        return [self.redact(text) for text in texts]

    def add_pattern(self, name: str, regex: str, replacement: str) -> None:
        """
        Add custom redaction pattern.

        Args:
            name: Pattern identifier
            regex: Regular expression string
            replacement: Text to replace matches with
        """
        self.patterns.append((name, re.compile(regex), replacement))


# Global instance with default patterns
_default_redactor = PIIRedactor()


def redact_pii(text: str) -> str:
    """
    Convenience function for redacting PII with default patterns.

    Args:
        text: Input text

    Returns:
        Redacted text
    """
    return _default_redactor.redact(text)


def redact_pii_batch(texts: List[str]) -> List[str]:
    """
    Convenience function for batch redaction with default patterns.

    Args:
        texts: List of input texts

    Returns:
        List of redacted texts
    """
    return _default_redactor.redact_batch(texts)
