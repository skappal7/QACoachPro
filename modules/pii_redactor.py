"""
PII Redaction Module
Regex-based PII detection and redaction (no ML libraries)
"""

import re
from typing import Dict, List


class PIIRedactor:
    """Lightweight PII redactor using regex patterns only"""
    
    def __init__(self):
        """Initialize regex patterns for PII detection"""
        self.patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE_US": re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "AADHAAR": re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b'),
            "ACCOUNT": re.compile(r'\b(?:account|acct)[\s#:]*(\d{6,})\b', re.IGNORECASE),
            "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "URL": re.compile(r'https?://[^\s]+'),
        }
        
        self.stats = {
            "EMAIL": 0,
            "PHONE_US": 0,
            "SSN": 0,
            "AADHAAR": 0,
            "ACCOUNT": 0,
            "CREDIT_CARD": 0,
            "URL": 0,
        }
    
    def redact(self, text: str) -> str:
        """
        Redact PII from text using regex patterns
        Args:
            text: Input text
        Returns:
            Redacted text with <REDACTED_TYPE> tokens
        """
        if not text or not isinstance(text, str):
            return text
        
        redacted = text
        
        # Apply each pattern
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(redacted)
            if matches:
                self.stats[pii_type] += len(matches)
                redacted = pattern.sub(f"<REDACTED_{pii_type}>", redacted)
        
        return redacted
    
    def redact_batch(self, texts: List[str]) -> List[str]:
        """
        Redact PII from a batch of texts
        Args:
            texts: List of input texts
        Returns:
            List of redacted texts
        """
        return [self.redact(text) for text in texts]
    
    def get_stats(self) -> Dict[str, int]:
        """Get redaction statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counter"""
        self.stats = {key: 0 for key in self.stats}


def redact_pii(text: str, enabled: bool = True) -> str:
    """
    Convenience function for single text redaction
    Args:
        text: Input text
        enabled: Whether redaction is enabled
    Returns:
        Redacted text (or original if disabled)
    """
    if not enabled:
        return text
    
    redactor = PIIRedactor()
    return redactor.redact(text)
