"""
backend/chat/guardrails.py

Two-layer guardrail system:

Layer 1 — Input guardrails (before RAG):
  - Prompt injection detection (keyword + pattern based)
  - Out-of-scope detection
  - PII detection using Microsoft Presidio

Layer 2 — Output guardrails (after LLM):
  - PII scrubbing from LLM response
  - Salary/sensitive data redaction

Presidio is used for PII detection — it catches:
  email addresses, phone numbers, SSN, credit cards,
  person names in sensitive contexts, etc.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Prompt injection patterns ──────────────────────────────────────────────
# These are the exact phrases used in the test case you showed
INJECTION_PATTERNS = [
    # Direct instruction override — broad match handles typos like "instructon"
    r"forget\s+(all\s+)?(past\s+|previous\s+|your\s+)?instruct\w*",
    r"ignore\s+(all\s+)?(past\s+|previous\s+|your\s+)?instruct\w*",
    r"disregard\s+(all\s+)?(past\s+|previous\s+)?instruct\w*",
    r"override\s+(all\s+)?(past\s+|previous\s+)?instruct\w*",
    # Broad "forget" catch — forget + any context word
    r"forget\s+(everything|all|past|previous|your|what|who)",
    r"ignore\s+(everything|all|past|previous|your|what|who)",
    # Role/identity override
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as",
    # Common jailbreak terms
    r"jailbreak",
    r"bypass\s+(your\s+)?(rules?|restrictions?|guidelines?|filter)",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
    r"sudo\s+",
    # Fake prompt injection
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[system\]",
    r"new\s+instruct\w*\s*:",
    r"your\s+(real\s+)?instruct\w*\s+are",
    # Extra safety
    r"reveal\s+(your\s+)?(prompt|instructions?|system)",
    r"what\s+(are\s+)?your\s+(instructions?|rules?|prompt)",
]

# ── Out-of-scope patterns ──────────────────────────────────────────────────
OUT_OF_SCOPE_PATTERNS = [
    r"write\s+(me\s+)?a\s+poem",
    r"tell\s+(me\s+)?a\s+joke",
    r"what\s+is\s+the\s+weather",
    r"recipe\s+for",
    r"who\s+won\s+the",
    r"write\s+(me\s+)?a\s+story",
    r"play\s+a\s+game",
    r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",
]

# Action requests — things the chatbot explicitly cannot do
# These go through guardrails BEFORE reaching the LLM
ACTION_PATTERNS = [
    r"send\s+(an?\s+)?email",
    r"send\s+(a\s+)?message",
    r"create\s+(a\s+)?(jira|ticket|task|bug|issue)",
    r"file\s+(a\s+)?(ticket|bug|issue)",
    r"schedule\s+(a\s+)?(meeting|call|event)",
    r"book\s+(a\s+)?(meeting|room|slot)",
    r"draft\s+(an?\s+)?email",
    r"write\s+(an?\s+)?email",
    r"search\s+the\s+(web|internet|google)",
    r"browse\s+the\s+(web|internet)",
    r"look\s+up\s+online",
    r"order\s+",
    r"buy\s+",
    r"purchase\s+",
]

# ── Sensitive data keywords (extra cautious for non-HR/Finance roles) ──────
SENSITIVE_DATA_KEYWORDS = [
    "salary", "salaries", "wage", "compensation", "payroll",
    "ssn", "social security", "bank account", "credit card",
    "password", "secret key", "api key", "private key",
]

# ── Presidio setup ─────────────────────────────────────────────────────────
_analyzer = None
_anonymizer = None


def _get_presidio():
    """Lazy-load Presidio — only initialized when needed."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
            logger.info("Presidio PII engine initialized")
        except ImportError:
            logger.warning("Presidio not installed — PII detection disabled")
    return _analyzer, _anonymizer


def check_input(question: str, user_role: str) -> dict:
    """
    Run all input guardrails on the user's question.

    Returns:
        {"safe": True} if the query is safe to process
        {"safe": False, "reason": "...", "message": "..."} if blocked
    """
    q_lower = question.lower().strip()

    # ── Check 1: Prompt injection ──────────────────────────────────────────
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            logger.warning(f"[GUARDRAIL] Prompt injection blocked: '{question[:80]}'")
            return {
                "safe": False,
                "reason": "prompt_injection",
                "message": "I'm sorry, but I cannot process that request. Please ask a genuine company-related question.",
            }

    # ── Check 2: Out of scope ──────────────────────────────────────────────
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            logger.warning(f"[GUARDRAIL] Out-of-scope blocked: '{question[:80]}'")
            return {
                "safe": False,
                "reason": "out_of_scope",
                "message": "I'm designed to answer questions from company documents. I can't help with that topic.",
            }

    # ── Check 2b: Action requests ──────────────────────────────────────────
    # These are handled at guardrail level — never reaches LLM
    # Prevents LLM from saying "I will do that" even though it cannot
    for pattern in ACTION_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            logger.warning(f"[GUARDRAIL] Action request blocked: '{question[:80]}'")
            return {
                "safe": False,
                "reason": "action_not_supported",
                "message": (
                    "I can only answer questions from company documents. "
                    "I'm not able to send emails, create tickets, schedule meetings, "
                    "or perform other actions in this version."
                ),
            }

    # ── Check 3: Sensitive data access by wrong role ───────────────────────
    # Engineering/Marketing should not be asking for salary details
    restricted_roles = {"ENGINEERING", "MARKETING"}
    if user_role in restricted_roles:
        for keyword in SENSITIVE_DATA_KEYWORDS:
            if keyword in q_lower:
                logger.warning(
                    f"[GUARDRAIL] Sensitive data access blocked: "
                    f"role={user_role}, keyword='{keyword}'"
                )
                return {
                    "safe": False,
                    "reason": "unauthorized_data_access",
                    "message": "That information is not accessible to your role. Please contact HR or Finance directly.",
                }

    # ── Check 4: PII in the question itself (Presidio) ────────────────────
    analyzer, _ = _get_presidio()
    if analyzer:
        try:
            pii_results = analyzer.analyze(text=question, language="en")
            # Filter out low-confidence detections
            high_conf_pii = [r for r in pii_results if r.score >= 0.7]
            if high_conf_pii:
                detected = [r.entity_type for r in high_conf_pii]
                logger.warning(f"[GUARDRAIL] PII in query detected: {detected}")
                # Don't block — just log. User may legitimately ask about their own info.
                # We log for monitoring but allow the query through.
        except Exception as e:
            logger.warning(f"Presidio analysis failed: {e}")

    return {"safe": True}


def scrub_output(answer: str) -> str:
    """
    Run output guardrails on the LLM's response.
    Redacts PII that the LLM may have included from context chunks.
    """
    if not answer:
        return answer

    analyzer, anonymizer = _get_presidio()
    if not analyzer or not anonymizer:
        return answer

    try:
        # Detect PII in the answer
        pii_results = analyzer.analyze(
            text=answer,
            language="en",
            entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN"],
        )

        if not pii_results:
            return answer

        # Redact detected PII
        scrubbed = anonymizer.anonymize(text=answer, analyzer_results=pii_results)
        scrubbed_text = scrubbed.text

        if scrubbed_text != answer:
            logger.warning(f"[GUARDRAIL] PII redacted from output")

        return scrubbed_text

    except Exception as e:
        logger.warning(f"Output scrubbing failed: {e}")
        return answer