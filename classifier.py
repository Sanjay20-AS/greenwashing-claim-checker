import re
from typing import Dict, List

from extraction import (
    granite_chat,
    extract_json
)


# ============================================================
# CLASSIFICATION PROMPT
# ============================================================

CLASSIFICATION_PROMPT = """
You are an environmental claim transparency analyzer.

Evaluate the sustainability claim using four criteria:

1. Specificity
   Is the claim clear and specific?

2. Measurability
   Does it provide numbers, percentages, dates,
   targets, or measurable environmental outcomes?

3. Evidence
   Does the document provide evidence, methodology,
   certification, data, or a source supporting the claim?

4. Scope clarity
   Does the claim clearly explain what product,
   operation, geography, period, or boundary it applies to?

Classify the claim as exactly one of:

- Well-supported
- Needs more evidence
- Vague/Unsupported

Important:

Do NOT accuse the company of lying, fraud, or greenwashing.

If evidence is missing, say:

"Insufficient evidence provided in this document."

Return ONLY valid JSON.

Use this structure:

{
  "status": "Well-supported",
  "specificity": {
    "score": 0,
    "reason": "short reason"
  },
  "measurability": {
    "score": 0,
    "reason": "short reason"
  },
  "evidence": {
    "score": 0,
    "reason": "short reason",
    "items_found": [],
    "items_missing": []
  },
  "scope_clarity": {
    "score": 0,
    "reason": "short reason"
  },
  "reason": "short neutral explanation",
  "missing_evidence": "what would strengthen the claim",
  "source_quote": "source quote"
}

Scoring:

Specificity: 0-25
Measurability: 0-25
Evidence: 0-30
Scope clarity: 0-20

Return scores using those exact maximums.
"""


# ============================================================
# RULE-BASED CLASSIFIER
# ============================================================

VAGUE_TERMS = [

    "green",
    "greener",
    "eco-friendly",
    "environmentally friendly",
    "sustainable",
    "planet friendly",
    "clean",
    "responsible",
    "good for the planet",
    "earth friendly",
    "natural"
]


def contains_number(
    text: str
) -> bool:

    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:%|percent|kg|ton|tonnes|t|"
            r"kgco2e|co2e|litre|liters|l|"
            r"mwh|kwh|years?)?\b",
            text.lower()
        )
    )


def rule_based_classification(
    claim: str,
    source_quote: str = ""
) -> Dict:

    lower = claim.lower()

    vague = any(
        term in lower
        for term in VAGUE_TERMS
    )

    measurable = contains_number(
        claim
    )

    has_time = bool(
        re.search(
            r"\b20\d{2}\b|"
            r"\b\d+\s*(?:year|years|month|months)\b",
            lower
        )
    )

    has_scope = any(
        word in lower
        for word in [

            "our products",
            "our operations",
            "manufacturing",
            "factory",
            "facility",
            "supply chain",
            "per unit",
            "per product",
            "globally",
            "in india",
            "in the us",
            "across"
        ]
    )

    has_evidence = any(
        word in lower
        for word in [

            "verified",
            "certified",
            "certificate",
            "audit",
            "audited",
            "report",
            "according to",
            "study",
            "data",
            "measured",
            "third-party",
            "third party",
            "ghg protocol",
            "sbti"
        ]
    )

    # UI maximums
    specificity = 8 if vague else 18

    measurability = (
        22 if measurable or has_time
        else 7
    )

    evidence = (
        25 if has_evidence
        else 6
    )

    scope = (
        16 if has_scope
        else 7
    )

    total = (
        specificity
        + measurability
        + evidence
        + scope
    )

    if total >= 70:

        status = "Well-supported"

        reason = (
            "The claim contains relatively clear "
            "and measurable information with "
            "supporting details."
        )

    elif total >= 40:

        status = "Needs more evidence"

        reason = (
            "The claim provides some useful "
            "information, but additional evidence "
            "or scope details are needed."
        )

    else:

        status = "Vague/Unsupported"

        reason = (
            "The claim uses broad environmental "
            "language without enough measurable "
            "or supporting information."
        )

    return {

        "status": status,

        "specificity": {
            "score": specificity,
            "reason": (
                "The claim is relatively specific."
                if not vague
                else
                "The claim uses broad environmental language."
            )
        },

        "measurability": {
            "score": measurability,
            "reason": (
                "The claim contains measurable information."
                if measurable or has_time
                else
                "No clear measurable value or time period is provided."
            )
        },

        "evidence": {
            "score": evidence,
            "reason": (
                "The claim contains an indicator of supporting evidence."
                if has_evidence
                else
                "Insufficient evidence provided in this document."
            ),
            "items_found": (
                ["Supporting evidence indicator"]
                if has_evidence
                else []
            ),
            "items_missing": (
                []
                if has_evidence
                else
                ["Methodology or supporting source"]
            )
        },

        "scope_clarity": {
            "score": scope,
            "reason": (
                "The scope or boundary is reasonably clear."
                if has_scope
                else
                "The applicable product, operation, geography, or boundary is unclear."
            )
        },

        "reason": reason,

        "source_quote": (
            source_quote or claim
        ),

        "missing_evidence": (
            "Provide measurable data, methodology, "
            "scope and supporting evidence."
        )
    }


# ============================================================
# GRANITE CLASSIFIER
# ============================================================

def classify_claim_with_granite(
    claim: str,
    source_quote: str = "",
    context: str = ""
) -> Dict:

    prompt = f"""
{CLASSIFICATION_PROMPT}

CLAIM:
{claim}

SOURCE QUOTE:
{source_quote}

RUBRIC CONTEXT:
{context}
"""

    try:

        result = granite_chat(
            prompt,
            max_new_tokens=900,
            temperature=0.1
        )

        parsed = extract_json(
            result
        )

        if not isinstance(
            parsed,
            dict
        ):
            raise ValueError(
                "Granite did not return valid JSON"
            )

        required = [

            "status",
            "specificity",
            "measurability",
            "evidence",
            "scope_clarity",
            "reason",
            "missing_evidence"
        ]

        if not all(
            key in parsed
            for key in required
        ):
            raise ValueError(
                "Granite response is missing required fields"
            )

        valid_statuses = [

            "Well-supported",
            "Needs more evidence",
            "Vague/Unsupported"
        ]

        if parsed["status"] not in valid_statuses:

            raise ValueError(
                "Invalid classification status"
            )

        # Normalize each criterion
        criteria = {

            "specificity": 25,
            "measurability": 25,
            "evidence": 30,
            "scope_clarity": 20
        }

        for key, maximum in criteria.items():

            value = parsed.get(
                key
            )

            # Handle both:
            # {"score": 20, "reason": "..."}
            # and a plain number
            if isinstance(
                value,
                dict
            ):

                try:
                    score = float(
                        value.get(
                            "score",
                            0
                        )
                    )

                except Exception:

                    score = 0

                score = max(
                    0,
                    min(
                        maximum,
                        score
                    )
                )

                parsed[key]["score"] = score

                if not value.get(
                    "reason"
                ):

                    parsed[key]["reason"] = ""

            else:

                try:

                    raw = float(
                        value
                    )

                except Exception:

                    raw = 0

                # If Granite returned 0-100,
                # convert to the criterion maximum.
                if raw > maximum:

                    raw = (
                        raw / 100
                    ) * maximum

                raw = max(
                    0,
                    min(
                        maximum,
                        raw
                    )
                )

                parsed[key] = {

                    "score": round(
                        raw,
                        2
                    ),

                    "reason": ""
                }

        parsed["source_quote"] = (
            parsed.get(
                "source_quote"
            )
            or source_quote
            or claim
        )

        return parsed

    except Exception as e:

        print(
            f"Granite classification failed: {e}"
        )

        return rule_based_classification(
            claim,
            source_quote
        )


# ============================================================
# MAIN CLASSIFICATION FUNCTION
# ============================================================

def classify_claim(
    claim: str,
    source_quote: str = "",
    context: str = "",
    use_llm: bool = True
) -> Dict:

    if use_llm:

        return classify_claim_with_granite(
            claim,
            source_quote,
            context
        )

    return rule_based_classification(
        claim,
        source_quote
    )


# ============================================================
# TRANSPARENCY SCORE
# ============================================================

def calculate_transparency_score(
    results: List[Dict]
) -> Dict:

    if not results:

        return {
            "score": 0,
            "specificity": 0,
            "measurability": 0,
            "evidence": 0,
            "scope": 0
        }

    specificity = (
        sum(
            r.get(
                "specificity",
                {}
            ).get(
                "score",
                0
            )
            for r in results
        )
        / len(results)
    )

    measurability = (
        sum(
            r.get(
                "measurability",
                {}
            ).get(
                "score",
                0
            )
            for r in results
        )
        / len(results)
    )

    evidence = (
        sum(
            r.get(
                "evidence",
                {}
            ).get(
                "score",
                0
            )
            for r in results
        )
        / len(results)
    )

    scope = (
        sum(
            r.get(
                "scope_clarity",
                {}
            ).get(
                "score",
                0
            )
            for r in results
        )
        / len(results)
    )

    score = (
        specificity
        + measurability
        + evidence
        + scope
    )

    return {

        "score": round(
            score,
            2
        ),

        "specificity": round(
            specificity,
            2
        ),

        "measurability": round(
            measurability,
            2
        ),

        "evidence": round(
            evidence,
            2
        ),

        "scope": round(
            scope,
            2
        )
    }