import os
import json
import re
from typing import List, Dict

import fitz
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ============================================================
# HUGGING FACE CONFIG
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "ibm-granite/granite-4.2-3b"
)

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "deepinfra"
)

client = None

if HF_TOKEN:
    client = InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using PyMuPDF.
    """

    document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        text = page.get_text("text")

        if text:
            pages.append(text)

    document.close()

    return "\n".join(pages)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean unnecessary whitespace.
    """

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text: str,
    max_chars: int = 6000
) -> List[str]:

    text = clean_text(text)

    if len(text) <= max_chars:
        return [text]

    chunks = []

    paragraphs = text.split("\n\n")

    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) + 2 <= max_chars:

            if current:
                current += "\n\n"

            current += paragraph

        else:

            if current:
                chunks.append(current)

            current = paragraph

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# GRANITE CHAT
# ============================================================

def granite_chat(
    prompt: str,
    max_new_tokens: int = 1000,
    temperature: float = 0.1
) -> str:
    """
    Send prompt to IBM Granite through Hugging Face.

    Uses the DeepInfra inference provider.
    """

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured. "
            "Add HF_TOKEN to your .env file."
        )

    if client is None:
        raise RuntimeError(
            "Hugging Face client could not be initialized."
        )

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_new_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content or ""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str):

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Direct JSON
    try:
        return json.loads(text)

    except Exception:
        pass

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(match.group(0))

        except Exception:
            pass

    # Find JSON array
    match = re.search(
        r"\[.*\]",
        text,
        re.DOTALL
    )

    if match:

        try:
            return json.loads(match.group(0))

        except Exception:
            pass

    return None


# ============================================================
# GRANITE CLAIM EXTRACTION
# ============================================================

def extract_claims_with_granite(
    text: str
) -> List[Dict]:

    prompt = f"""
You are an environmental sustainability document analyzer.

Extract environmental and sustainability claims from the document.

A claim is a statement about:

- environmental impact
- sustainability
- emissions
- carbon
- renewable energy
- recycling
- waste reduction
- water conservation
- climate impact
- sustainable materials
- environmental performance

Do NOT invent claims.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "claims": [
    {{
      "claim": "exact or near-exact claim",
      "context": "short surrounding context",
      "location": "page or section if available, otherwise unknown"
    }}
  ]
}}

DOCUMENT:

{text}
"""

    try:

        result = granite_chat(
            prompt,
            max_new_tokens=1200,
            temperature=0.1
        )

        parsed = extract_json(result)

        if isinstance(parsed, dict):

            claims = parsed.get(
                "claims",
                []
            )

            if isinstance(claims, list):

                cleaned_claims = []

                for item in claims:

                    if not isinstance(item, dict):
                        continue

                    claim = str(
                        item.get("claim", "")
                    ).strip()

                    if not claim:
                        continue

                    context = str(
                        item.get("context", "")
                    ).strip()

                    cleaned_claims.append({
                        "claim": claim,
                        "quote": context or claim,
                        "context": context or claim,
                        "location": item.get(
                            "location",
                            "unknown"
                        )
                    })

                return cleaned_claims

    except Exception as e:

        print(
            f"Granite extraction failed: {e}"
        )

    # Use fallback if Granite fails
    return extract_claims_rule_based(text)


# ============================================================
# RULE-BASED FALLBACK
# ============================================================

ENVIRONMENTAL_KEYWORDS = [

    "carbon",
    "co2",
    "emission",
    "emissions",
    "greenhouse gas",
    "ghg",
    "climate",
    "renewable",
    "solar",
    "wind energy",
    "sustainable",
    "sustainability",
    "recycl",
    "waste",
    "water",
    "energy efficiency",
    "energy consumption",
    "eco-friendly",
    "environment",
    "net zero",
    "carbon neutral",
    "plastic",
    "landfill",
    "biodiversity"
]


def extract_claims_rule_based(
    text: str
) -> List[Dict]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for sentence in sentences:

        sentence_clean = sentence.strip()

        if len(sentence_clean) < 20:
            continue

        lower = sentence_clean.lower()

        if any(
            keyword in lower
            for keyword in ENVIRONMENTAL_KEYWORDS
        ):

            claims.append({
                "claim": sentence_clean,
                "quote": sentence_clean,
                "context": sentence_clean,
                "location": "unknown"
            })

    return claims


# ============================================================
# MAIN CLAIM EXTRACTION
# ============================================================

def extract_claims(
    text: str,
    use_llm: bool = True
) -> List[Dict]:

    text = clean_text(text)

    if not text:
        return []

    chunks = chunk_text(text)

    all_claims = []

    for chunk in chunks:

        if use_llm:

            claims = extract_claims_with_granite(
                chunk
            )

        else:

            claims = extract_claims_rule_based(
                chunk
            )

        all_claims.extend(claims)

    # Remove duplicates
    unique_claims = []

    seen = set()

    for claim in all_claims:

        claim_text = claim.get(
            "claim",
            ""
        ).strip()

        if not claim_text:
            continue

        key = claim_text.lower()

        if key not in seen:

            seen.add(key)

            unique_claims.append(
                claim
            )

    return unique_claims
