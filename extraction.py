import json, os, re
from pathlib import Path
from typing import List, Dict, Optional
import fitz

ENVIRONMENTAL_TERMS = [
    "carbon","emission","emissions","greenhouse gas","ghg","climate","net zero",
    "carbon neutral","renewable","energy","water","waste","recycl","sustainable",
    "sustainability","eco-friendly","environment","environmental","biodiversity",
    "plastic","packaging","deforestation","landfill","pollution","scope 1","scope 2","scope 3"
]

def extract_pdf_text(pdf_bytes: bytes) -> str:
    pages=[]
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for n,page in enumerate(document,1):
            text=page.get_text("text")
            if text.strip(): pages.append(f"\n--- Page {n} ---\n{text.strip()}")
    return "\n".join(pages).strip()

def clean_text(text):
    text=text.replace("\x00"," ")
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def chunk_text(text,max_chars=7000,overlap=500):
    text=clean_text(text)
    if len(text)<=max_chars: return [text]
    paragraphs=text.split("\n\n"); chunks=[]; current=""
    for p in paragraphs:
        p=p.strip()
        if not p: continue
        candidate=f"{current}\n\n{p}".strip() if current else p
        if len(candidate)<=max_chars: current=candidate
        else:
            if current: chunks.append(current)
            if len(p)<=max_chars:
                tail=current[-overlap:] if current else ""
                current=f"{tail}\n\n{p}".strip() if tail else p
            else:
                start=0
                while start<len(p):
                    end=min(start+max_chars,len(p))
                    chunks.append(p[start:end].strip())
                    start=end-overlap if end<len(p) else end
                current=""
    if current: chunks.append(current)
    return chunks

def _looks_environmental(text):
    lower=text.lower()
    return any(term in lower for term in ENVIRONMENTAL_TERMS)

def rule_based_claim_extraction(text):
    sentences=re.split(r"(?<=[.!?])\s+|\n+",text)
    claims=[]
    for s in sentences:
        s=s.strip()
        if len(s)>=20 and _looks_environmental(s):
            claims.append({"claim":s,"quote":s,"source":"Rule-based extraction"})
    unique=[]; seen=set()
    for c in claims:
        k=c["claim"].lower()
        if k not in seen: seen.add(k); unique.append(c)
    return unique

EXTRACTION_PROMPT = r"""You are an environmental sustainability document analyst.
Identify explicit environmental or sustainability claims from the supplied corporate report.
A claim communicates environmental performance, impact, target, benefit, certification, reduction,
neutrality, sustainability characteristic, or environmental commitment.
Return ONLY valid JSON.
Schema:
{"claims":[{"claim":"short normalized description","quote":"exact source quote","location":"page/section/unknown","claim_type":"performance|target|product|packaging|certification|commitment|other"}]}
Rules: preserve exact quotes; never invent evidence; do not improve quotes; extract distinct claims;
if none return {"claims":[]}."""

def _extract_json(text):
    try: return json.loads(text.strip())
    except: pass
    m=re.search(r"\{.*\}",text,re.S)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None

def llm_chat(prompt):
    provider=os.getenv("LLM_PROVIDER","huggingface").lower()
    try:
        if provider=="huggingface":
            from huggingface_hub import InferenceClient
            token=os.getenv("HF_TOKEN")
            if not token: return None
            model=os.getenv("HF_MODEL","ibm-granite/granite-3.3-2b-instruct")
            client=InferenceClient(model=model,token=token)
            r=client.chat_completion(messages=[{"role":"user","content":prompt}],max_tokens=3000,temperature=0.1)
            return r.choices[0].message.content
        if provider=="openai":
            import requests
            key=os.getenv("OPENAI_API_KEY")
            if not key: return None
            base=os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/")
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini")
            r=requests.post(f"{base}/chat/completions",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                json={"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.1},timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None
    return None

def _deduplicate_claims(claims):
    out=[]; seen=set()
    for c in claims:
        q=c.get("quote","").strip(); k=re.sub(r"\s+"," ",q.lower())
        if k and k not in seen: seen.add(k); out.append(c)
    return out

def extract_claims(text,use_llm=True,max_chunks=12):
    text=clean_text(text)
    if not text: return []
    if use_llm:
        all_claims=[]
        for chunk in chunk_text(text)[:max_chunks]:
            raw=llm_chat(EXTRACTION_PROMPT+"\n\nREPORT TEXT:\n"+chunk)
            parsed=_extract_json(raw) if raw else None
            if parsed:
                for c in parsed.get("claims",[]):
                    if c.get("quote"):
                        c["source"]="LLM extraction"; all_claims.append(c)
        if all_claims: return _deduplicate_claims(all_claims)
    return rule_based_claim_extraction(text)
