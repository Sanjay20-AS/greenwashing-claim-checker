import json, re
from typing import Dict, List, Optional
from extraction import llm_chat

CLASSIFICATION_PROMPT = r"""You are an environmental claim transparency analyst.
Evaluate ONE environmental sustainability claim using ONLY the claim, source quote, and supplied rubric.
Do not independently verify truth. Evaluate evidence and specificity in the supplied document.

Dimensions:
1. Specificity: specific vs vague environmental language.
2. Measurability: numbers, baseline year, target year, measurement period.
3. Evidence: data, methodology, audit, third-party assurance, GHG Protocol, SBTi, ISO 14001.
4. Scope clarity: company/product boundary, geography, operations, Scope 1/2/3.

Status:
Well-supported = substantial evidence and clear scope.
Needs more evidence = reasonably specific but important evidence missing.
Vague/Unsupported = broad/vague and little supporting information.

IMPORTANT: Vague/Unsupported means insufficient evidence in the document. It does NOT mean the company is lying or committing fraud.

Return ONLY valid JSON:
{"status":"Well-supported | Needs more evidence | Vague/Unsupported","reason":"short neutral explanation",
"specificity":{"score":0,"label":"specific | vague","reason":"..."},
"measurability":{"score":0,"label":"high | medium | low","reason":"..."},
"evidence":{"score":0,"label":"strong | partial | weak","items_found":[],"items_missing":[],"reason":"..."},
"scope_clarity":{"score":0,"label":"clear | partial | unclear","reason":"..."},
"source_quote":"exact supplied quote"}"""

def _parse_json(raw):
    try: return json.loads(raw.strip())
    except: pass
    m=re.search(r"\{.*\}",raw,re.S)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None

def _contains_number(t): return bool(re.search(r"\b\d+(?:\.\d+)?\s*%?|\b\d{4}\b",t))
def _contains_baseline(t):
    l=t.lower()
    return any(x in l for x in ["baseline","base year","from 2020","from 2019","from 2021","from 2022","from 2023","from 2024","from 2025"])
def _contains_target_year(t): return bool(re.search(r"\b20(?:2[6-9]|3[0-9])\b",t))
def _contains_scope(t):
    l=t.lower()
    return any(x in l for x in ["scope 1","scope 2","scope 3","company-wide","company wide","operations","our facilities","product","packaging"])
def _contains_evidence(t):
    l=t.lower()
    return any(x in l for x in ["verified","verification","third-party","third party","assurance","audit","audited","ghg protocol","sbti","science based targets","iso 14001","methodology","measured","independently assured","certified"])

def rule_based_classification(claim,quote,rag_context=""):
    text=f"{claim} {quote}"; l=text.lower()
    vague_terms=["eco-friendly","eco friendly","green","planet-friendly","planet friendly","sustainable","greener","clean","environmentally friendly","better for the planet","responsible"]
    vague=any(x in l for x in vague_terms)
    number=_contains_number(text); baseline=_contains_baseline(text); target=_contains_target_year(text)
    scope=_contains_scope(text); evidence=_contains_evidence(text)
    specificity=8 if vague else 23
    meas=(12 if number else 0)+(7 if baseline else 0)+(6 if target else 0); meas=min(25,meas)
    found=[]
    if number: found.append("quantitative information")
    if baseline: found.append("baseline information")
    if target: found.append("target timeline")
    if evidence: found.append("methodology/verification/standard reference")
    ev=(12 if number else 0)+(12 if evidence else 0)+(4 if baseline else 0); ev=min(30,ev)
    scope_score=20 if scope else 6
    total=specificity+meas+ev+scope_score
    status="Vague/Unsupported" if vague and total<55 else ("Well-supported" if total>=75 else "Needs more evidence")
    missing=[]
    if not number: missing.append("quantitative measurement")
    if not baseline: missing.append("baseline/reference period")
    if not target and any(x in l for x in ["target","aim","commit","neutral","net zero"]): missing.append("target timeline")
    if not evidence: missing.append("methodology or verification evidence")
    if not scope: missing.append("clear scope/boundary")
    reason=("The claim uses broad environmental language and provides limited measurable or supporting information in the supplied document." if vague else
            "The claim contains measurable information and multiple supporting or scope-related details in the supplied document." if status=="Well-supported" else
            "The claim is reasonably specific, but the supplied document does not provide all important evidence needed for verification.")
    return {"status":status,"reason":reason,
        "specificity":{"score":specificity,"label":"vague" if vague else "specific","reason":"Broad environmental wording is used without enough definition." if vague else "The environmental subject is relatively specific."},
        "measurability":{"score":meas,"label":"high" if meas>=20 else "medium" if meas>=10 else "low","reason":"The claim contains quantitative and/or time-based information." if meas>=10 else "Few measurable details are present."},
        "evidence":{"score":ev,"label":"strong" if ev>=22 else "partial" if ev>=10 else "weak","items_found":found,"items_missing":missing,"reason":"Several supporting indicators are present." if ev>=22 else "Some supporting information is present, but important details remain unavailable."},
        "scope_clarity":{"score":scope_score,"label":"clear" if scope else "unclear","reason":"The claim identifies a recognizable scope or boundary." if scope else "The claim does not clearly identify the relevant scope."},
        "source_quote":quote,"rag_context_used":bool(rag_context)}

def classify_claim(claim,quote,rag_context="",use_llm=True):
    if use_llm:
        raw=llm_chat(CLASSIFICATION_PROMPT+"\n\nRUBRIC CONTEXT:\n"+rag_context+"\n\nCLAIM:\n"+claim+"\n\nSOURCE QUOTE:\n"+quote)
        parsed=_parse_json(raw) if raw else None
        if parsed:
            parsed["source_quote"]=quote; parsed["rag_context_used"]=bool(rag_context)
            return parsed
    return rule_based_classification(claim,quote,rag_context)

def calculate_transparency_score(results):
    if not results: return {"score":0,"specificity":0,"measurability":0,"evidence":0,"scope":0}
    avg=lambda key: sum(float(r[key].get("score",0)) for r in results)/len(results)
    s,m,e,sc=avg("specificity"),avg("measurability"),avg("evidence"),avg("scope_clarity")
    return {"score":round(max(0,min(100,s+m+e+sc)),1),"specificity":round(s,1),"measurability":round(m,1),"evidence":round(e,1),"scope":round(sc,1)}
