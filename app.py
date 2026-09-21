import os
import pandas as pd
import streamlit as st
from extraction import extract_pdf_text, extract_claims, clean_text
from classifier import classify_claim, calculate_transparency_score
from rag import RubricRAG

st.set_page_config(page_title="GreenClaim — Sustainability Claim Checker",page_icon="🌱",layout="wide")

def sample():
    try: return open("sample_report.txt","r",encoding="utf-8").read()
    except: return "We reduced Scope 1 and Scope 2 emissions by 32% from our 2020 baseline across our global manufacturing operations."

@st.cache_resource
def load_rag(): return RubricRAG()

def dataframe(claims,results):
    return pd.DataFrame([{
        "Claim #":i,"Claim":c.get("claim",""),"Status":r.get("status","Needs more evidence"),
        "Specificity":r["specificity"].get("score",0),"Measurability":r["measurability"].get("score",0),
        "Evidence":r["evidence"].get("score",0),"Scope":r["scope_clarity"].get("score",0),
        "Source Quote":r.get("source_quote",c.get("quote",""))
    } for i,(c,r) in enumerate(zip(claims,results),1)])

def main():
    st.title("🌱 GreenClaim")
    st.subheader("AI-Powered Sustainability Claim Verification Assistant")
    st.markdown("Analyze environmental claims in sustainability reports and identify where additional evidence or clarification may be needed. **This tool evaluates evidence completeness in the supplied document; it does not determine whether a company is guilty of greenwashing, fraud, or deception.**")
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider=os.getenv("LLM_PROVIDER","huggingface")
        available=bool(os.getenv("HF_TOKEN")) if provider=="huggingface" else bool(os.getenv("OPENAI_API_KEY"))
        st.success("LLM available") if available else st.info("No LLM API key — deterministic fallback active")
        st.caption("RAG: Sentence Transformers + FAISS")
        st.divider(); st.markdown("**Score:** Specificity 25 • Measurability 25 • Evidence 30 • Scope 20")
    st.header("1. Provide a sustainability report")
    a,b=st.columns(2)
    with a: uploaded=st.file_uploader("Upload PDF",type=["pdf"])
    with b:
        if st.button("📄 Load example",use_container_width=True): st.session_state["report_text"]=sample()
    text=st.text_area("Or paste report text",value=st.session_state.get("report_text",""),height=280)
    if uploaded:
        try:
            extracted=extract_pdf_text(uploaded.read())
            if extracted: text=extracted; st.success(f"Extracted {len(extracted):,} characters.")
            else: st.warning("No selectable text found. This may be a scanned PDF.")
        except Exception as e: st.error(f"Could not extract PDF text: {e}")
    if not st.button("🔍 Analyze Sustainability Claims",type="primary",use_container_width=True):
        st.info("Upload a report, paste text, or click Load example."); return
    text=clean_text(text)
    if not text: st.error("Please upload a PDF or provide report text."); return
    with st.spinner("Loading rubric and analyzing claims..."):
        try:
            rag=load_rag(); claims=extract_claims(text,use_llm=True)
            if not claims: st.warning("No environmental claims were detected."); return
            results=[]; progress=st.progress(0)
            for i,c in enumerate(claims):
                ctx=rag.context(c.get("claim","")+" "+c.get("quote",""),top_k=4)
                results.append(classify_claim(c.get("claim",""),c.get("quote",""),ctx,use_llm=True))
                progress.progress((i+1)/len(claims))
            progress.empty()
        except Exception as e: st.error(f"Analysis failed: {e}"); return
    score=calculate_transparency_score(results)
    st.header("2. Transparency overview")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Transparency Score",f"{score['score']}/100"); m2.metric("Claims Detected",len(claims))
    counts={"Well-supported":0,"Needs more evidence":0,"Vague/Unsupported":0}
    for r in results:
        if r.get("status") in counts: counts[r["status"]]+=1
    m3.metric("Well-supported",counts["Well-supported"]); m4.metric("Needs Review",counts["Needs more evidence"]+counts["Vague/Unsupported"])
    st.progress(min(score["score"]/100,1.0)); st.caption("Evidence-completeness indicator, not a company rating.")
    st.subheader("Score breakdown")
    bd=pd.DataFrame({"Dimension":["Specificity","Measurability","Evidence","Scope clarity"],"Score":[score["specificity"],score["measurability"],score["evidence"],score["scope"]]})
    st.bar_chart(bd.set_index("Dimension")["Score"])
    st.header("3. Claim-level results")
    df=dataframe(claims,results); st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download CSV",df.to_csv(index=False).encode("utf-8"),"greenclaim_results.csv","text/csv")
    st.header("4. Detailed claim analysis")
    for i,(c,r) in enumerate(zip(claims,results),1):
        with st.expander(f"Claim {i}: {r.get('status','Needs more evidence')}"):
            st.markdown(f"### {c.get('claim','')}")
            st.markdown("**Source quote**"); st.info(r.get("source_quote",c.get("quote","")))
            st.markdown(f"**Status:** `{r.get('status','Needs more evidence')}`"); st.write(r.get("reason",""))
            x1,x2,x3,x4=st.columns(4)
            x1.metric("Specificity",f"{r['specificity'].get('score',0)}/25")
            x2.metric("Measurability",f"{r['measurability'].get('score',0)}/25")
            x3.metric("Evidence",f"{r['evidence'].get('score',0)}/30")
            x4.metric("Scope",f"{r['scope_clarity'].get('score',0)}/20")
            st.write("**Specificity:**",r["specificity"].get("reason",""))
            st.write("**Measurability:**",r["measurability"].get("reason",""))
            st.write("**Evidence:**",r["evidence"].get("reason",""))
            if r["evidence"].get("items_found"): st.write("Evidence indicators found:",", ".join(r["evidence"]["items_found"]))
            if r["evidence"].get("items_missing"): st.write("Information that may need clarification:",", ".join(r["evidence"]["items_missing"]))
            st.write("**Scope clarity:**",r["scope_clarity"].get("reason",""))
    st.header("5. Responsible AI")
    st.info("GreenClaim does not determine whether a company is engaging in greenwashing. It identifies environmental claims and evaluates the completeness of supporting information contained in the supplied document. A flagged claim means additional evidence, clarification, or human review may be appropriate.")

if __name__=="__main__": main()
