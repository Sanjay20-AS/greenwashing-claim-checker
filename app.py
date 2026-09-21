import os

import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from extraction import (
    extract_pdf_text,
    extract_claims,
    clean_text
)

from classifier import (
    classify_claim,
    calculate_transparency_score
)

from rag import RubricRAG


load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GreenClaim — Sustainability Claim Checker",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# SAMPLE REPORT
# ============================================================

def sample():

    try:

        with open(
            "sample_report.txt",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except Exception:

        return """
Our company reduced Scope 1 and Scope 2 emissions by 32%
from our 2020 baseline across our global manufacturing operations.

We use renewable energy wherever possible.

Our products are environmentally friendly and good for the planet.

We reduced plastic usage by 25% in 2025.

Our manufacturing facilities achieved a 40% reduction
in water consumption compared with 2021.

We are committed to becoming net zero.

Our packaging is made using sustainable materials.

A third-party audit verified our reported emissions data.

We recycle 85% of production waste across our facilities.

Our energy efficiency initiatives reduced electricity
consumption by 18% during 2025.
"""


# ============================================================
# RAG
# ============================================================

@st.cache_resource
def load_rag():

    return RubricRAG()


# ============================================================
# RESULTS DATAFRAME
# ============================================================

def dataframe(
    claims,
    results
):

    rows = []

    for i, (claim, result) in enumerate(
        zip(
            claims,
            results
        ),
        1
    ):

        rows.append({

            "Claim #": i,

            "Claim": claim.get(
                "claim",
                ""
            ),

            "Status": result.get(
                "status",
                "Needs more evidence"
            ),

            "Specificity": result.get(
                "specificity",
                {}
            ).get(
                "score",
                0
            ),

            "Measurability": result.get(
                "measurability",
                {}
            ).get(
                "score",
                0
            ),

            "Evidence": result.get(
                "evidence",
                {}
            ).get(
                "score",
                0
            ),

            "Scope": result.get(
                "scope_clarity",
                {}
            ).get(
                "score",
                0
            ),

            "Source Quote": result.get(
                "source_quote",
                claim.get(
                    "quote",
                    ""
                )
            )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    st.title(
        "🌱 GreenClaim"
    )

    st.subheader(
        "AI-Powered Sustainability Claim Verification Assistant"
    )

    st.markdown(
        """
Analyze environmental claims in sustainability reports and
identify where additional evidence or clarification may be needed.

**This tool evaluates evidence completeness in the supplied
document; it does not determine whether a company is guilty
of greenwashing, fraud, or deception.**
"""
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "⚙️ Configuration"
        )

        provider = os.getenv(
            "LLM_PROVIDER",
            "huggingface"
        )

        hf_token = os.getenv(
            "HF_TOKEN"
        )

        if provider == "huggingface":

            if hf_token:

                st.success(
                    "IBM Granite + Hugging Face available"
                )

                st.caption(
                    f"Model: {os.getenv('HF_MODEL', 'ibm-granite/granite-4.2-3b')}"
                )

                st.caption(
                    f"Provider: {os.getenv('HF_PROVIDER', 'deepinfra')}"
                )

            else:

                st.warning(
                    "HF_TOKEN not configured — "
                    "deterministic fallback may be used"
                )

        else:

            st.info(
                "Rule-based fallback available"
            )

        st.caption(
            "RAG: Sentence Transformers + FAISS"
        )

        st.divider()

        st.markdown(
            """
**Score**

Specificity 25 •
Measurability 25 •
Evidence 30 •
Scope 20
"""
        )


    # ========================================================
    # REPORT INPUT
    # ========================================================

    st.header(
        "1. Provide a sustainability report"
    )

    a, b = st.columns(
        2
    )

    with a:

        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

    with b:

        if st.button(
            "📄 Load example",
            use_container_width=True
        ):

            st.session_state[
                "report_text"
            ] = sample()


    text = st.text_area(
        "Or paste report text",
        value=st.session_state.get(
            "report_text",
            ""
        ),
        height=280
    )


    # ========================================================
    # PDF
    # ========================================================

    if uploaded:

        try:

            extracted = extract_pdf_text(
                uploaded.read()
            )

            if extracted:

                text = extracted

                st.success(
                    f"Extracted {len(extracted):,} characters."
                )

            else:

                st.warning(
                    "No selectable text found. "
                    "This may be a scanned PDF."
                )

        except Exception as e:

            st.error(
                f"Could not extract PDF text: {e}"
            )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if not st.button(
        "🔍 Analyze Sustainability Claims",
        type="primary",
        use_container_width=True
    ):

        st.info(
            "Upload a report, paste report text, "
            "or click Load example."
        )

        return


    text = clean_text(
        text
    )


    if not text:

        st.error(
            "Please upload a PDF or provide report text."
        )

        return


    # ========================================================
    # ANALYSIS
    # ========================================================

    with st.spinner(
        "Loading rubric and analyzing claims..."
    ):

        try:

            rag = load_rag()

            claims = extract_claims(
                text,
                use_llm=True
            )


            if not claims:

                st.warning(
                    "No environmental claims were detected."
                )

                return


            results = []

            progress = st.progress(
                0
            )


            for i, claim in enumerate(
                claims
            ):

                ctx = rag.context(
                    claim.get(
                        "claim",
                        ""
                    )
                    + " "
                    + claim.get(
                        "quote",
                        ""
                    ),
                    top_k=4
                )


                result = classify_claim(

                    claim.get(
                        "claim",
                        ""
                    ),

                    claim.get(
                        "quote",
                        ""
                    ),

                    ctx,

                    use_llm=True
                )


                results.append(
                    result
                )


                progress.progress(
                    (i + 1)
                    / len(claims)
                )


            progress.empty()


        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )

            return


    # ========================================================
    # TRANSPARENCY SCORE
    # ========================================================

    score = calculate_transparency_score(
        results
    )


    st.header(
        "2. Transparency overview"
    )


    m1, m2, m3, m4 = st.columns(
        4
    )


    m1.metric(
        "Transparency Score",
        f"{score['score']}/100"
    )


    m2.metric(
        "Claims Detected",
        len(claims)
    )


    counts = {

        "Well-supported": 0,

        "Needs more evidence": 0,

        "Vague/Unsupported": 0
    }


    for result in results:

        status = result.get(
            "status"
        )

        if status in counts:

            counts[status] += 1


    m3.metric(
        "Well-supported",
        counts["Well-supported"]
    )


    m4.metric(
        "Needs Review",
        counts["Needs more evidence"]
        + counts["Vague/Unsupported"]
    )


    st.progress(
        min(
            score["score"]
            / 100,
            1.0
        )
    )


    st.caption(
        "Evidence-completeness indicator, "
        "not a company rating."
    )


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.subheader(
        "Score breakdown"
    )


    bd = pd.DataFrame({

        "Dimension": [

            "Specificity",
            "Measurability",
            "Evidence",
            "Scope clarity"
        ],

        "Score": [

            score["specificity"],
            score["measurability"],
            score["evidence"],
            score["scope"]
        ]
    })


    st.bar_chart(
        bd.set_index(
            "Dimension"
        )["Score"]
    )


    # ========================================================
    # CLAIM TABLE
    # ========================================================

    st.header(
        "3. Claim-level results"
    )


    df = dataframe(
        claims,
        results
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(

        "⬇️ Download CSV",

        df.to_csv(
            index=False
        ).encode(
            "utf-8"
        ),

        "greenclaim_results.csv",

        "text/csv"
    )


    # ========================================================
    # DETAILED RESULTS
    # ========================================================

    st.header(
        "4. Detailed claim analysis"
    )


    for i, (claim, result) in enumerate(
        zip(
            claims,
            results
        ),
        1
    ):

        with st.expander(
            f"Claim {i}: "
            f"{result.get('status', 'Needs more evidence')}"
        ):

            st.markdown(
                f"### {claim.get('claim', '')}"
            )


            st.markdown(
                "**Source quote**"
            )


            st.info(
                result.get(
                    "source_quote",
                    claim.get(
                        "quote",
                        ""
                    )
                )
            )


            st.markdown(
                f"**Status:** "
                f"`{result.get('status', 'Needs more evidence')}`"
            )


            st.write(
                result.get(
                    "reason",
                    ""
                )
            )


            x1, x2, x3, x4 = st.columns(
                4
            )


            x1.metric(
                "Specificity",
                f"{result.get('specificity', {}).get('score', 0)}/25"
            )


            x2.metric(
                "Measurability",
                f"{result.get('measurability', {}).get('score', 0)}/25"
            )


            x3.metric(
                "Evidence",
                f"{result.get('evidence', {}).get('score', 0)}/30"
            )


            x4.metric(
                "Scope",
                f"{result.get('scope_clarity', {}).get('score', 0)}/20"
            )


            st.write(
                "**Specificity:**",
                result.get(
                    "specificity",
                    {}
                ).get(
                    "reason",
                    ""
                )
            )


            st.write(
                "**Measurability:**",
                result.get(
                    "measurability",
                    {}
                ).get(
                    "reason",
                    ""
                )
            )


            st.write(
                "**Evidence:**",
                result.get(
                    "evidence",
                    {}
                ).get(
                    "reason",
                    ""
                )
            )


            evidence = result.get(
                "evidence",
                {}
            )


            if evidence.get(
                "items_found"
            ):

                st.write(
                    "Evidence indicators found:",
                    ", ".join(
                        evidence[
                            "items_found"
                        ]
                    )
                )


            if evidence.get(
                "items_missing"
            ):

                st.write(
                    "Information that may need clarification:",
                    ", ".join(
                        evidence[
                            "items_missing"
                        ]
                    )
                )


            st.write(
                "**Scope clarity:**",
                result.get(
                    "scope_clarity",
                    {}
                ).get(
                    "reason",
                    ""
                )
            )


    # ========================================================
    # RESPONSIBLE AI
    # ========================================================

    st.header(
        "5. Responsible AI"
    )


    st.info(
        """
GreenClaim does not determine whether a company is engaging
in greenwashing. It identifies environmental claims and
evaluates the completeness of supporting information contained
in the supplied document.

A flagged claim means additional evidence, clarification,
or human review may be appropriate.
"""
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
