"""
AI Resume Analyzer & ATS Scorer
================================
Built with: Streamlit · spaCy · scikit-learn (TF-IDF) · pdfplumber · python-docx
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.parser import parse_resume
from utils.scorer import (
    compute_ats_score,
    get_score_breakdown,
    keyword_gap_analysis,
    generate_suggestions,
    score_label,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Analyzer & ATS Scorer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #1A1D2E;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .metric-value { font-size: 3rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; color: #aaa; margin-top: 4px; }
    .badge-matched {
        display: inline-block; background: #1e4d2b; color: #2ecc71;
        border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: 0.82rem;
    }
    .badge-missing {
        display: inline-block; background: #4d1e1e; color: #e74c3c;
        border-radius: 20px; padding: 3px 12px; margin: 3px; font-size: 0.82rem;
    }
    .tip-box {
        background: #1A1D2E; border-left: 4px solid #4A90D9;
        padding: 10px 16px; border-radius: 6px; margin: 6px 0;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] { font-size: 2rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar – inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Resume Analyzer")
    st.markdown("**Upload your resume and paste the job description to get your ATS score.**")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Resume", type=["pdf", "docx", "txt"],
        help="Supported: PDF, DOCX, TXT"
    )

    st.markdown("##### Job Description")
    jd_text = st.text_area(
        "Paste the job description here",
        height=280,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning, "
                    "REST APIs, AWS, and agile methodologies…",
    )

    analyze_btn = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")
    st.divider()
    st.caption("Built with Streamlit · spaCy · scikit-learn")


# ---------------------------------------------------------------------------
# Hero section (when nothing is uploaded yet)
# ---------------------------------------------------------------------------
if not uploaded_file and not analyze_btn:
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 20px 20px'>
            <h1>🤖 AI Resume Analyzer & ATS Scorer</h1>
            <p style='font-size:1.1rem; color:#aaa;'>
                Upload your resume and a job description to instantly get your ATS compatibility score,
                keyword gap analysis, and actionable improvement tips powered by NLP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1**\nUpload your resume (PDF / DOCX / TXT) in the sidebar.")
    with col2:
        st.info("📋 **Step 2**\nPaste the target job description in the sidebar.")
    with col3:
        st.info("📊 **Step 3**\nClick **Analyze Resume** to get your ATS score & insights.")
    st.stop()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if analyze_btn:
    if not uploaded_file:
        st.error("Please upload a resume file.", icon="⚠️")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.", icon="⚠️")
        st.stop()

    # ------------------------------------------------------------------
    # Parse & score
    # ------------------------------------------------------------------
    with st.spinner("Parsing resume and computing ATS score…"):
        file_bytes = uploaded_file.read()
        parsed = parse_resume(file_bytes, uploaded_file.name)
        resume_text = parsed["raw_text"]

        breakdown = get_score_breakdown(resume_text, jd_text)
        ats_score = compute_ats_score(resume_text, jd_text)
        gap = keyword_gap_analysis(resume_text, jd_text, top_n=40)
        suggestions = generate_suggestions(resume_text, gap["missing"])
        label, color = score_label(ats_score)

    # ------------------------------------------------------------------
    # Header row
    # ------------------------------------------------------------------
    st.markdown(f"## 📊 Analysis Results — *{uploaded_file.name}*")
    st.divider()

    # ------------------------------------------------------------------
    # Score breakdown
    # ------------------------------------------------------------------
    st.markdown("#### 🧮 Weighted ATS Breakdown")
    st.caption(
        "Final score uses: 40% keyword match + 30% skill match + "
        "20% semantic similarity + 10% resume structure"
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Keyword Match (40%)", f"{breakdown['keyword_match']}%")
    b2.metric("Skill Match (30%)", f"{breakdown['skill_match']}%")
    b3.metric("Semantic (20%)", f"{breakdown['semantic_similarity']}%")
    b4.metric("Structure (10%)", f"{breakdown['resume_structure']}%")

    st.caption(f"Semantic engine: {breakdown['semantic_method']}")
    st.divider()

    # ------------------------------------------------------------------
    # Top metrics
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{ats_score}%</div>
                <div class="metric-label">ATS Score</div>
                <div style="font-size:0.8rem;color:{color};margin-top:4px">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#2ecc71">{len(gap['matched'])}</div>
                <div class="metric-label">Keywords Matched</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#e74c3c">{len(gap['missing'])}</div>
                <div class="metric-label">Keywords Missing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        exp = parsed.get("experience_years", 0)
        exp_display = f"{exp} yrs" if exp else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#4A90D9">{exp_display}</div>
                <div class="metric-label">Experience Detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # ATS gauge chart
    # ------------------------------------------------------------------
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        st.markdown("#### ATS Compatibility Gauge")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=ats_score,
                delta={"reference": 60, "increasing": {"color": "#2ecc71"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#aaa"},
                    "bar": {"color": color},
                    "bgcolor": "#1A1D2E",
                    "bordercolor": "#333",
                    "steps": [
                        {"range": [0, 20],  "color": "#2d0000"},
                        {"range": [20, 40], "color": "#4d1e00"},
                        {"range": [40, 60], "color": "#4d3d00"},
                        {"range": [60, 80], "color": "#1e3d1e"},
                        {"range": [80, 100], "color": "#003d1e"},
                    ],
                    "threshold": {
                        "line": {"color": "#fff", "width": 3},
                        "thickness": 0.75,
                        "value": 60,
                    },
                },
                number={"suffix": "%", "font": {"size": 40, "color": color}},
                title={"text": label, "font": {"size": 18, "color": "#aaa"}},
            )
        )
        fig.update_layout(
            paper_bgcolor="#0E1117",
            font_color="#FAFAFA",
            height=320,
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown("#### Candidate Profile")
        info_items = {
            "👤 Name":     parsed.get("name", "—"),
            "📧 Email":    parsed.get("email", "—") or "—",
            "📞 Phone":    parsed.get("phone", "—") or "—",
            "🔗 LinkedIn": parsed.get("linkedin", "—") or "—",
            "💻 GitHub":   parsed.get("github", "—") or "—",
        }
        for label_txt, val in info_items.items():
            st.markdown(f"**{label_txt}:** {val}")

        if parsed.get("education"):
            st.markdown("**🎓 Education:**")
            for edu in parsed["education"][:4]:
                st.markdown(f"  - {edu}")

        if parsed.get("skills"):
            st.markdown(f"**🛠 Skills found ({len(parsed['skills'])}):**")
            skill_badges = " ".join(
                f'<span class="badge-matched">{s}</span>'
                for s in sorted(parsed["skills"])[:20]
            )
            st.markdown(skill_badges, unsafe_allow_html=True)

    st.divider()

    # ------------------------------------------------------------------
    # Keyword gap analysis
    # ------------------------------------------------------------------
    st.markdown("#### 🔑 Keyword Gap Analysis")
    st.caption(
        f"Analyzed top **{len(gap['jd_keywords'])}** keywords from the job description — "
        f"**{gap['match_pct']}%** match rate."
    )

    kw_col1, kw_col2 = st.columns(2)

    with kw_col1:
        st.markdown("**✅ Matched Keywords**")
        if gap["matched"]:
            badges = " ".join(
                f'<span class="badge-matched">{kw}</span>' for kw in sorted(gap["matched"])
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.warning("No keyword matches found.")

    with kw_col2:
        st.markdown("**❌ Missing Keywords**")
        if gap["missing"]:
            badges = " ".join(
                f'<span class="badge-missing">{kw}</span>' for kw in sorted(gap["missing"])
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.success("No missing keywords — great coverage!")

    # Keyword match bar chart
    if gap["jd_keywords"]:
        st.markdown("<br>", unsafe_allow_html=True)
        df_kw = pd.DataFrame(
            [
                {"Keyword": kw, "Status": "Matched", "Value": 1}
                for kw in gap["matched"]
            ]
            + [
                {"Keyword": kw, "Status": "Missing", "Value": 1}
                for kw in gap["missing"][:20]  # cap for readability
            ]
        )
        matched_count = len(gap["matched"])
        missing_count = len(gap["missing"])

        fig2 = go.Figure(
            data=[
                go.Bar(name="Matched", x=["Keywords"], y=[matched_count],
                       marker_color="#2ecc71", text=[matched_count],
                       textposition="outside"),
                go.Bar(name="Missing", x=["Keywords"], y=[missing_count],
                       marker_color="#e74c3c", text=[missing_count],
                       textposition="outside"),
            ]
        )
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="#FAFAFA",
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Improvement suggestions
    # ------------------------------------------------------------------
    st.markdown("#### 💡 Improvement Suggestions")
    if suggestions:
        for tip in suggestions:
            st.markdown(f'<div class="tip-box">💡 {tip}</div>', unsafe_allow_html=True)
    else:
        st.success("Your resume looks well-optimized for this job description!")

    st.divider()

    # ------------------------------------------------------------------
    # Raw extracted text (expandable)
    # ------------------------------------------------------------------
    with st.expander("📄 View Extracted Resume Text"):
        st.text_area("Resume content", value=parsed["raw_text"], height=300, disabled=True)

    # ------------------------------------------------------------------
    # Download report
    # ------------------------------------------------------------------
    report_lines = [
        "AI Resume Analyzer — ATS Report",
        "=" * 40,
        f"File          : {uploaded_file.name}",
        f"ATS Score     : {ats_score}% ({label})",
        f"Keyword Match : {breakdown['keyword_match']}% (weight 40%)",
        f"Skill Match   : {breakdown['skill_match']}% (weight 30%)",
        f"Semantic Sim  : {breakdown['semantic_similarity']}% (weight 20%)",
        f"Structure     : {breakdown['resume_structure']}% (weight 10%)",
        f"Semantic Engine: {breakdown['semantic_method']}",
        f"Keyword Gap Coverage : {gap['match_pct']}%",
        f"Matched KWs   : {', '.join(gap['matched']) or 'None'}",
        f"Missing KWs   : {', '.join(gap['missing']) or 'None'}",
        "",
        "Improvement Suggestions",
        "-" * 40,
    ] + [f"• {s}" for s in suggestions]

    st.download_button(
        label="⬇️ Download ATS Report (.txt)",
        data="\n".join(report_lines),
        file_name="ats_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
