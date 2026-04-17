import os
import io
import zipfile
from pathlib import Path
import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except Exception:
    PINGOUIN_AVAILABLE = False
STATS = PINGOUIN_AVAILABLE

try:
    from src.wildlife_nlp import (
        extract_wildlife_entities,
        predict_conservation_risk,
        train_conservation_risk_classifier,
    )
except Exception:
    class _SimpleRiskClassifier:
        def __init__(self, mode='fallback', accuracy=None):
            self.mode = mode
            self.accuracy = accuracy

    def extract_wildlife_entities(texts):
        lexicon = {
            'SPECIES': ['tiger','lion','jaguar','cheetah','serval','monkey','chimp','lemur','parrot','macaw','cockatoo','python','snake','iguana','turtle','tortoise','frog'],
            'LEGAL': ['illegal','law','permit','license','banned','regulated','cites'],
            'WELFARE': ['suffer','stress','cruel','captivity','neglect','abuse'],
            'CONSERVATION': ['endangered','extinction','ecosystem','wildlife','biodiversity','poaching','trafficking'],
            'SAFETY': ['bite','attack','danger','injury','disease','risk'],
            'TRADE': ['for sale','selling','breeder','shipping','contact','dm','telegram','whatsapp'],
        }
        counts = {}
        for text in texts:
            t = str(text).lower()
            for label, terms in lexicon.items():
                for term in terms:
                    if term in t:
                        counts[(term,label)] = counts.get((term,label),0)+1
        rows = [{'entity':k[0],'label':k[1],'count':v} for k,v in counts.items()]
        return pd.DataFrame(rows).sort_values(['count','entity'], ascending=[False,True]) if rows else pd.DataFrame(columns=['entity','label','count'])

    def train_conservation_risk_classifier(texts):
        return _SimpleRiskClassifier()

    def predict_conservation_risk(texts, classifier):
        hi = ['illegal','endangered','for sale','selling','trafficking','poaching','attack','danger','bite','cites']
        med = ['wildlife','permit','captivity','stress','shipping','breeder','risk']
        rows=[]
        for text in texts:
            t = str(text).lower()
            matched = [term for term in hi+med if term in t]
            score = sum(2 for term in hi if term in t) + sum(1 for term in med if term in t)
            label = 'high' if score >= 4 else 'medium' if score >= 2 else 'low'
            rows.append({'text': text, 'risk_score': float(score), 'risk_label': label, 'matched_terms': ', '.join(matched)})
        return pd.DataFrame(rows)

st.set_page_config(page_title="Unified Exotic Pet Research Platform", page_icon="🦜", layout="wide")

def apply_global_theme():
    st.markdown(
        """
        <style>
            html, body, [class*="css"]  {
                font-family: "Georgia", "Times New Roman", serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(123, 45, 55, 0.28), transparent 24%),
                    radial-gradient(circle at top right, rgba(84, 14, 78, 0.22), transparent 24%),
                    radial-gradient(circle at bottom center, rgba(35, 21, 48, 0.34), transparent 28%),
                    linear-gradient(180deg, #120b16 0%, #1a111f 35%, #24162b 100%);
                color: #f8f4f7;
            }

            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.5rem;
                max-width: 1400px;
            }

            .hero-card {
                padding: 1.4rem 1.6rem 1.1rem 1.6rem;
                border-radius: 22px;
                background: linear-gradient(180deg, rgba(27, 17, 33, 0.88), rgba(18, 11, 22, 0.82));
                border: 1px solid rgba(204, 163, 177, 0.18);
                box-shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
                backdrop-filter: blur(12px);
                margin-bottom: 1rem;
                text-align: center;
            }

            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.22em;
                font-size: 0.76rem;
                color: #d9b5c3;
                margin-bottom: 0.25rem;
                font-weight: 700;
            }

            .hero-title {
                font-size: 2.25rem;
                line-height: 1.1;
                font-weight: 800;
                color: #fff7fb;
                margin: 0.1rem 0 0.35rem 0;
                text-align: center;
            }

            .hero-subtitle {
                color: #f0dfe7;
                font-size: 0.98rem;
                max-width: 900px;
                margin: 0 auto;
                text-align: center;
            }

            h1, h2, h3, h4, h5, h6, p, li, label, span, div {
                color: inherit;
            }

            h1, h2, h3 {
                letter-spacing: 0.01em;
            }

            .stMarkdown, .stMarkdown p {
                color: #f5ebf0;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.6rem;
                background: rgba(20, 12, 24, 0.82);
                padding: 0.45rem;
                border-radius: 16px;
                border: 1px solid rgba(204, 163, 177, 0.14);
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 12px;
                padding: 0.55rem 1.15rem;
                font-weight: 600;
                color: #f4e8ee !important;
                border: 1px solid rgba(205, 171, 182, 0.12);
                background: linear-gradient(180deg, rgba(46, 30, 46, 0.9), rgba(26, 17, 26, 0.9));
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
                transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
            }

            .stTabs [data-baseweb="tab"]:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #8a294c 0%, #b33b67 100%) !important;
                color: white !important;
                border-color: rgba(255, 213, 227, 0.3);
            }

            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(36, 22, 43, 0.92), rgba(24, 15, 29, 0.92));
                border: 1px solid rgba(205, 171, 182, 0.16);
                padding: 0.9rem 1rem;
                border-radius: 16px;
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
            }

            div[data-testid="stDataFrame"], div[data-testid="stTable"] {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(205, 171, 182, 0.14);
                box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button,
            div[data-testid="stFormSubmitButton"] > button {
                border-radius: 999px !important;
                border: 1px solid rgba(205, 171, 182, 0.2) !important;
                background: linear-gradient(135deg, #6d1f3d 0%, #9a2f58 100%) !important;
                color: white !important;
                font-weight: 700 !important;
                padding: 0.55rem 1rem !important;
                box-shadow: 0 10px 24px rgba(109, 31, 61, 0.28);
                text-align: center !important;
            }

            div[data-testid="stDownloadButton"] > button:hover,
            div[data-testid="stButton"] > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 28px rgba(154, 47, 88, 0.34);
            }

            div[data-testid="stSelectbox"],
            div[data-testid="stMultiSelect"],
            div[data-testid="stTextInput"],
            div[data-testid="stNumberInput"],
            div[data-testid="stTextArea"] {
                border-radius: 14px;
            }

            .section-note {
                background: rgba(179, 59, 103, 0.10);
                padding: 0.9rem 1rem;
                border-radius: 14px;
                border-left: 4px solid #d9b5c3;
                margin: 0.5rem 0 1rem 0;
            }

            .metric-card {
                background: linear-gradient(180deg, rgba(36, 22, 43, 0.92), rgba(24, 15, 29, 0.92));
                padding: 1rem 1.2rem;
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.08);
                margin-bottom: 0.6rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #18101d 0%, #110b15 100%);
                border-right: 1px solid rgba(205, 171, 182, 0.12);
            }

            .stSidebar, .stSidebar label, .stSidebar p, .stSidebar span {
                color: #f7eef3 !important;
            }

            .stAlert {
                border-radius: 14px;
            }

            [data-testid="stMarkdownContainer"] p {
                text-align: left;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_global_theme()

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = APP_DIR / 'data'
ZIP_PATH = APP_DIR / 'data.zip'

@st.cache_data(show_spinner=False)
def prepare_data_dir() -> Path:
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    unzip_dir = APP_DIR / '_unzipped_data'
    if ZIP_PATH.exists():
        unzip_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(unzip_dir)
        inner_data = unzip_dir / 'data'
        if inner_data.exists():
            return inner_data
        return unzip_dir
    return DEFAULT_DATA_DIR

DATA_DIR = prepare_data_dir()

st.sidebar.title('Dashboard Selector')
dashboard_mode = st.sidebar.radio('Choose dashboard', ['WWF Final Data Dashboard', 'Uploaded / Experimental Research Dashboard'])



def render_wwf_dashboard():

    # =========================================================
    # PATHS
    # =========================================================
    APP_DIR = Path(__file__).resolve().parent
    DEFAULT_DATA_DIR = APP_DIR / "data"
    ZIP_PATH = APP_DIR / "data.zip"


    # =========================================================
    # DATA DISCOVERY
    # =========================================================
    def prepare_data_dir() -> Path:
        """
        Auto-detect data directory.
        Priority:
        1. ./data
        2. unzip ./data.zip into ./_unzipped_data and use inner /data if present
        """
        if DEFAULT_DATA_DIR.exists():
            return DEFAULT_DATA_DIR

        unzip_dir = APP_DIR / "_unzipped_data"
        if ZIP_PATH.exists():
            unzip_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(unzip_dir)

            inner_data = unzip_dir / "data"
            if inner_data.exists():
                return inner_data

            return unzip_dir

        return DEFAULT_DATA_DIR


    DATA_DIR = prepare_data_dir()


    # =========================================================
    # CACHE HELPERS
    # =========================================================
    @st.cache_data(show_spinner=False)
    def load_csv(name: str) -> pd.DataFrame | None:
        path = DATA_DIR / name
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None


    @st.cache_data(show_spinner=False)
    def list_available_files():
        if not DATA_DIR.exists():
            return []
        return sorted([p.name for p in DATA_DIR.iterdir() if p.is_file() and not p.name.startswith("._")])


    def exists(name: str) -> bool:
        return (DATA_DIR / name).exists()


    # =========================================================
    # SHARED THEME APPLIED GLOBALLY
    # =========================================================

    # =========================================================
    # DESCRIPTIONS
    # =========================================================
    SECTION_DESCRIPTIONS = {
        "overview": "The overview page summarizes the full study dashboard using the same analysis outputs produced in your original Python script. It presents the major themes, risk dimensions, and public sentiment toward exotic pet ownership in one landing view.",
        "sentiment": "This section follows your Question 1 logic and shows the overall sentiment distribution toward high-risk exotic pet ownership using the VADER sentiment approach used in your original script.",
        "category_sentiment": "This section follows your Question 2 logic and compares how positive, neutral, and negative sentiment varies across exotic pet categories such as Big Cats, Primates, Birds, Turtles, Reptiles, and Amphibians.",
        "knowledge_gap": "This section visualizes the distribution of pet categories within each knowledge gap type, matching the original crosstab percentage analysis and stacked bar chart logic.",
        "motivation": "This section summarizes motivations in the full dataset, including admiration and active ownership desire, following the exact summary structure in your original code.",
        "marketplace": "This section shows marketplace or purchase-related signals in the discussion, including overall purchase intensity and category breakdown for purchase-related comments.",
        "welfare": "This section summarizes animal welfare concern as a distinct analytical dimension in the full dataset, using the same full-summary logic from the script.",
        "safety": "This section compares the two safety themes in your original script: strong opposition due to danger, and safety concerns despite interest.",
        "conservation": "This section displays overall conservation concern, category-level conservation distribution, and conservation subthemes such as wildlife trade and habitat concern.",
        "legality": "This section follows the legality analysis in your original script by showing full-dataset legality classification and the pure legality-only breakdown.",
        "geography": "This section reproduces the state-level distribution logic across themes and sentiment outputs, using the same location_geographic field and topic mapping approach.",
        "llm": "This section provides a stable theme-classification page. It uses a safe fallback rule-based classifier so the app keeps working even when no external LLM API is connected."
    }


    # =========================================================
    # GENERIC CHART HELPERS
    # =========================================================
    def plot_bar(df, x, y, title, text_col=None, color=None, horizontal=False):
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            st.info("No data available for this chart.")
            return

        if horizontal:
            fig = px.bar(df, x=y, y=x, orientation="h", text=text_col or y, title=title, color=color)
        else:
            fig = px.bar(df, x=x, y=y, text=text_col or y, title=title, color=color)

        fig.update_layout(height=430, xaxis_title=x, yaxis_title=y)
        if not horizontal:
            fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)


    def plot_stacked_percent(df, x, y_cols, title):
        if df is None or df.empty or x not in df.columns:
            st.info("No data available for this chart.")
            return

        melted = df.melt(id_vars=x, value_vars=y_cols, var_name="Measure", value_name="Percentage")
        fig = px.bar(
            melted,
            x=x,
            y="Percentage",
            color="Measure",
            barmode="stack",
            text="Percentage",
            title=title
        )
        fig.update_traces(texttemplate="%{text:.1f}", textposition="inside")
        fig.update_layout(height=470)
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)


    def safe_percent(df, col_name):
        if df is None or df.empty or col_name not in df.columns:
            return None
        try:
            return float(df[col_name].sum())
        except Exception:
            return None


    def show_description(key: str):
        st.markdown(f"<div class='section-note'>{SECTION_DESCRIPTIONS[key]}</div>", unsafe_allow_html=True)


    def metric(label, value):
        st.metric(label, value)


    # =========================================================
    # SENTIMENT RECOMPUTE FROM ORIGINAL LOGIC
    # =========================================================
    @st.cache_data(show_spinner=False)
    def compute_sentiment_from_raw():
        raw = load_csv("wwf_final_youtube_clean_only_english.csv")
        if raw is None or "text_content" not in raw.columns:
            return None, None

        df = raw.copy()
        df["text_content"] = df["text_content"].astype(str).fillna("").str.strip()
        df = df[df["text_content"] != ""].copy()

        analyzer = SentimentIntensityAnalyzer()

        def classify(text):
            scores = analyzer.polarity_scores(str(text))
            compound = scores["compound"]
            if compound >= 0.05:
                label = "Positive"
            elif compound <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            return pd.Series({
                "sentiment_negative_score": scores["neg"],
                "sentiment_neutral_score": scores["neu"],
                "sentiment_positive_score": scores["pos"],
                "sentiment_compound_score": compound,
                "sentiment_label": label
            })

        sentiment_results = df["text_content"].apply(classify)
        df_sentiment = pd.concat([df, sentiment_results], axis=1)

        summary = (
            df_sentiment["sentiment_label"]
            .value_counts(dropna=False)
            .rename_axis("sentiment_label")
            .reset_index(name="comment_count")
        )
        summary["percentage"] = (
            summary["comment_count"] / summary["comment_count"].sum() * 100
        ).round(2)

        return df_sentiment, summary


    # =========================================================
    # GEOGRAPHY BUILD (same logic as original script)
    # =========================================================
    @st.cache_data(show_spinner=False)
    def build_geography_table():
        files = [
            "Q1_sentiment_scored_comments.csv",
            "knowledge_gap_cleaned.csv",
            "wwf_conservation_pure_comments.csv",
            "wwf_legality_pure_comments.csv",
            "wwf_market_purchase_only_comments.csv",
            "wwf_motivation_pure_comments.csv",
            "wwf_safety_pure_comments_final.csv",
            "wwf_welfare_pure_comments_final.csv"
        ]

        file_name_map = {
            "knowledge_gap_cleaned.csv": "Knowledge Gap",
            "wwf_conservation_pure_comments.csv": "Conservation",
            "wwf_legality_pure_comments.csv": "Legality",
            "wwf_market_purchase_only_comments.csv": "Market Purchase",
            "wwf_motivation_pure_comments.csv": "Motivation",
            "wwf_safety_pure_comments_final.csv": "Safety",
            "wwf_welfare_pure_comments_final.csv": "Animal Welfare"
        }

        state_col = "location_geographic"
        all_data = []

        for file in files:
            df = load_csv(file)
            if df is None or state_col not in df.columns:
                continue

            if file == "Q1_sentiment_scored_comments.csv":
                if "sentiment_label" not in df.columns:
                    continue

                for sentiment_name in ["Positive", "Negative"]:
                    temp = df[
                        df["sentiment_label"].astype(str).str.strip().str.lower() == sentiment_name.lower()
                    ].dropna(subset=[state_col]).copy()

                    temp = temp[temp[state_col].astype(str).str.strip().str.lower() != "us (text signal)"]
                    if temp.empty:
                        continue

                    counts = temp[state_col].value_counts()
                    pct = (counts / counts.sum()) * 100
                    out = pct.reset_index()
                    out.columns = ["State", "Percentage"]
                    out["File"] = f"{sentiment_name} Sentiment"
                    all_data.append(out)
            else:
                temp = df.dropna(subset=[state_col]).copy()
                temp = temp[temp[state_col].astype(str).str.strip().str.lower() != "us (text signal)"]
                if temp.empty:
                    continue

                counts = temp[state_col].value_counts()
                pct = (counts / counts.sum()) * 100
                out = pct.reset_index()
                out.columns = ["State", "Percentage"]
                out["File"] = file_name_map.get(file, file.replace(".csv", ""))
                all_data.append(out)

        if not all_data:
            return None

        final_df = pd.concat(all_data, ignore_index=True)

        top_states = (
            final_df.groupby("State")["Percentage"]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .index
        )

        filtered = final_df[final_df["State"].isin(top_states)].copy()

        topic_order = [
            "Negative Sentiment",
            "Positive Sentiment",
            "Knowledge Gap",
            "Conservation",
            "Legality",
            "Market Purchase",
            "Motivation",
            "Safety",
            "Animal Welfare"
        ]

        filtered["File"] = pd.Categorical(filtered["File"], categories=topic_order, ordered=True)
        filtered = filtered.sort_values("File")
        return filtered


    # =========================================================
    # RULE-BASED THEME CLASSIFIER
    # =========================================================
    def classify_theme_rule_based(text: str) -> str:
        x = str(text).lower()
        if any(term in x for term in ["illegal", "law", "legal", "permit", "ban", "restricted"]):
            return "Legality"
        if any(term in x for term in ["danger", "attack", "unsafe", "bite", "risk", "hurt"]):
            return "Safety"
        if any(term in x for term in ["wild", "endangered", "habitat", "conservation", "trafficking", "trade"]):
            return "Conservation"
        if any(term in x for term in ["cruel", "cage", "suffer", "welfare", "abuse"]):
            return "Animal Welfare"
        if any(term in x for term in ["buy", "sell", "price", "purchase", "market"]):
            return "Market Purchase"
        if any(term in x for term in ["want one", "i want", "my dream pet", "would love", "so cute"]):
            return "Motivation / Admiration"
        return "Other"


    # =========================================================
    # SIDEBAR FILTERS
    # =========================================================
    st.sidebar.title("Dashboard Controls")
    available_files = list_available_files()

    raw_df = load_csv("wwf_final_youtube_clean_only_english.csv")
    country_options = ["All"]
    species_options = ["All"]

    if raw_df is not None:
        if "country_context" in raw_df.columns:
            vals = sorted([v for v in raw_df["country_context"].dropna().astype(str).unique() if v.strip()])
            country_options += vals
        if "keyword_used" in raw_df.columns:
            vals = sorted([v for v in raw_df["keyword_used"].dropna().astype(str).unique() if v.strip()])
            species_options += vals

    selected_country = st.sidebar.selectbox("Country filter", country_options, index=0)
    selected_species = st.sidebar.selectbox("Species / keyword filter", species_options, index=0)
    sentiment_threshold = st.sidebar.slider("Sentiment threshold", 0.00, 0.50, 0.05, 0.01)

    st.sidebar.caption(f"Detected data source: {DATA_DIR}")


    def apply_common_filters(df: pd.DataFrame | None):
        if df is None or df.empty:
            return df
        out = df.copy()
        if selected_country != "All" and "country_context" in out.columns:
            out = out[out["country_context"].astype(str) == selected_country]
        if selected_species != "All" and "keyword_used" in out.columns:
            out = out[out["keyword_used"].astype(str) == selected_species]
        return out


    # =========================================================
    # TITLE
    # =========================================================
    st.title("WWF Exotic Pet Ownership Research Dashboard")
    st.caption("Interactive dashboard built to follow the logic and output structure of your original Python analysis script.")


    # =========================================================
    # TABS
    # =========================================================
    tabs = st.tabs([
        "Overview",
        "Q1 Sentiment",
        "Q2 Category Sentiment",
        "Knowledge Gap",
        "Motivation",
        "Marketplace",
        "Welfare",
        "Safety",
        "Conservation",
        "Legality",
        "Geography",
        "Word Cloud",
        "LLM Themes",
        "Exports"
    ])


    # =========================================================
    # OVERVIEW
    # =========================================================
    with tabs[0]:
        st.subheader("Study Dashboard Overview")
        show_description("overview")

        q1_summary = load_csv("Q1_sentiment_summary.csv")
        if q1_summary is None:
            _, q1_summary = compute_sentiment_from_raw()

        q2_percent = load_csv("Q2_final_category_sentiment_percent.csv")
        motivation = load_csv("wwf_motivation_overall_summary.csv")
        conservation = load_csv("wwf_conservation_overall_summary.csv")
        legality = load_csv("wwf_legality_overall_summary.csv")
        welfare = load_csv("wwf_welfare_overall_summary_final.csv")
        market = load_csv("wwf_market_purchase_overall_summary.csv")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if q1_summary is not None and not q1_summary.empty:
                pos = q1_summary.loc[q1_summary["sentiment_label"] == "Positive", "percentage"]
                metric("Positive sentiment", f"{float(pos.iloc[0]):.2f}%" if not pos.empty else "N/A")
        with c2:
            if conservation is not None and not conservation.empty:
                con = conservation.loc[conservation["Type"] == "Conservation Concern", "Percentage_of_full_dataset"]
                metric("Conservation concern", f"{float(con.iloc[0]):.2f}%" if not con.empty else "N/A")
        with c3:
            if welfare is not None and not welfare.empty:
                w = welfare.loc[welfare["Type"] == "Animal Welfare Concern", "Percentage_of_full_dataset"]
                metric("Animal welfare concern", f"{float(w.iloc[0]):.2f}%" if not w.empty else "N/A")
        with c4:
            if market is not None and not market.empty:
                m = market.loc[market["Purchase Type"] == "Purchase Related", "Percentage_of_28k"]
                metric("Purchase-related comments", f"{float(m.iloc[0]):.2f}%" if not m.empty else "N/A")

        st.markdown("### Dashboard snapshots")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("**Overall sentiment distribution**")
            st.caption("This snapshot summarizes whether public comments are supportive, neutral, or critical.")
            plot_bar(q1_summary, "sentiment_label", "comment_count", "Q1: Overall Sentiment Distribution", text_col="percentage")
        with r1c2:
            st.markdown("**Sentiment across pet categories**")
            st.caption("This snapshot compares positive, neutral, and negative sentiment across species groups.")
            if q2_percent is not None:
                plot_stacked_percent(q2_percent, "Category", ["Positive_%", "Neutral_%", "Negative_%"], "Q2: Sentiment Across Categories")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**Theme prevalence in the full discussion**")
            st.caption("This reflects the cross-theme distribution from your Book_3 summary file.")
            book3 = load_csv("Book_3.csv")
            if book3 is not None and not book3.empty:
                df = book3.copy()
                df.columns = [c.strip() for c in df.columns]
                df = df.rename(columns={"Themes": "Theme", "%": "Count"})
                if "Count" in df.columns and "Total" in df.columns:
                    df["Percentage"] = (df["Count"] / df["Total"] * 100).round(2)
                    theme_map = {
                        "Knowledge gap": "Knowledge Gap",
                        "conservation": "Conservation",
                        "legality": "Legality",
                        "market purchase": "Market Purchase",
                        "active motivation/ownership desire": "Motivation / Ownership",
                        "Safety": "Safety",
                        "animal welfare": "Animal Welfare",
                        "admiration": "Admiration"
                    }
                    df["Theme"] = df["Theme"].replace(theme_map)
                    df = df.sort_values("Percentage", ascending=True)
                    plot_bar(df, "Theme", "Percentage", "YouTube Discussion Themes and Motivations (%)", horizontal=True)
            else:
                st.info("Book_3 theme summary is not available.")

        with r2c2:
            st.markdown("**Risk and regulatory dimensions**")
            st.caption("This landing view highlights legality, welfare, and marketplace dimensions from the original script outputs.")
            combo = []
            if legality is not None and not legality.empty:
                legal_risk = legality.loc[legality["Type"] == "Legal Risk / Restriction", "Percentage_of_full_dataset"]
                if not legal_risk.empty:
                    combo.append({"Metric": "Legal Risk / Restriction", "Percentage": float(legal_risk.iloc[0])})
            if welfare is not None and not welfare.empty:
                aw = welfare.loc[welfare["Type"] == "Animal Welfare Concern", "Percentage_of_full_dataset"]
                if not aw.empty:
                    combo.append({"Metric": "Animal Welfare Concern", "Percentage": float(aw.iloc[0])})
            if market is not None and not market.empty:
                pr = market.loc[market["Purchase Type"] == "Purchase Related", "Percentage_of_28k"]
                if not pr.empty:
                    combo.append({"Metric": "Purchase Related", "Percentage": float(pr.iloc[0])})
            if conservation is not None and not conservation.empty:
                cc = conservation.loc[conservation["Type"] == "Conservation Concern", "Percentage_of_full_dataset"]
                if not cc.empty:
                    combo.append({"Metric": "Conservation Concern", "Percentage": float(cc.iloc[0])})

            if combo:
                combo_df = pd.DataFrame(combo)
                plot_bar(combo_df, "Metric", "Percentage", "Key Non-Sentiment Signals in the Full Dataset", text_col="Percentage")
            else:
                st.info("Summary files for key metrics are not available.")


    # =========================================================
    # Q1 SENTIMENT
    # =========================================================
    with tabs[1]:
        st.subheader("Q1: Overall Sentiment Toward High-Risk Exotic Pet Ownership")
        show_description("sentiment")

        df_sentiment = load_csv("Q1_sentiment_scored_comments.csv")
        summary = load_csv("Q1_sentiment_summary.csv")

        if df_sentiment is None or summary is None:
            df_sentiment, summary = compute_sentiment_from_raw()

        df_sentiment = apply_common_filters(df_sentiment)

        if df_sentiment is not None and not df_sentiment.empty:
            if "sentiment_compound_score" not in df_sentiment.columns:
                analyzer = SentimentIntensityAnalyzer()
                df_sentiment["sentiment_compound_score"] = df_sentiment["text_content"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])

            if "sentiment_label" not in df_sentiment.columns:
                def relabel(x):
                    if x >= sentiment_threshold:
                        return "Positive"
                    elif x <= -sentiment_threshold:
                        return "Negative"
                    return "Neutral"
                df_sentiment["sentiment_label"] = df_sentiment["sentiment_compound_score"].apply(relabel)

            local_summary = (
                df_sentiment["sentiment_label"]
                .value_counts()
                .rename_axis("sentiment_label")
                .reset_index(name="comment_count")
            )
            local_summary["percentage"] = (local_summary["comment_count"] / local_summary["comment_count"].sum() * 100).round(2)

            a, b, c = st.columns(3)
            for col, label, target in zip([a, b, c], ["Positive", "Neutral", "Negative"], ["Positive", "Neutral", "Negative"]):
                pct = local_summary.loc[local_summary["sentiment_label"] == target, "percentage"]
                with col:
                    metric(label, f"{float(pct.iloc[0]):.2f}%" if not pct.empty else "0.00%")

            c1, c2 = st.columns(2)
            with c1:
                plot_bar(local_summary, "sentiment_label", "comment_count", "Overall Sentiment Distribution", text_col="percentage")
            with c2:
                fig = px.pie(local_summary, values="comment_count", names="sentiment_label", title="Share of Overall Sentiment")
                st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(
                df_sentiment,
                x="sentiment_compound_score",
                nbins=30,
                title="Distribution of Sentiment Compound Scores"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Sample comments by sentiment")
            sample_label = st.selectbox("Choose sentiment sample", ["Positive", "Neutral", "Negative"], key="sample_sentiment")
            sample_df = df_sentiment[df_sentiment["sentiment_label"] == sample_label][["text_content", "sentiment_compound_score"]].head(25)
            st.dataframe(sample_df, use_container_width=True)
        else:
            st.warning("Sentiment source data is not available.")


    # =========================================================
    # Q2 CATEGORY SENTIMENT
    # =========================================================
    with tabs[2]:
        st.subheader("Q2: How Sentiment Varies Across Exotic Pet Categories")
        show_description("category_sentiment")

        final_df = load_csv("Q2_final_category_sentiment.csv")
        percent_df = load_csv("Q2_final_category_sentiment_percent.csv")

        if percent_df is not None:
            plot_stacked_percent(percent_df, "Category", ["Positive_%", "Neutral_%", "Negative_%"], "Sentiment Distribution Across Exotic Pet Categories (%)")

        if final_df is not None:
            st.markdown("#### Final category sentiment table")
            st.dataframe(final_df, use_container_width=True)

        st.markdown("#### Pet-level category files")
        category_file_map = {
            "Big Cats": "Q2_big_cats_pet_sentiment.csv",
            "Primates": "Q2_primates_pet_sentiment.csv",
            "Parrots & Birds": "Q2_parrots_and_birds_pet_sentiment.csv",
            "Turtles / Tortoises": "Q2_turtles___tortoises_pet_sentiment.csv",
            "Reptiles": "Q2_reptiles_pet_sentiment.csv",
            "Amphibians": "Q2_amphibians_pet_sentiment.csv",
        }

        selected_cat = st.selectbox("Choose category detail", list(category_file_map.keys()))
        cat_df = load_csv(category_file_map[selected_cat])
        if cat_df is not None:
            st.dataframe(cat_df, use_container_width=True)
        else:
            st.info("Pet-level file for this category is not available.")


    # =========================================================
    # KNOWLEDGE GAP
    # =========================================================
    with tabs[3]:
        st.subheader("Knowledge Gap by Category")
        show_description("knowledge_gap")

        gap = load_csv("knowledge_gap_category_percentages.csv")
        if gap is not None:
            y_cols = [c for c in gap.columns if c != "knowledge_gap_type"]
            plot_stacked_percent(gap, "knowledge_gap_type", y_cols, "Pet Category Distribution Within Each Knowledge Gap Type (%)")
            st.dataframe(gap, use_container_width=True)
        else:
            st.info("Knowledge gap summary is not available.")


    # =========================================================
    # MOTIVATION
    # =========================================================
    with tabs[4]:
        st.subheader("Motivation Classification")
        show_description("motivation")

        overall_summary = load_csv("wwf_motivation_overall_summary.csv")
        plot_bar(overall_summary, "Type", "Count", "Motivation Classification (Full Dataset)", text_col="Percentage_of_full_dataset")
        if overall_summary is not None:
            st.dataframe(overall_summary, use_container_width=True)


    # =========================================================
    # MARKETPLACE
    # =========================================================
    with tabs[5]:
        st.subheader("Marketplace Signals")
        show_description("marketplace")

        summary = load_csv("wwf_market_purchase_overall_summary.csv")
        category_percent = load_csv("wwf_market_purchase_category_summary.csv")

        c1, c2 = st.columns(2)
        with c1:
            plot_bar(summary, "Purchase Type", "Count", "Marketplace Signals in Exotic Pet Discussions", text_col="Percentage_of_28k")
        with c2:
            if category_percent is not None and "Purchase Category" in category_percent.columns:
                plot_bar(
                    category_percent,
                    "Purchase Category",
                    "Percentage_within_purchase",
                    "Marketplace Signals by Exotic Pet Category (Percentage within Purchase)",
                    text_col="Percentage_within_purchase"
                )
            else:
                st.info("Marketplace category breakdown is not available.")

        if summary is not None:
            st.dataframe(summary, use_container_width=True)


    # =========================================================
    # WELFARE
    # =========================================================
    with tabs[6]:
        st.subheader("Animal Welfare")
        show_description("welfare")

        overall_summary = load_csv("wwf_welfare_overall_summary_final.csv")
        plot_bar(overall_summary, "Type", "Count", "Animal Welfare Classification (Full Dataset)", text_col="Percentage_of_full_dataset")
        if overall_summary is not None:
            st.dataframe(overall_summary, use_container_width=True)


    # =========================================================
    # SAFETY
    # =========================================================
    with tabs[7]:
        st.subheader("Safety Themes")
        show_description("safety")

        df_strong = load_csv("wwf_safety_pure_comments.csv")
        df_concern = load_csv("wwf_safety_pure_comments_final.csv")

        if df_strong is not None and df_concern is not None:
            strong_count = len(df_strong)
            concern_count = len(df_concern)
            total = strong_count + concern_count

            if total > 0:
                summary_df = pd.DataFrame({
                    "Safety Theme": [
                        "Strong Opposition Due to Danger",
                        "Safety Concerns Despite Interest"
                    ],
                    "Comments": [strong_count, concern_count],
                    "Percentage": [(strong_count / total) * 100, (concern_count / total) * 100]
                })
                plot_bar(summary_df, "Safety Theme", "Percentage", "Safety Themes in Exotic Pet Comments", text_col="Percentage")
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("Safety files are not available.")


    # =========================================================
    # CONSERVATION
    # =========================================================
    with tabs[8]:
        st.subheader("Conservation Concern")
        show_description("conservation")

        overall_summary = load_csv("wwf_conservation_overall_summary.csv")
        category_summary = load_csv("wwf_conservation_category_summary.csv")
        subtheme_summary = load_csv("wwf_conservation_subtheme_summary.csv")

        c1, c2 = st.columns(2)
        with c1:
            plot_bar(overall_summary, "Type", "Count", "Conservation Concern in Full Dataset", text_col="Percentage_of_full_dataset")
        with c2:
            plot_bar(category_summary, "Category", "Percentage_within_conservation", "Conservation Concerns by Pet Category (%)", text_col="Percentage_within_conservation")

        plot_bar(subtheme_summary, "Conservation Subtheme", "Count", "Types of Conservation Concerns", text_col="Percentage_within_conservation")
        if category_summary is not None:
            st.dataframe(category_summary, use_container_width=True)


    # =========================================================
    # LEGALITY
    # =========================================================
    with tabs[9]:
        st.subheader("Legality Classification")
        show_description("legality")

        overall_summary = load_csv("wwf_legality_overall_summary.csv")
        pure_summary = load_csv("wwf_legality_pure_summary.csv")

        c1, c2 = st.columns(2)
        with c1:
            plot_bar(overall_summary, "Type", "Count", "Legality Classification Across Full Dataset", text_col="Percentage_of_full_dataset")
        with c2:
            plot_bar(pure_summary, "Type", "Count", "Legality Awareness vs Legal Risk", text_col="Percentage_within_legality")

        if pure_summary is not None:
            st.dataframe(pure_summary, use_container_width=True)


    # =========================================================
    # GEOGRAPHY
    # =========================================================
    with tabs[10]:
        st.subheader("US State Distribution Across Themes and Sentiments")
        show_description("geography")

        geo_df = build_geography_table()
        if geo_df is not None and not geo_df.empty:
            fig = px.bar(
                geo_df,
                x="File",
                y="Percentage",
                color="State",
                barmode="group",
                text="Percentage",
                title="US State Distribution Across WWF Topics and Sentiments (%)"
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=540, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(geo_df, use_container_width=True)
        else:
            st.info("Geographic source files are not fully available.")


    # =========================================================
    # WORD CLOUD
    # =========================================================
    with tabs[11]:
        st.subheader("Word Cloud and NLP Snapshot")
        raw = apply_common_filters(load_csv("wwf_final_youtube_clean_only_english.csv"))
        if raw is not None and "text_content" in raw.columns:
            text = " ".join(raw["text_content"].astype(str).dropna().tolist()).strip()
            if text:
                wc = WordCloud(width=1200, height=500, background_color="white", collocations=False).generate(text)
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                st.caption("This word cloud is generated directly from the text_content column and provides a quick lexical snapshot of the discussion.")
            else:
                st.info("No usable text available after filters.")
        else:
            st.info("Raw text source file is not available.")


    # =========================================================
    # LLM THEMES
    # =========================================================
    with tabs[12]:
        st.subheader("Theme Classification")
        show_description("llm")

        raw = apply_common_filters(load_csv("wwf_final_youtube_clean_only_english.csv"))
        if raw is not None and "text_content" in raw.columns:
            work = raw.copy()
            work["theme"] = work["text_content"].astype(str).apply(classify_theme_rule_based)
            theme_counts = work["theme"].value_counts().reset_index()
            theme_counts.columns = ["theme", "count"]

            fig = px.pie(theme_counts, values="count", names="theme", title="Rule-Based Theme Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(work[["text_content", "theme"]].head(100), use_container_width=True)
        else:
            st.info("Theme classification source text is not available.")


    # =========================================================
    # EXPORTS
    # =========================================================
    with tabs[13]:
        st.subheader("Exports")
        st.write("Download the source CSVs currently detected by the app.")

        for file in available_files:
            path = DATA_DIR / file
            if path.suffix.lower() == ".csv":
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f.read(),
                        file_name=file,
                        mime="text/csv",
                        key=f"dl_{file}"
                    )

def render_uploaded_dashboard():

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Research Dashboard</div>
            <div class="hero-title">Exotic Pet Trade Research Dashboard</div>
            <div class="hero-subtitle">Understand online discourse on exotic pets: sentiment, themes, species risk, platform differences, language patterns, and high-priority posts for WWF intervention.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload your dataset.")
        return

    # -----------------------------
    # TEXT COLUMN DETECTION
    # -----------------------------
    TEXT_COL = None
    for candidate in ["text_content", "text", "snippet"]:
        if candidate in df.columns:
            TEXT_COL = candidate
            break

    if TEXT_COL is None:
        st.error("No text column found (expected one of: text_content, text, snippet).")
        return

    if TEXT_COL != "text_content":
        df["text_content"] = df[TEXT_COL]
        TEXT_COL = "text_content"

    # -----------------------------
    # SENTIMENT ANALYSIS
    # -----------------------------
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        return analyzer.polarity_scores(str(text))["compound"]

    df[TEXT_COL] = df[TEXT_COL].fillna("")

    # Prefer provided sentiment score if available.
    if "sentiment_score" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    else:
        df["sentiment"] = df[TEXT_COL].apply(get_sentiment)

    def label_sentiment(score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

    SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
    SENTIMENT_COLOR_MAP = {
        "Negative": "#C44E52",
        "Neutral": "#9A9A9A",
        "Positive": "#4C9F70",
    }

    DETAIL_SOURCE_OPTIONS = ["Uploaded Data Visuals", "Final Data Visuals"]
    st.sidebar.subheader("Detailed Visual Settings")
    detail_visual_source = st.sidebar.radio(
        "Data source for Detailed Visual Dashboard",
        DETAIL_SOURCE_OPTIONS,
        index=1,
    )

    # -----------------------------
    # STANCE CLASSIFICATION (Simple)
    # -----------------------------
    df["stance"] = np.where(
        df["sentiment"] > 0.1,
        "Supportive",
        np.where(df["sentiment"] < -0.1, "Critical", "Neutral"),
    )

    # Compute conservation risk once so all tabs can safely reference these columns.
    risk_classifier = train_conservation_risk_classifier(df[TEXT_COL].tolist())
    risk_df = predict_conservation_risk(df[TEXT_COL].tolist(), risk_classifier)
    df["conservation_risk_label"] = risk_df["risk_label"]
    df["conservation_risk_score"] = risk_df["risk_score"]

    @st.cache_data
    def load_optional_csv(path):
        try:
            candidate = Path(path)
            if candidate.exists():
                return pd.read_csv(candidate)
            alt = DATA_DIR / candidate.name
            if alt.exists():
                return pd.read_csv(alt)
            return None
        except Exception:
            return None


    def build_uploaded_q1_summary(df_input):
        q1 = (
            df_input["sentiment_label"]
            .value_counts(dropna=False)
            .rename_axis("sentiment_label")
            .reset_index(name="comment_count")
        )
        q1["sentiment_label"] = pd.Categorical(q1["sentiment_label"], categories=SENTIMENT_ORDER, ordered=True)
        q1 = q1.sort_values("sentiment_label")
        q1["percentage"] = (q1["comment_count"] / max(q1["comment_count"].sum(), 1) * 100).round(2)
        return q1


    def build_uploaded_q2_percent(df_input, text_col):
        category_patterns = {
            "Big Cats": ["tiger", "lion", "jaguar", "cheetah", "serval", "leopard"],
            "Primates": ["monkey", "chimp", "gibbon", "lemur", "macaque", "loris", "marmoset"],
            "Parrots & Birds": ["parrot", "macaw", "cockatoo", "bird", "falcon"],
            "Turtles / Tortoises": ["turtle", "tortoise"],
            "Reptiles": ["python", "boa", "iguana", "lizard", "snake", "reptile"],
            "Amphibians": ["frog", "toad", "amphibian", "salamander"],
        }

        labeled_df = df_input.copy()
        labeled_df["_category"] = "General Exotic Pet"
        for category_name, words in category_patterns.items():
            mask = labeled_df[text_col].str.contains("|".join(words), case=False, na=False)
            labeled_df.loc[mask & (labeled_df["_category"] == "General Exotic Pet"), "_category"] = category_name

        q2_counts = (
            labeled_df.groupby(["_category", "sentiment_label"]) 
            .size()
            .unstack(fill_value=0)
            .reindex(columns=SENTIMENT_ORDER, fill_value=0)
        )

        q2_percent = q2_counts.div(q2_counts.sum(axis=1).replace(0, 1), axis=0) * 100
        q2_percent = q2_percent.round(2).reset_index().rename(columns={"_category": "Category"})
        q2_percent = q2_percent.rename(columns={
            "Positive": "Positive_%",
            "Neutral": "Neutral_%",
            "Negative": "Negative_%",
        })
        return q2_percent


    tab_sentiment, tab_themes, tab_species, tab_platform, tab_language, tab_risk_triage, tab_detail, tab_experimental = st.tabs(
        [
            "1. Sentiment Landscape",
            "2. Themes & Motivations",
            "3. Species Risk",
            "4. Platform Intelligence",
            "5. Language Insights",
            "6. Risk Triage",
            "7. Detailed Visual Dashboard",
            "8. Experimental",
        ]
    )

    # ==================== RESEARCH QUESTION 1: SENTIMENT LANDSCAPE ====================
    with tab_sentiment:
        st.subheader("📊 Research Q1: How much is negative, neutral, or supportive?")
        st.caption("This tab answers the core sentiment landscape of exotic pet discourse.")
    
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Comments", len(df))
        col2.metric("Avg Sentiment Score", round(df["sentiment"].mean(), 3))
        neg_pct = round((df['sentiment_label'] == 'Negative').mean() * 100, 1)
        pos_pct = round((df['sentiment_label'] == 'Positive').mean() * 100, 1)
        neu_pct = round((df['sentiment_label'] == 'Neutral').mean() * 100, 1)
        col3.metric("Negative %", f"{neg_pct}%")
        col4.metric("Positive %", f"{pos_pct}%")
    
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        fig_sent = px.histogram(
            df, 
            x="sentiment_label", 
            color="sentiment_label",
            category_orders={"sentiment_label": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLOR_MAP,
            title="How much discourse is negative, neutral, or supportive?"
        )
        st.plotly_chart(fig_sent, use_container_width=True)
    
        # Stance distribution (related but distinct)
        st.subheader("Stance Distribution")
        fig_stance = px.histogram(
            df, 
            x="stance", 
            color="stance",
            category_orders={"stance": ["Critical", "Neutral", "Supportive"]},
            title="Stance: Critical vs Neutral vs Supportive"
        )
        st.plotly_chart(fig_stance, use_container_width=True)
    
        # Optional: sentiment by platform if available
        if "platform" in df.columns:
            st.subheader("Sentiment by Platform")
            fig_platform = px.box(
                df, 
                x="platform", 
                y="sentiment",
                title="Platform Comparison (Sentiment Score Distribution)"
            )
            st.plotly_chart(fig_platform, use_container_width=True)

    # ==================== RESEARCH QUESTION 2: THEMES & MOTIVATIONS ====================
    with tab_themes:
        st.subheader("🎯 Research Q2: Which themes dominate?")
        st.caption("Multi-level evidence scoring: mention count, document frequency, cross-theme overlap, and platform prevalence.")
    
        # Define themes clearly
        keywords = {
            "legality": ["illegal", "law", "permit", "license", "regulated", "banned"],
            "safety": ["bite", "attack", "danger", "injury", "disease", "risk"],
            "welfare": ["suffer", "stress", "cruel", "captivity", "neglect", "care"],
            "conservation": ["endangered", "extinction", "ecosystem", "wildlife", "biodiversity"],
            "trade": ["for sale", "selling", "breeder", "expo", "shipping", "contact"],
        }

        # Multi-level evidence scoring
        theme_evidence = []
        for theme, words in keywords.items():
            df[theme] = df[TEXT_COL].str.contains("|".join(words), case=False, na=False)
        
            mentions = int(df[theme].sum())
            doc_freq = round(mentions / len(df) * 100, 1)  # % of documents mentioning theme
            risk_avg = df.loc[df[theme], "conservation_risk_score"].mean() if mentions > 0 else 0
        
            theme_evidence.append({
                "Theme": theme.capitalize(),
                "Mentions": mentions,
                "Document Frequency %": doc_freq,
                "Avg Risk Score": round(risk_avg, 2),
                "Risk Intensity": "High" if risk_avg >= 4 else "Medium" if risk_avg >= 2 else "Low"
            })
    
        theme_df = pd.DataFrame(theme_evidence).sort_values("Mentions", ascending=False)
    
        fig_theme = px.bar(
            theme_df, 
            x="Theme", 
            y="Mentions", 
            color="Avg Risk Score",
            title="Theme Prevalence with Risk Intensity",
            labels={"Mentions": "Number of posts", "Avg Risk Score": "Avg Conservation Risk"}
        )
        st.plotly_chart(fig_theme, use_container_width=True)
    
        st.subheader("Multi-Level Theme Evidence")
        st.dataframe(theme_df, use_container_width=True, hide_index=True)
    
        # Cross-theme overlap analysis
        st.subheader("Cross-Theme Overlap (Posts with Multiple Concerns)")
        df["theme_count"] = df[keywords.keys()].sum(axis=1)
        overlap_data = pd.DataFrame({
            "Themes Combined": ["1 Theme", "2 Themes", "3+ Themes"],
            "Post Count": [
                sum(df["theme_count"] == 1),
                sum(df["theme_count"] == 2),
                sum(df["theme_count"] >= 3)
            ]
        })
        overlap_data["Percentage"] = round(overlap_data["Post Count"] / len(df) * 100, 1)
    
        fig_overlap = px.bar(
            overlap_data,
            x="Themes Combined",
            y="Post Count",
            color="Percentage",
            title="Posts with Multiple Concurrent Themes (Higher complexity = riskier)",
            text="Percentage"
        )
        st.plotly_chart(fig_overlap, use_container_width=True)
    
        st.subheader("Theme Definitions")
        theme_defs = {
            "Legality": "Posts discussing laws, permits, regulations, bans, or legal status",
            "Safety": "Posts about physical risks: bites, attacks, disease, injuries",
            "Welfare": "Posts about animal care needs, suffering, stress, captivity conditions",
            "Conservation": "Posts about species endangerment, ecosystem impact, biodiversity",
            "Trade": "Posts about purchasing, breeding, shipping, or commercial channels",
        }
        for theme, definition in theme_defs.items():
            st.write(f"**{theme.upper()}**: {definition}")

    # ==================== RESEARCH QUESTION 3: SPECIES RISK PROFILE ====================
    with tab_species:
        st.subheader("🐅 Research Q3: Which species attract highest-risk discussions?")
        st.caption("Identify which species groups drive the most risky discourse.")
    
        # Species groups
        species_groups = {
            "Big Cats": ["tiger", "lion", "jaguar", "cheetah", "leopard", "serval"],
            "Primates": ["monkey", "chimpanzee", "gibbon", "lemur", "macaque"],
            "Parrots/Birds": ["parrot", "macaw", "cockatoo", "eagle", "falcon"],
            "Reptiles": ["python", "boa", "monitor", "iguana", "serpent"],
            "Other": []
        }
    
        # Count species and average risk
        species_risk_data = []
        for group, species_list in species_groups.items():
            if species_list:
                mask = df[TEXT_COL].str.contains("|".join(species_list), case=False, na=False)
                count = mask.sum()
                avg_risk = df.loc[mask, "conservation_risk_score"].mean() if count > 0 else 0
                high_risk_pct = round((df.loc[mask, "conservation_risk_label"] == "high").sum() / max(count, 1) * 100, 1) if count > 0 else 0
            else:
                count = len(df)
                avg_risk = df["conservation_risk_score"].mean()
                high_risk_pct = round((df["conservation_risk_label"] == "high").sum() / max(count, 1) * 100, 1) if count > 0 else 0
        
            species_risk_data.append({
                "Species Group": group,
                "Post Count": count,
                "Avg Risk Score": round(avg_risk, 2),
                "% High Risk": high_risk_pct
            })
    
        species_risk_df = pd.DataFrame(species_risk_data).sort_values("Post Count", ascending=False)
    
        fig_species = px.bar(
            species_risk_df,
            x="Species Group",
            y="Post Count",
            color="Avg Risk Score",
            title="Species Groups and Risk Profile",
            labels={"Post Count": "Posts mentioning group"}
        )
        st.plotly_chart(fig_species, use_container_width=True)
    
        st.subheader("Species Group Risk Details")
        st.dataframe(species_risk_df, use_container_width=True, hide_index=True)


        # High-risk examples
        high_risk = df[df["conservation_risk_label"] == "high"].head(5)
        if not high_risk.empty:
            st.subheader("High-Risk Post Examples")
            for idx, row in high_risk.iterrows():
                st.write(f"**{row['title'][:80]}...**" if "title" in row.index else "Post")
                st.info(f"Risk Score: {row['conservation_risk_score']:.2f} | {row['conservation_risk_label'].upper()}")

    # ==================== RESEARCH QUESTION 4: PLATFORM INTELLIGENCE ====================
    with tab_platform:
        st.subheader("🌐 Research Q4: How do platforms differ?")
        st.caption("Compare sentiment, themes, and content patterns across platforms.")
    
        if "platform" in df.columns:
            platforms = df["platform"].unique()
        
            # Sentiment by platform
            st.subheader("Sentiment Distribution by Platform")
            platform_sentiment = df.groupby(["platform", "sentiment_label"]).size().unstack(fill_value=0)
            fig_plat_sent = px.bar(
                platform_sentiment.reset_index().melt(id_vars="platform", var_name="Sentiment", value_name="Count"),
                x="platform",
                y="Count",
                color="Sentiment",
                barmode="group",
                category_orders={"Sentiment": SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_COLOR_MAP,
                title="Sentiment by Platform"
            )
            st.plotly_chart(fig_plat_sent, use_container_width=True)
        
            # Theme intensity by platform
            st.subheader("Theme Intensity by Platform")
            keywords = {
                "legality": ["illegal", "law", "permit", "license", "regulated", "banned"],
                "safety": ["bite", "attack", "danger", "injury", "disease", "risk"],
                "welfare": ["suffer", "stress", "cruel", "captivity", "neglect", "care"],
                "conservation": ["endangered", "extinction", "ecosystem", "wildlife", "biodiversity"],
                "trade": ["for sale", "selling", "breeder", "expo", "shipping", "contact"],
            }
        
            theme_by_platform = []
            for platform in platforms:
                platform_df = df[df["platform"] == platform]
                for theme, words in keywords.items():
                    count = platform_df[TEXT_COL].str.contains("|".join(words), case=False, na=False).sum()
                    pct = round(count / len(platform_df) * 100, 1) if len(platform_df) > 0 else 0
                    theme_by_platform.append({"Platform": platform, "Theme": theme, "% of Posts": pct})
        
            theme_plat_df = pd.DataFrame(theme_by_platform)
            fig_plat_theme = px.bar(
                theme_plat_df,
                x="Theme",
                y="% of Posts",
                color="Platform",
                barmode="group",
                title="Theme Intensity by Platform"
            )
            st.plotly_chart(fig_plat_theme, use_container_width=True)
        
            # Theme-by-Platform Heatmap (relationship visualization)
            st.subheader("Theme-by-Platform Heatmap (Relationship Matrix)")
            pivot_theme_platform = theme_plat_df.pivot(index="Theme", columns="Platform", values="% of Posts").fillna(0)
            fig_heatmap = px.imshow(
                pivot_theme_platform,
                labels=dict(x="Platform", y="Theme", color="% of Posts"),
                title="Which platforms emphasize which themes? (Darker = higher %)",
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
            # Platform statistics table
            st.subheader("Platform Summary Statistics")
            platform_stats = []
            for platform in platforms:
                plat_data = df[df["platform"] == platform]
                platform_stats.append({
                    "Platform": platform,
                    "Post Count": len(plat_data),
                    "Avg Sentiment": round(plat_data["sentiment"].mean(), 3),
                    "% Negative": round((plat_data["sentiment_label"] == "Negative").mean() * 100, 1),
                    "% Positive": round((plat_data["sentiment_label"] == "Positive").mean() * 100, 1),
                    "Avg Risk Score": round(plat_data["conservation_risk_score"].mean(), 2)
                })
        
            platform_stats_df = pd.DataFrame(platform_stats)
            st.dataframe(platform_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("Platform column not found in data. Upload data that includes platform information.")

    # ==================== RESEARCH QUESTION 5: LANGUAGE INSIGHTS ====================
    with tab_language:
        st.subheader("💬 Research Q5: Which words/phrases dominate?")
        st.caption("Analyze top terms and language patterns by platform or stance.")
    
        # Option to segment by platform or stance
        segment_by = st.radio("Segment language patterns by:", ["Platform", "Stance"])
    
        if segment_by == "Platform" and "platform" in df.columns:
            platform_selected = st.selectbox("Select platform:", df["platform"].unique())
            plat_texts = df[df["platform"] == platform_selected][TEXT_COL].astype(str)
        
            if len(plat_texts) >= 5:
                try:
                    vectorizer = CountVectorizer(stop_words="english", max_features=20)
                    X = vectorizer.fit_transform(plat_texts)
                    term_freq = np.asarray(X.sum(axis=0)).flatten()
                    terms = vectorizer.get_feature_names_out()
                
                    term_df = pd.DataFrame({
                        "Term": terms,
                        "Frequency": term_freq
                    }).sort_values("Frequency", ascending=False)
                
                    fig_terms = px.bar(
                        term_df.head(15),
                        x="Term",
                        y="Frequency",
                        title=f"Top Terms on {platform_selected}"
                    )
                    st.plotly_chart(fig_terms, use_container_width=True)
                
                    st.subheader("Top Terms Table")
                    st.dataframe(term_df.head(20), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not extract terms: {e}")
            else:
                st.info(f"Not enough posts on {platform_selected} for language analysis.")
    
        else:  # Segment by Stance
            stance_selected = st.selectbox("Select stance:", df["stance"].unique())
            stance_texts = df[df["stance"] == stance_selected][TEXT_COL].astype(str)
        
            if len(stance_texts) >= 5:
                try:
                    vectorizer = CountVectorizer(stop_words="english", max_features=20)
                    X = vectorizer.fit_transform(stance_texts)
                    term_freq = np.asarray(X.sum(axis=0)).flatten()
                    terms = vectorizer.get_feature_names_out()
                
                    term_df = pd.DataFrame({
                        "Term": terms,
                        "Frequency": term_freq
                    }).sort_values("Frequency", ascending=False)
                
                    fig_terms = px.bar(
                        term_df.head(15),
                        x="Term",
                        y="Frequency",
                        title=f"Top Terms in {stance_selected} Comments"
                    )
                    st.plotly_chart(fig_terms, use_container_width=True)
                
                    st.subheader("Top Terms Table")
                    st.dataframe(term_df.head(20), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not extract terms: {e}")
            else:
                st.info(f"Not enough {stance_selected} posts for language analysis.")

    # ==================== RESEARCH QUESTION 6: RISK TRIAGE ====================
    with tab_risk_triage:
        st.subheader("⚠️ Research Q6: Which posts need WWF attention?")
        st.caption("High-priority posts combining trade signals with conservation or welfare risk.")
    
        # Define trade and risk keywords
        trade_keywords = ["for sale", "selling", "breeder", "expo", "shipping", "contact", "dm", "pm", "whatsapp", "telegram"]
        welfare_keywords = ["suffer", "cruel", "captivity", "stress", "neglect", "abuse"]
        conservation_keywords = ["endangered", "extinction", "ecosystem", "biodiversity", "poaching", "trafficking", "invasive", "cites"]
    
        # Create flags
        has_trade = df[TEXT_COL].str.contains("|".join(trade_keywords), case=False, na=False)
        has_welfare = df[TEXT_COL].str.contains("|".join(welfare_keywords), case=False, na=False)
        has_conservation = df[TEXT_COL].str.contains("|".join(conservation_keywords), case=False, na=False)
        has_legal = df[TEXT_COL].str.contains("|".join(["illegal", "ban", "permit", "cites"]), case=False, na=False)
    
        # High priority: trade + (welfare or conservation risk)
        high_priority = (has_trade) & ((has_welfare) | (has_conservation))
    
        st.subheader("Risk Profile Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts with Trade Signals", int(has_trade.sum()))
        col2.metric("Posts with Welfare Concern", int(has_welfare.sum()))
        col3.metric("Posts with Conservation Concern", int(has_conservation.sum()))
        col4.metric("🚨 High Priority Posts", int(high_priority.sum()))
    
        st.subheader("High-Priority Posts (Trade + Welfare/Conservation Risk)")
    
        if high_priority.sum() > 0:
            priority_posts = df[high_priority].copy()
            priority_posts["has_welfare_risk"] = priority_posts[TEXT_COL].str.contains("|".join(welfare_keywords), case=False, na=False)
            priority_posts["has_conservation_risk"] = priority_posts[TEXT_COL].str.contains("|".join(conservation_keywords), case=False, na=False)
            priority_posts["risk_type"] = priority_posts.apply(
                lambda row: "Welfare+Trade" if row["has_welfare_risk"] and not row["has_conservation_risk"] 
                           else "Conservation+Trade" if row["has_conservation_risk"] and not row["has_welfare_risk"]
                           else "Welfare+Conservation+Trade",
                axis=1
            )
        
            # Display table
            display_cols = ["title"] if "title" in priority_posts.columns else [TEXT_COL]
            if "platform" in priority_posts.columns:
                display_cols.append("platform")
            if "source_url" in priority_posts.columns:
                display_cols.append("source_url")
            display_cols.extend(["sentiment_label", "conservation_risk_label", "risk_type"])
        
            st.dataframe(priority_posts[display_cols].head(50), use_container_width=True)
        
            # Download high-priority posts
            st.download_button(
                label="Download High-Priority Posts CSV",
                data=priority_posts.to_csv(index=False),
                file_name="high_priority_posts_wwf.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No high-priority posts found combining trade and risk signals.")
    
        st.subheader("What Makes These Posts High Priority?")
        st.write("""
        - **Trade Signals**: Posts mentioning commercial channels (for sale, breeder, shipping, contact info)
        - **Welfare Risk**: Posts discussing animal suffering, cruelty, poor captivity conditions
        - **Conservation Risk**: Posts mentioning endangered species, extinction, ecosystem impact
        - **Combined Signal**: Post combines BOTH a trade signal AND a welfare/conservation concern
    
        These are the posts where WWF's messaging would be most strategic—intervening where people are discussing purchasing while simultaneously expressing concern about harm.
        """)
    # ==================== TAB 7: DETAILED VISUAL DASHBOARD ====================
    with tab_detail:
        st.subheader("📈 Detailed Visual Dashboard")
        st.caption("Integrated visual summaries from final_dashboard.py with automatic file and column checks.")
        st.info(f"Current source: {detail_visual_source}")

        use_uploaded = detail_visual_source == "Uploaded Data Visuals"

        # --------- Legality charts ---------
        st.markdown("### Legality")
        legality_overall = None
        legality_pure = None
        if not use_uploaded:
            legality_overall = load_optional_csv("data/wwf_legality_overall_summary.csv")
            legality_pure = load_optional_csv("data/wwf_legality_pure_summary.csv")

        c1, c2 = st.columns(2)
        with c1:
            if legality_overall is not None and {"Type", "Count"}.issubset(legality_overall.columns):
                fig_legality_overall = px.bar(
                    legality_overall,
                    x="Type",
                    y="Count",
                    text="Percentage_of_full_dataset" if "Percentage_of_full_dataset" in legality_overall.columns else None,
                    title="Legality Classification Across Full Dataset",
                )
                fig_legality_overall.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_legality_overall, use_container_width=True)
            else:
                st.info("Legality overall summary is unavailable for this source or missing required columns.")

        with c2:
            if legality_pure is not None and {"Type", "Count"}.issubset(legality_pure.columns):
                fig_legality_pure = px.bar(
                    legality_pure,
                    x="Type",
                    y="Count",
                    text="Percentage_within_legality" if "Percentage_within_legality" in legality_pure.columns else None,
                    title="Legality Awareness vs Legal Risk",
                )
                fig_legality_pure.update_traces(texttemplate="%{text}%", textposition="outside")
                st.plotly_chart(fig_legality_pure, use_container_width=True)
            else:
                st.info("Legality pure summary is unavailable for this source or missing required columns.")

        legality_download = None
        if legality_overall is not None and legality_pure is not None:
            legality_download = pd.concat(
                [
                    legality_overall.assign(Section="Legality Overall"),
                    legality_pure.assign(Section="Legality Pure"),
                ],
                ignore_index=True,
            )
        if legality_download is not None:
            st.download_button(
                label="Download Legality Chart Data (CSV)",
                data=legality_download.to_csv(index=False).encode("utf-8"),
                file_name="detail_legality_chart_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # --------- Q1 sentiment charts ---------
        st.markdown("### Q1 Sentiment Detail")
        q1_summary = build_uploaded_q1_summary(df) if use_uploaded else load_optional_csv("data/Q1_sentiment_summary.csv")
        if q1_summary is not None and {"sentiment_label", "comment_count"}.issubset(q1_summary.columns):
            s1, s2 = st.columns(2)
            with s1:
                fig_q1_bar = px.bar(
                    q1_summary,
                    x="sentiment_label",
                    y="comment_count",
                    title="Q1 Overall Sentiment Distribution",
                    color="sentiment_label",
                    category_orders={"sentiment_label": SENTIMENT_ORDER},
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                )
                st.plotly_chart(fig_q1_bar, use_container_width=True)
            with s2:
                fig_q1_pie = px.pie(
                    q1_summary,
                    names="sentiment_label",
                    values="comment_count",
                    title="Q1 Share of Overall Sentiment",
                    color="sentiment_label",
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                )
                st.plotly_chart(fig_q1_pie, use_container_width=True)
            st.download_button(
                label="Download Q1 Sentiment Chart Data (CSV)",
                data=q1_summary.to_csv(index=False).encode("utf-8"),
                file_name="detail_q1_sentiment_chart_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Q1 sentiment summary is unavailable or missing required columns.")

        # --------- Q2 category sentiment stacked chart ---------
        st.markdown("### Q2 Category Sentiment")
        q2_percent = build_uploaded_q2_percent(df, TEXT_COL) if use_uploaded else load_optional_csv("data/Q2_final_category_sentiment_percent.csv")
        if q2_percent is not None and {"Category", "Positive_%", "Neutral_%", "Negative_%"}.issubset(q2_percent.columns):
            q2_melt = q2_percent.melt(
                id_vars="Category",
                value_vars=["Positive_%", "Neutral_%", "Negative_%"],
                var_name="Sentiment",
                value_name="Percentage",
            )
            q2_melt["Sentiment"] = q2_melt["Sentiment"].str.replace("_%", "", regex=False)
            fig_q2_stack = px.bar(
                q2_melt,
                x="Category",
                y="Percentage",
                color="Sentiment",
                barmode="stack",
                category_orders={"Sentiment": ["Negative", "Neutral", "Positive"]},
                color_discrete_map=SENTIMENT_COLOR_MAP,
                title="Sentiment Distribution Across Exotic Pet Categories (%)",
            )
            st.plotly_chart(fig_q2_stack, use_container_width=True)
            st.download_button(
                label="Download Q2 Category Chart Data (CSV)",
                data=q2_melt.to_csv(index=False).encode("utf-8"),
                file_name="detail_q2_category_chart_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Q2 category sentiment percent file is unavailable or missing required columns.")

        # --------- Knowledge gap by category ---------
        st.markdown("### Knowledge Gap by Category")
        gap_pct = None if use_uploaded else load_optional_csv("data/knowledge_gap_category_percentages.csv")
        if gap_pct is not None and len(gap_pct.columns) > 1:
            first_col = gap_pct.columns[0]
            if first_col != "Knowledge Gap Type":
                gap_pct = gap_pct.rename(columns={first_col: "Knowledge Gap Type"})

            gap_melt = gap_pct.melt(
                id_vars="Knowledge Gap Type",
                var_name="Category",
                value_name="Percentage",
            )
            fig_gap = px.bar(
                gap_melt,
                x="Knowledge Gap Type",
                y="Percentage",
                color="Category",
                barmode="stack",
                title="Pet Category Distribution Within Each Knowledge Gap Type (%)",
            )
            st.plotly_chart(fig_gap, use_container_width=True)
            st.download_button(
                label="Download Knowledge Gap Chart Data (CSV)",
                data=gap_melt.to_csv(index=False).encode("utf-8"),
                file_name="detail_knowledge_gap_chart_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Knowledge gap percentages are available only for Final Data Visuals.")

        # --------- Additional topic summaries ---------
        st.markdown("### Additional Topic Summaries")
        topic_file_map = {
            "Motivation Classification": ("data/wwf_motivation_overall_summary.csv", "Type", "Count", "Percentage_of_full_dataset"),
            "Marketplace Signals": ("data/wwf_market_purchase_overall_summary.csv", "Purchase Type", "Count", "Percentage_of_28k"),
            "Conservation Overall": ("data/wwf_conservation_overall_summary.csv", "Type", "Count", "Percentage_of_full_dataset"),
            "Conservation Subthemes": ("data/wwf_conservation_subtheme_summary.csv", "Conservation Subtheme", "Count", "Percentage_within_conservation"),
        }

        for title, (file_path, x_col, y_col, pct_col) in topic_file_map.items():
            if use_uploaded:
                st.info(f"{title}: available only for Final Data Visuals.")
                continue

            topic_df = load_optional_csv(file_path)
            if topic_df is None or not {x_col, y_col}.issubset(topic_df.columns):
                st.info(f"{title}: file unavailable or missing required columns.")
                continue

            fig_topic = px.bar(
                topic_df,
                x=x_col,
                y=y_col,
                text=pct_col if pct_col in topic_df.columns else None,
                title=title,
            )
            if pct_col in topic_df.columns:
                fig_topic.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(fig_topic, use_container_width=True)
            st.download_button(
                label=f"Download {title} Chart Data (CSV)",
                data=topic_df.to_csv(index=False).encode("utf-8"),
                file_name=f"detail_{title.lower().replace(' ', '_')}_chart_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ==================== TAB 7: EXPERIMENTAL & ADVANCED ANALYTICS ====================
    with tab_experimental:
        st.write("👋 Welcome to the experimental and advanced analytics section. This tab includes wildlife NER, detailed risk analysis, intervention simulations, and ANOVA workflows.")
    
        # Subtab selection
        exp_subtab = st.radio("Select analysis type:", 
                              ["Wildlife NER & Risk Details", "Intervention Simulation", "ANOVA Analysis"],
                              horizontal=True)
    
        if exp_subtab == "Wildlife NER & Risk Details":
            st.subheader("🔍 Wildlife-Aware Named Entity Recognition")
            st.caption("Custom lexicon-based extraction for species, legal, welfare, conservation, safety, and trade signals.")

            ner_summary = extract_wildlife_entities(df[TEXT_COL].tolist())
            if ner_summary.empty:
                st.info("No wildlife entities found in the uploaded text.")
            else:
                ner_label_options = ["All"] + sorted(ner_summary["label"].unique().tolist())
                ner_label = st.selectbox("Filter entity label", ner_label_options, key="ner_label_filter")

                filtered_ner = ner_summary if ner_label == "All" else ner_summary[ner_summary["label"] == ner_label]
                st.dataframe(filtered_ner.head(30), use_container_width=True)

                if not filtered_ner.empty:
                    fig_ner = px.bar(
                        filtered_ner.head(20),
                        x="entity",
                        y="count",
                        color="label",
                        title="Top Wildlife Entities",
                    )
                    st.plotly_chart(fig_ner, use_container_width=True)

                st.download_button(
                    label="Download entity summary CSV",
                    data=filtered_ner.to_csv(index=False).encode("utf-8"),
                    file_name="wildlife_entity_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.subheader("Conservation-Risk Classifier Details")
            risk_classifier = train_conservation_risk_classifier(df[TEXT_COL].tolist())
            risk_df_detail = predict_conservation_risk(df[TEXT_COL].tolist(), risk_classifier)

            risk_col1, risk_col2, risk_col3 = st.columns(3)
            risk_col1.metric("High-risk items", int((risk_df_detail["risk_label"] == "high").sum()))
            risk_col2.metric("Medium-risk items", int((risk_df_detail["risk_label"] == "medium").sum()))
            risk_col3.metric("Low-risk items", int((risk_df_detail["risk_label"] == "low").sum()))

            if getattr(risk_classifier, "accuracy", None) is not None:
                st.metric("Risk classifier accuracy", f"{risk_classifier.accuracy:.3f}")
            st.caption(f"Risk mode: {risk_classifier.mode}")

            fig_risk = px.histogram(
                risk_df_detail,
                x="risk_label",
                color="risk_label",
                category_orders={"risk_label": ["low", "medium", "high"]},
                title="Conservation Risk Distribution",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            st.download_button(
                label="Download risk predictions CSV",
                data=risk_df_detail.to_csv(index=False).encode("utf-8"),
                file_name="conservation_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

            high_risk_examples = risk_df_detail[risk_df_detail["risk_label"] == "high"].head(10)
            if not high_risk_examples.empty:
                st.subheader("High-Risk Examples")
                st.dataframe(high_risk_examples[["text", "risk_score", "matched_terms"]], use_container_width=True)
    
    
        elif exp_subtab == "Intervention Simulation":
            st.subheader("💬 Simulate Communication Intervention")
            st.caption("How could different framing messages affect sentiment toward exotic pets?")
        
            comment_type = st.selectbox(
                "Select intervention type:",
                ["Emotional", "Cognitive", "Conservation", "Behavioral", "Neutral"],
                key="condition_select",
            )

            comments = {
                "Emotional": "Exotic pets often suffer severe psychological distress in captivity.",
                "Cognitive": "Many exotic pets are illegal and require specialized care.",
                "Conservation": "Escaped exotic animals can destroy ecosystems.",
                "Behavioral": "Support conservation efforts instead of buying exotic pets.",
                "Neutral": "Exotic pets are non-domesticated animals kept in homes.",
            }
            st.info(f"**Sample message**: {comments[comment_type]}")

            condition_shift = {
                "Emotional": -0.18,
                "Cognitive": -0.10,
                "Conservation": -0.14,
                "Behavioral": -0.08,
                "Neutral": 0.00,
            }
            condition_keywords = {
                "Emotional": ["suffer", "stress", "pain", "cruel", "captivity", "abuse"],
                "Cognitive": ["illegal", "law", "permit", "regulated", "risk", "care"],
                "Conservation": ["ecosystem", "biodiversity", "endangered", "extinction", "wildlife"],
                "Behavioral": ["adopt", "responsibility", "training", "commitment", "rehome"],
                "Neutral": [],
            }

            selected_keywords = condition_keywords[comment_type]
            if selected_keywords:
                keyword_pattern = "|".join(selected_keywords)
                exposure_mask = df[TEXT_COL].str.contains(keyword_pattern, case=False, na=False)
            else:
                exposure_mask = pd.Series(False, index=df.index)

            delta = condition_shift[comment_type]
            df["sentiment_adjusted"] = np.where(exposure_mask, df["sentiment"] + delta, df["sentiment"])
            df["sentiment_adjusted"] = df["sentiment_adjusted"].clip(-1, 1)
            df["sentiment_adjusted_label"] = df["sentiment_adjusted"].apply(label_sentiment)

            before_after = pd.concat(
                [
                    df[["sentiment_label"]].rename(columns={"sentiment_label": "label"}).assign(stage="Before"),
                    df[["sentiment_adjusted_label"]].rename(columns={"sentiment_adjusted_label": "label"}).assign(stage="After"),
                ]
            )
            summary = before_after.groupby(["stage", "label"]).size().reset_index(name="count")

            fig_cond = px.bar(
                summary,
                x="label",
                y="count",
                color="stage",
                barmode="group",
                category_orders={"label": ["Negative", "Neutral", "Positive"]},
                title=f"Sentiment Impact: {comment_type} Intervention",
            )
            st.plotly_chart(fig_cond, use_container_width=True)

            impact_col1, impact_col2 = st.columns(2)
            impact_col1.metric("Posts Exposed To This Message", int(exposure_mask.sum()))
            impact_col2.metric("Avg Sentiment Shift", round(float((df["sentiment_adjusted"] - df["sentiment"]).mean()), 4))
    
        elif exp_subtab == "ANOVA Analysis":
            st.subheader("📊 Statistical Analysis (ANOVA)")
            st.caption("Run experimental ANOVA workflows on condition effects and outcomes.")
        
            exp_file = st.file_uploader("Upload Experimental Data (optional)", type=["csv"], key="exp_file")

            comments = {
                "Emotional": "Exotic pets often suffer severe psychological distress in captivity.",
                "Cognitive": "Many exotic pets are illegal and require specialized care.",
                "Conservation": "Escaped exotic animals can destroy ecosystems.",
                "Behavioral": "Support conservation efforts instead of buying exotic pets.",
                "Neutral": "Exotic pets are non-domesticated animals kept in homes.",
            }

            if exp_file:
                df_exp = pd.read_csv(exp_file)
            else:
                np.random.seed(42)
                df_exp = pd.DataFrame(
                    {
                        "participant": np.arange(120),
                        "condition": np.random.choice(list(comments.keys()), 120),
                        "time": np.tile(["pre", "post"], 60),
                        "attitude": np.random.normal(3, 1, 120),
                        "desire": np.random.normal(3, 1, 120),
                        "civic_action": np.random.normal(3, 1, 120),
                    }
                )

            st.write("**Data preview:**")
            st.dataframe(df_exp.head(10), use_container_width=True)

            st.subheader("Mixed ANOVA Results")
            required_mixed = {"participant", "condition", "time", "attitude"}
            if STATS:
                if required_mixed.issubset(df_exp.columns):
                    try:
                        aov = pg.mixed_anova(
                            dv="attitude",
                            within="time",
                            between="condition",
                            subject="participant",
                            data=df_exp,
                        )
                        st.write("**Attitude ANOVA Results**")
                        st.dataframe(aov, use_container_width=True)
                    except Exception as e:
                        st.error(f"ANOVA error: {e}")
                else:
                    st.warning("Experimental data missing columns for mixed ANOVA.")
            else:
                st.warning("Install pingouin for ANOVA: pip install pingouin")

            st.subheader("Civic Action ANOVA")
            required_one_way = {"condition", "civic_action"}
            if STATS and required_one_way.issubset(df_exp.columns):
                try:
                    aov2 = pg.anova(dv="civic_action", between="condition", data=df_exp)
                    st.write("**Civic Action ANOVA Results**")
                    st.dataframe(aov2, use_container_width=True)
                except Exception as e:
                    st.error(f"ANOVA error: {e}")

if dashboard_mode == "WWF Final Data Dashboard":
    render_wwf_dashboard()
else:
    render_uploaded_dashboard()
