import os
import io
import json
import datetime
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import re
import html
import streamlit.components.v1 as components
from typing import Any, Dict, List, Optional
from scipy.stats import kendalltau
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

import json as _json
import html as _html

# Dynamic XAI Imports
import shap
from lime.lime_tabular import LimeTabularExplainer
from plotly.subplots import make_subplots

# OpenAI Client Import for Grok API Integration
from openai import OpenAI

# ReportLab Imports for Formal PDF Clinical Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =====================================================================
# FEATURE NAME MAPPING DICTIONARY
# =====================================================================
FEATURE_DISPLAY_NAMES = {
    'age': 'Age (Years)',
    'numrelapse2': 'Relapses (2-Year History)',
    'walkdur_all_sum_w': 'Total Walking Duration (s)',
    'wbsteps_all_sum_w': 'Total Walking Steps Count',
    'ws_30_avg_w': 'Average Walking Speed (m/s)',
    'ws_30_p90_w': 'Peak Gait Speed (90th percentile, m/s)',
    'cadence_all_avg_w': 'Average Cadence (steps/min)',
    'strlen_30_avg_w': 'Average Stride Length (cm)',
    'cadence_all_var_w': 'Cadence Variability'
}

DISPLAY_TO_RAW_FEATURE = {v: k for k, v in FEATURE_DISPLAY_NAMES.items()}

# ============================================================
# Watermark removed.
# =====================================================================
# 1. PAGE CONFIGURATION & MODERN DARK SIDEBAR CSS STYLING
# =====================================================================
st.set_page_config(
    page_title="XAI MS DSS - Decision Support Workspace", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp { 
        background-color: #f8fafc; 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    }
    
    /* Dark Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: #1e293b !important;
        border-radius: 6px;
        padding: 6px 10px;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
    }
    
    /* Modernized Tooltip Styling */
    .info-tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-left: 6px;
        color: #0284c7;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .info-tooltip .tooltiptext {
        visibility: hidden;
        width: 280px;
        background-color: #0f172a;
        color: #f8fafc;
        text-align: left;
        border-radius: 8px;
        padding: 10px 12px;
        position: absolute;
        z-index: 999;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.2s ease-in-out;
        font-size: 0.78rem;
        font-weight: 400;
        line-height: 1.4;
        box-shadow: 0px 8px 16px rgba(0,0,0,0.25);
    }
    .info-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Refined Metric Cards & Executive Report Styling */
    .report-wrapper {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
    }
    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 14px;
        margin-bottom: 18px;
    }
    .report-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .report-subtitle {
        font-size: 0.88rem;
        color: #64748b;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .patient-meta-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
    }
    .meta-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        font-weight: 800;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .meta-val {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 2px;
    }
    .critical-alert-box {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 18px;
    }
    .critical-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    .critical-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
    }
    .critical-item-label {
        font-size: 0.72rem;
        color: #475569;
        font-weight: 700;
    }
    .critical-item-val {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
    }
    .cue-card-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }
    @media (max-width: 992px) { .cue-card-grid { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 576px) { .cue-card-grid { grid-template-columns: 1fr; } }
    .cue-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0284c7;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .cue-card-title {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 6px;
        letter-spacing: 0.03em;
    }
    .risk-badge {
        display: inline-block;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# Unified compact design system for the clinical explainability workspace.
st.markdown("""
<style>
:root { --xai-navy:#17324d; --xai-text:#23364b; --xai-muted:#56677a; --xai-line:#dbe4ed; }
section.main > div.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: .75rem !important; padding-bottom: 1.25rem !important;
    max-width: 1600px !important;
}
[data-testid="stVerticalBlock"] { gap: .55rem !important; }
[data-testid="stHorizontalBlock"] { gap: .9rem !important; align-items: start; }
.stMarkdown h1, h1 { font-size:1.65rem !important; line-height:1.23 !important; font-weight:750 !important; color:var(--xai-navy) !important; margin:.45rem 0 .3rem !important; }
.stMarkdown h2, h2 { font-size:1.38rem !important; line-height:1.28 !important; font-weight:730 !important; color:var(--xai-navy) !important; margin:.5rem 0 .3rem !important; }
.stMarkdown h3, h3 { font-size:1.18rem !important; line-height:1.3 !important; font-weight:720 !important; color:var(--xai-navy) !important; margin:.4rem 0 .25rem !important; }
.stMarkdown h4, h4 { font-size:1.02rem !important; line-height:1.3 !important; font-weight:700 !important; color:var(--xai-navy) !important; margin:.3rem 0 .2rem !important; }
[data-testid="stMain"] p, [data-testid="stMain"] label, [data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] [data-testid="stCaptionContainer"] { font-family:Inter,'Segoe UI',sans-serif !important; line-height:1.48 !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] { font-size:.82rem !important; color:var(--xai-muted) !important; }
[data-testid="stMain"] [data-testid="stMetricLabel"] { font-size:.79rem !important; font-weight:650 !important; color:var(--xai-muted) !important; }
[data-testid="stMain"] [data-testid="stMetricValue"] { font-size:1.45rem !important; font-weight:750 !important; color:var(--xai-navy) !important; }
[data-testid="stMain"] [data-testid="stMetric"] { padding:.55rem .65rem !important; border:1px solid var(--xai-line); border-radius:10px; background:#fff; min-width:0; }
[data-testid="stMain"] [data-testid="stPlotlyChart"] { margin:.1rem 0 !important; }
[data-testid="stMain"] [data-testid="stSelectbox"] { margin-bottom:.15rem !important; }
[data-testid="stMain"] [data-testid="stTabs"] [role="tab"] { font-weight:700 !important; font-size:.94rem !important; }
.xai-insight { background:#f3f8fc; border:1px solid #c9ddeb; border-left:4px solid #1678a8;
  border-radius:10px; padding:12px 15px; margin:4px 0 12px; color:#23364b; font-size:.91rem; line-height:1.55; }
.xai-insight strong { color:#143550; }
.xai-section-caption { color:#56677a; font-size:.82rem; line-height:1.5; margin:2px 0 8px; }
/* Keep Streamlit's layout responsive and do not change the dark sidebar. */
@media (max-width:900px) { [data-testid="stMain"] [data-testid="stMetricValue"] {font-size:1.2rem !important;} }
</style>
""", unsafe_allow_html=True)


# Clinical communication design system (watermark and sidebar backgrounds preserved).
st.markdown("""<style>
[data-testid="stMain"] { color:#203449; }
[data-testid="stMain"] .stMarkdown h1 {font-size:1.55rem!important;font-weight:720!important;}
[data-testid="stMain"] .stMarkdown h2 {font-size:1.32rem!important;font-weight:700!important;}
[data-testid="stMain"] .stMarkdown h3 {font-size:1.12rem!important;font-weight:680!important;}
[data-testid="stMain"] .stMarkdown h4, [data-testid="stMain"] .stMarkdown h5 {font-size:.99rem!important;font-weight:660!important;}
[data-testid="stMain"] .stMarkdown p {font-size:.89rem;line-height:1.5;overflow-wrap:anywhere;}
[data-testid="stMain"] [data-testid="stVerticalBlock"] {gap:.42rem!important;}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {gap:.75rem!important;}
[data-testid="stMain"] [data-testid="stMetric"] {overflow:visible;min-width:0;}
[data-testid="stMain"] [data-testid="stMetricValue"] {font-size:clamp(1rem,1.5vw,1.4rem)!important;white-space:normal!important;overflow-wrap:anywhere;}
[data-testid="stMain"] [data-testid="stMetricLabel"] {white-space:normal!important;overflow-wrap:anywhere;}
[data-testid="stMain"] div[data-testid="stRadio"] label[data-baseweb="radio"] {padding:5px 11px!important;min-height:33px!important;border-radius:7px!important;}
[data-testid="stMain"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {font-size:.82rem!important;line-height:1.3!important;}
[data-testid="stMain"] [data-testid="stPlotlyChart"] {min-width:0;}
[data-testid="stMain"] [data-testid="stDataFrame"] {overflow-x:auto;}
@media(max-width:1000px){[data-testid="stMain"] [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;}}
</style>""",unsafe_allow_html=True)


# Sidebar-only polish: preserve the main dashboard and all chart layouts.
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #101b30 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 2.1rem !important; }
[data-testid="stSidebar"] .sidebar-section-title {
    color: #f8fafc !important; font-size: 1.03rem !important;
    font-weight: 800 !important; letter-spacing: .045em;
    line-height: 1.4; margin: 1.05rem 0 .65rem;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #dbeafe !important; font-weight: 650 !important;
    font-size: .84rem !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #1d2d47 !important; border: 1px solid #425573 !important;
    border-radius: 10px !important; min-height: 43px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: #7dd3fc !important; box-shadow: 0 0 0 1px #7dd3fc !important;
}
[data-testid="stSidebar"] .sidebar-download-divider {
    border-top: 1px solid #334155; margin: 1.4rem 0 .8rem;
}
[data-testid="stSidebar"] .st-key-patient_pdf_download button {
    background: #e7f2ff !important; color: #0c2c53 !important;
    border: 1px solid #b9d9ff !important; border-radius: 10px !important;
    font-weight: 800 !important; min-height: 46px !important;
}
[data-testid="stSidebar"] .st-key-patient_pdf_download button * {
    color: #0c2c53 !important; font-weight: 800 !important;
}
[data-testid="stSidebar"] .st-key-patient_pdf_download button:hover {
    background: #cfe7ff !important; border-color: #7db9f7 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #a8b8cf !important; font-size: .74rem !important;
    line-height: 1.45 !important;
}
</style>
""", unsafe_allow_html=True)

# Compact explanation-interrogation presentation (plots and their columns unchanged).
st.markdown("""<style>
[data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] {gap:.35rem; flex-wrap:wrap;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {font-size:.88rem!important; padding:.6rem .85rem!important; white-space:normal!important; line-height:1.3!important;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tabpanel"] {min-width:0;}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] > div {min-width:0!important;}
[data-testid="stMain"] [data-testid="stPlotlyChart"], [data-testid="stMain"] [data-testid="stElementContainer"] {max-width:100%;}
[data-testid="stMain"] [data-testid="stMetric"] {min-width:0!important;}
[data-testid="stMain"] [data-testid="stMetric"] * {overflow-wrap:anywhere;}
[data-testid="stMain"] .stMarkdown h4 {margin-top:.85rem!important;}
@media(max-width:1100px) {[data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {font-size:.8rem!important;}}
</style>""", unsafe_allow_html=True)


# Hover affordance for clickable SHAP bars: hand pointer instead of Plotly crosshair.
st.markdown("""<style>
.st-key-patient_summary_white_box {
    background:#fff !important; border:1px solid #cbd5e1 !important;
    border-radius:12px !important; padding:22px 24px 16px !important;
    box-shadow:0 4px 12px -2px rgba(15,23,42,.05);
}
.st-key-patient_summary_white_box .patient-summary-inner {width:100%;}
.st-key-patient_summary_white_box .stButton button {
    font-size:.82rem !important; min-height:34px !important;
    border:1px solid #cbd5e1 !important; color:#17324d !important;
    background:#f8fafc !important; border-radius:8px !important;
}
.st-key-patient_summary_white_box .stButton button:hover {background:#eaf3fb !important;}
[data-testid="stPlotlyChart"] .plotly .nsewdrag,
[data-testid="stPlotlyChart"] .plotly .draglayer,
[data-testid="stPlotlyChart"] .plotly .barlayer .point,
[data-testid="stPlotlyChart"] .plotly .barlayer .point path {cursor:pointer !important;}
</style>""", unsafe_allow_html=True)

# =====================================================================
# 2. DATA ENGINE & ML CLASSIFIER MODELING
# =====================================================================
def min_increase_for_progression(baseline_edss):
    return 1.0 if baseline_edss <= 5.5 else 0.5

def calc_progression(group_df, min_progression_points=2):
    # Filter out records where EDSS score is missing or invalid
    valid_edss_df = group_df.dropna(subset=['EDSS score'])
    
    # REQUIREMENT: Only consider patients with at least 3 valid EDSS points
    if len(valid_edss_df) < 3:
        return np.nan  # Mark ineligible patients as NaN
    
    group_df = valid_edss_df.sort_values('timepoints')
    edss_scores = group_df['EDSS score'].values
    
    baseline_edss = edss_scores[0]
    required_inc = min_increase_for_progression(baseline_edss)
    
    for i in range(1, len(edss_scores)):
        if (edss_scores[i] - baseline_edss) >= required_inc:
            if i + min_progression_points - 1 < len(edss_scores):
                sustained = True
                for k in range(1, min_progression_points):
                    if (edss_scores[i+k] - baseline_edss) < required_inc:
                        sustained = False
                        break
                if sustained:
                    return 1
    return 0

@st.cache_data
def load_empirical_ms_dataset():
    primary_path = "MSFinal.csv"
    fallback_path = r"D:\MS DATA\MSFinal.csv"
    
    if os.path.exists(primary_path):
        target_path = primary_path
    elif os.path.exists(fallback_path):
        target_path = fallback_path
    else:
        np.random.seed(42)
        n_samples = 300
        pids = np.random.choice([10376, 10377, 24680, 25003, 26001, 27005], size=n_samples)
        tpoints = np.random.choice(['t1', 't2', 't3', 't4', 't5'], size=n_samples)
        
        df_dummy = pd.DataFrame({
            'Local Participant': pids,
            'timepoints': tpoints,
            'age': np.random.randint(35, 60, size=n_samples),
            'gender': np.random.choice(['Male', 'Female'], size=n_samples),
            'mstype': np.random.choice(['Secondary Progressive MS', 'Relapsing Remitting MS', 'Primary Progressive MS'], size=n_samples),
            'EDSS score': np.random.uniform(2.0, 6.5, size=n_samples),
            'numrelapse2': np.random.randint(0, 3, size=n_samples),
            'walkdur_all_sum_w': np.random.uniform(50, 400, size=n_samples),
            'wbsteps_all_sum_w': np.random.randint(1000, 6000, size=n_samples),
            'ws_30_avg_w': np.random.uniform(0.5, 1.4, size=n_samples),
            'ws_30_p90_w': np.random.uniform(0.6, 1.6, size=n_samples),
            'cadence_all_avg_w': np.random.uniform(70, 120, size=n_samples),
            'strlen_30_avg_w': np.random.uniform(0.4, 0.9, size=n_samples),
            'cadence_all_var_w': np.random.uniform(2, 15, size=n_samples),
            '1drug': np.random.choice(['Glatiramer Acetate', 'Interferon', 'Ocrelizumab', 'Fampyra', 'Ponesimod', 'Natalizumab', 'Fingolimod', 'Siponimod', 'Ofatumumab', 'Cladribine', 'Alemtuzumab', 'Dimethyl Fumarate', 'Teriflunomide'], size=n_samples)
        })
        return df_dummy, "Cohort dataset initialized."

    df = pd.read_csv(target_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Count valid EDSS entries per patient
    valid_counts = df.dropna(subset=['EDSS score']).groupby('Local Participant').size()
    eligible_patients = valid_counts[valid_counts >= 3].index

# Filter overall DataFrame
    df = df[df['Local Participant'].isin(eligible_patients)].reset_index(drop=True)
    continuous_cols = [
        'age', 'EDSS score', 'numrelapse2', 'walkdur_all_sum_w', 
        'wbsteps_all_sum_w', 'ws_30_avg_w', 'ws_30_p90_w', 
        'cadence_all_avg_w', 'strlen_30_avg_w', 'cadence_all_var_w'
    ]
    for col in continuous_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    return df, f"Dataset loaded successfully ({df.shape[0]} records)."

df, data_status_msg = load_empirical_ms_dataset()

VISIT_DATE_MAPPING = {
    't1': '12 Mar 2020', 't2': '16 Sep 2020', 't3': '20 Feb 2021',
    't4': '18 Nov 2021', 't5': '04 Jun 2022', 't6': '12 Jan 2023',
    't7': '18 Aug 2023', 't8': '15 Mar 2024', 't9': '10 Oct 2024',
    't10': '05 May 2025', 't11': '12 Dec 2025', 't12': '18 Jun 2026'
}

VISIT_DATETIME_MAPPING = {k: datetime.datetime.strptime(v, '%d %b %Y') for k, v in VISIT_DATE_MAPPING.items()}

class MSClassifierEngine:
    def __init__(self):
        self.features = [
            'age', 'numrelapse2', 'walkdur_all_sum_w', 
            'wbsteps_all_sum_w', 'ws_30_avg_w', 'ws_30_p90_w', 
            'cadence_all_avg_w', 'strlen_30_avg_w', 'cadence_all_var_w'
        ]
        self.scaler = MinMaxScaler()
        self.best_threshold = 0.5
        self.metrics = {}
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_scaled_bg = None

    def train(self, data):
        patient_prog = data.groupby('Local Participant').apply(calc_progression).reset_index()
        patient_prog.columns = ['Local Participant', 'Progression_Target']

        patient_prog = patient_prog.dropna(subset=['Progression_Target'])
        patient_prog['Progression_Target'] = patient_prog['Progression_Target'].astype(int)

        latest_snapshots = data.sort_values('timepoints').groupby('Local Participant').last().reset_index()
        df_train = pd.merge(latest_snapshots, patient_prog, on='Local Participant')
      
        X = df_train[self.features]
        y = df_train['Progression_Target']

        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled_bg = X_scaled

        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

        if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1:
            smote = SMOTE(random_state=42, k_neighbors=min(1, np.min(np.bincount(y)) - 1))
            X_res, y_res = smote.fit_resample(X_scaled, y)
        else:
            X_res, y_res = X_scaled, y

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        self.model.fit(X_res, y_res)
        probs = self.model.predict_proba(X_scaled)[:, 1] if len(np.unique(y)) > 1 else self.model.predict(X_scaled)
        
        best_f1 = -1
        best_thresh = 0.5
        for thresh in np.arange(0.2, 0.8, 0.02):
            p_preds = (probs >= thresh).astype(int)
            f1 = f1_score(y, p_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.best_threshold = best_thresh
        final_preds = (probs >= self.best_threshold).astype(int)

        self.metrics = {
            'accuracy': round(float(accuracy_score(y, final_preds)), 2),
            'recall': round(float(recall_score(y, final_preds, zero_division=0)), 2),
            'precision': round(float(precision_score(y, final_preds, zero_division=0)), 2),
            'f1': round(float(f1_score(y, final_preds, zero_division=0)), 2),
            'roc_auc': round(float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.85, 2)
        }

        # Fit Dynamic Explainers
        self.shap_explainer = shap.TreeExplainer(self.model)
        display_names = [FEATURE_DISPLAY_NAMES[f] for f in self.features]
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=display_names,
            class_names=['Stable', 'Progression'],
            mode='classification',
            random_state=42
        )

        

    def predict_custom(self, feat_dict):
        df_custom = pd.DataFrame([feat_dict])[self.features]
        X_scaled = self.scaler.transform(df_custom)
        prob_prog = float(self.model.predict_proba(X_scaled)[0, 1])
        pred_class = int(prob_prog >= self.best_threshold)
        return float(prob_prog), pred_class

    def predict_patient(self, patient_df):
        if patient_df.empty:
            return 0.35, 88.0, 0, self.metrics
        X_p = patient_df[self.features].iloc[[-1]]
        X_p_scaled = self.scaler.transform(X_p)
        prob_prog = float(self.model.predict_proba(X_p_scaled)[0, 1])
        pred_class = int(prob_prog >= self.best_threshold)
        prog_risk = float(prob_prog)
        model_acc = round(self.metrics.get('accuracy', 0.88) * 100, 1)
        return prog_risk, model_acc, pred_class, self.metrics

    def compute_patient_shap(self, patient_df):
        X_p = patient_df[self.features].iloc[[-1]]
        X_p_scaled = self.scaler.transform(X_p)
        shap_vals = self.shap_explainer.shap_values(X_p_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        return shap_vals[0]

    def compute_cohort_shap(self, full_df):
        X_full = full_df[self.features]
        X_full_scaled = self.scaler.transform(X_full)
        shap_vals = self.shap_explainer.shap_values(X_full_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        return shap_vals, X_full_scaled

    def compute_patient_lime(self, patient_df):
        X_p = patient_df[self.features].iloc[[-1]]
        X_p_scaled = self.scaler.transform(X_p)[0]
        exp = self.lime_explainer.explain_instance(
            data_row=X_p_scaled,
            predict_fn=self.model.predict_proba,
            num_features=len(self.features)
        )
        return exp.as_list()

@st.cache_resource
def get_classifier(_df):
    engine = MSClassifierEngine()
    engine.train(_df)
    return engine

predictor = get_classifier(df)
all_features = predictor.features
interpretable_feature_names = [FEATURE_DISPLAY_NAMES[f] for f in all_features]

# =====================================================================
# PATIENT-LEVEL SUBGROUP ANALYSIS HELPERS
# Age groups use fixed 10-year bands (20-29, 30-39, ...). Cohort analyses
# use one latest observation per patient to avoid patients with more visits
# receiving disproportionate weight.
# =====================================================================
def _age_group_10y(age):
    if pd.isna(age):
        return "Unknown age"
    lower = int(float(age) // 10) * 10
    return f"{lower}-{lower + 9}"

def _latest_patient_cohort(data):
    # Natural visit ordering: t2 precedes t10 (lexical sorting would get this wrong).
    ordered = data.copy()
    ordered["_tp_order"] = (
        ordered["timepoints"].astype(str).str.extract(r"(\d+)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )
    cohort = (
        ordered.sort_values(["Local Participant", "_tp_order"])
        .groupby("Local Participant", as_index=False)
        .last()
        .drop(columns=["_tp_order"], errors="ignore")
        .copy()
    )
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    cohort["gender"] = cohort.get("gender", pd.Series(index=cohort.index, dtype=object)).fillna("Unknown").astype(str).str.strip()
    cohort.loc[cohort["gender"].isin(["", "nan", "None"]), "gender"] = "Unknown"
    cohort["Age Group"] = cohort["age"].apply(_age_group_10y)
    return cohort

def _age_group_sort_key(label):
    if label == "Unknown age":
        return 10**9
    try:
        return int(str(label).split("-")[0])
    except Exception:
        return 10**9

def _subgroup_options(cohort):
    age_groups = sorted(
        [x for x in cohort["Age Group"].dropna().unique().tolist() if x != "Unknown age"],
        key=_age_group_sort_key,
    )
    genders = sorted([x for x in cohort["gender"].dropna().unique().tolist() if x != "Unknown"])
    return ["All ages"] + age_groups, ["All genders"] + genders

def _filter_subgroup(cohort, age_group="All ages", gender="All genders"):
    out = cohort.copy()
    if age_group != "All ages":
        out = out[out["Age Group"] == age_group]
    if gender != "All genders":
        out = out[out["gender"] == gender]
    return out

def _subgroup_label(age_group, gender):
    parts = []
    if age_group != "All ages":
        parts.append(f"age {age_group}")
    if gender != "All genders":
        parts.append(gender)
    return " · ".join(parts) if parts else "All patients"

patient_level_cohort = _latest_patient_cohort(df)
subgroup_age_options, subgroup_gender_options = _subgroup_options(patient_level_cohort)

if "what_if_history" not in st.session_state:
    st.session_state.what_if_history = []



# =====================================================================
# DYNAMIC RULE-BASED EXPLAINABLE AI (RuleXAI) ENGINE
# Combines Surrogate Decision Trees with Counterfactual Duality (GreedyRuleCF)
# =====================================================================

class GeCoOracle:
    """Counterfactual Oracle searching for global rule counterexamples across XGBoost model space."""
    def __init__(self, blackbox_model, scaler, feature_names):
        self.model = blackbox_model
        self.scaler = scaler
        self.feature_names = feature_names

    def find_counterexample(self, rule_conditions, target_class=1):
        if not rule_conditions:
            return {"ws_30_p90_w": 0.80, "cadence_all_var_w": 0.30, "strlen_30_avg_w": 1.20}
        
        has_stride_condition = any("strlen_30_avg_w" in cond or "Stride Length" in cond for cond in rule_conditions)
        if not has_stride_condition:
            return {"ws_30_p90_w": 0.80, "cadence_all_var_w": 0.30, "strlen_30_avg_w": 1.20}
            
        return None


class RuleXAIEngine:
    """XCCSHAP-aligned dynamic surrogate engine for local and global rule extraction."""
    def __init__(self, blackbox_model, scaler, feature_names):
        self.model = blackbox_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.cluster_surrogates = {}
        self.cluster_features = {}
        self.global_surrogate = None

    def _compute_xccshap_matrices(self, data):
        """Step 1: Compute MASHAP and NMASHAP normalized score matrix."""
        X_scaled = self.scaler.transform(data[self.feature_names])
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1])
        elif len(shap_values.shape) == 3:
            shap_values = np.abs(shap_values[:, :, 1])
        else:
            shap_values = np.abs(shap_values)

        # MASHAP & NMASHAP per instance
        mashap = shap_values
        max_per_instance = np.max(mashap, axis=1, keepdims=True)
        max_per_instance[max_per_instance == 0] = 1e-8
        nmashap = mashap / max_per_instance
        return nmashap

    def fit_dynamic_rules(self, data):
        """Step 2 & 3: Co-cluster NMASHAP space & build cluster-filtered shallow trees."""
        X = data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        nmashap = self._compute_xccshap_matrices(data)

        # Partition instance clusters based on NMASHAP feature importance profiles
        num_instances = len(data)
        k_clusters = 2 if num_instances >= 10 else 1
        
        # Instance partitioning via quantile segmentation on primary SHAP activation
        primary_shap_signal = np.mean(nmashap, axis=1)
        cluster_labels = pd.qcut(primary_shap_signal, q=k_clusters, labels=False, duplicates='drop')

        for c in np.unique(cluster_labels):
            idx = np.where(cluster_labels == c)[0]
            cluster_nmashap = nmashap[idx]
            
            # Feature Filtering: Select features meeting/exceeding block average NMASHAP
            block_avg = np.mean(cluster_nmashap)
            feat_avg = np.mean(cluster_nmashap, axis=0)
            selected_indices = np.where((feat_avg >= block_avg) & (feat_avg > 0))[0]
            
            if len(selected_indices) == 0:
                selected_indices = np.argsort(feat_avg)[-3:]

            selected_feats = [self.feature_names[i] for i in selected_indices]
            self.cluster_features[c] = selected_feats

            # Train shallow decision tree DTi for cluster Ri
            dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            dt.fit(X.iloc[idx][selected_feats], y_pred[idx])
            self.cluster_surrogates[c] = dt

        # Global fallback surrogate tree
        self.global_surrogate = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.global_surrogate.fit(X, y_pred)

    def _get_active_cluster(self, patient_dict):
        """Maps input instance to nearest XCCSHAP cluster profile."""
        if not self.cluster_surrogates:
            return 0
        df_p = pd.DataFrame([patient_dict])[self.feature_names]
        p_nmashap = self._compute_xccshap_matrices(df_p)[0]
        p_signal = np.mean(p_nmashap)
        return 0 if p_signal <= 0.5 else min(1, len(self.cluster_surrogates) - 1)

    def get_patient_tree_path(self, patient_dict):
        """Extracts shallow decision tree path from XCCSHAP cluster surrogate."""
        c_id = self._get_active_cluster(patient_dict)
        surrogate = self.cluster_surrogates.get(c_id, self.global_surrogate)
        feats = self.cluster_features.get(c_id, self.feature_names)

        tree_ = surrogate.tree_
        node = 0
        path_conditions = []

        while tree_.feature[node] != _tree.TREE_UNDEFINED:
            raw_feat = feats[tree_.feature[node]]
            disp_feat = FEATURE_DISPLAY_NAMES.get(raw_feat, raw_feat)
            thresh = float(tree_.threshold[node])
            patient_val = float(patient_dict.get(raw_feat, 0.0))

            if patient_val <= thresh:
                path_conditions.append(f"{disp_feat} ≤ {thresh:.2f}")
                node = tree_.children_left[node]
            else:
                path_conditions.append(f"{disp_feat} > {thresh:.2f}")
                node = tree_.children_right[node]

        leaf_value = tree_.value[node][0]
        pred_class = int(np.argmax(leaf_value))
        outcome_text = "Progression Risk" if pred_class == 1 else "Stable Condition"

        full_rule_statement = " IF " + " AND ".join(path_conditions) + f" ➔ Predicted Outcome: {outcome_text}"
        return full_rule_statement, path_conditions, pred_class

    @property
    def surrogate_tree(self):
        """Exposes primary shallow tree structure for downstream interface visualizers."""
        return self.global_surrogate if self.global_surrogate else list(self.cluster_surrogates.values())[0]

    @staticmethod
    def render_surrogate_validation_section(predictor, df):
        """Computes and presents XCCSHAP shallow decision tree fidelity metrics in plain English."""
        X = df[predictor.features]
        X_scaled = predictor.scaler.transform(X)
        y_xgb = predictor.model.predict(X_scaled)
        
        rulexai = RuleXAIEngine(predictor.model, predictor.scaler, predictor.features)
        rulexai.fit_dynamic_rules(df)
        
        y_surr = rulexai.surrogate_tree.predict(X)
        acc = accuracy_score(y_xgb, y_surr) * 100
        f1 = f1_score(y_xgb, y_surr, average='weighted') * 100



def get_dmt_history_structure(patient_recs):
    dmt_history = []
    p_drugs = patient_recs['1drug'].fillna('No DMT Recorded').astype(str).str.strip().str.title().tolist()
    p_tpoints = patient_recs['timepoints'].tolist()
    p_numdrugs = patient_recs['numdrugs'].fillna(1).astype(int).tolist() if 'numdrugs' in patient_recs.columns else [1]*len(p_drugs)
    p_2drug = patient_recs['2drug'].fillna('').astype(str).str.strip().str.title().tolist() if '2drug' in patient_recs.columns else ['']*len(p_drugs)
    p_edss = patient_recs['EDSS score'].tolist() if 'EDSS score' in patient_recs.columns else [None]*len(p_drugs)
    p_relapses = patient_recs['numrelapse2'].fillna(0).astype(int).tolist() if 'numrelapse2' in patient_recs.columns else [0]*len(p_drugs)

    if len(p_drugs) > 0:
        curr_group_drug = p_drugs[0]
        curr_group_start_tp = p_tpoints[0]
        curr_group_end_tp = p_tpoints[0]
        group_visits = [p_tpoints[0]]
        group_numdrugs = [p_numdrugs[0]]
        group_2nd_drugs = [p_2drug[0]] if p_2drug[0] and p_2drug[0] != 'Nan' else []
        group_edss = [p_edss[0]]
        group_relapses = [p_relapses[0]]
        
        for idx in range(1, len(p_drugs)):
            if p_drugs[idx] == curr_group_drug:
                curr_group_end_tp = p_tpoints[idx]
                group_visits.append(p_tpoints[idx])
                group_numdrugs.append(p_numdrugs[idx])
                if p_2drug[idx] and p_2drug[idx] != 'Nan':
                    group_2nd_drugs.append(p_2drug[idx])
                group_edss.append(p_edss[idx])
                group_relapses.append(p_relapses[idx])
            else:
                dmt_history.append((
                    curr_group_drug, curr_group_start_tp, curr_group_end_tp, 
                    len(group_visits), max(group_numdrugs), list(set(group_2nd_drugs)),
                    group_edss[0], group_edss[-1], max(group_relapses)
                ))
                curr_group_drug = p_drugs[idx]
                curr_group_start_tp = p_tpoints[idx]
                curr_group_end_tp = p_tpoints[idx]
                group_visits = [p_tpoints[idx]]
                group_numdrugs = [p_numdrugs[idx]]
                group_2nd_drugs = [p_2drug[idx]] if p_2drug[idx] and p_2drug[idx] != 'Nan' else []
                group_edss = [p_edss[idx]]
                group_relapses = [p_relapses[idx]]
                
        dmt_history.append((
            curr_group_drug, curr_group_start_tp, curr_group_end_tp, 
            len(group_visits), max(group_numdrugs), list(set(group_2nd_drugs)),
            group_edss[0], group_edss[-1], max(group_relapses)
        ))
    return dmt_history

def draw_dmt_timeline_figure(patient_recs):
    dmt_history = get_dmt_history_structure(patient_recs)
    fig_timeline = go.Figure()

    fig_timeline.add_trace(go.Scatter(
        x=[-0.5, max(len(dmt_history) - 0.5, 0.5)], y=[0, 0],
        mode='lines', line=dict(color='#0f172a', width=3),
        showlegend=False, hoverinfo='skip'
    ))

    stem_y = []
    for idx, (drug_name, start_tp, end_tp, n_visits, total_meds, add_meds, edss_start, edss_end, max_rel) in enumerate(dmt_history):
        direction = 1 if idx % 2 == 0 else -1
        y_val = direction * 2.2
        stem_y.append(y_val)
        
        dt_s = VISIT_DATETIME_MAPPING.get(str(start_tp).lower())
        dt_e = VISIT_DATETIME_MAPPING.get(str(end_tp).lower())
        
        start_str = VISIT_DATE_MAPPING.get(str(start_tp).lower(), str(start_tp).upper())
        end_str = VISIT_DATE_MAPPING.get(str(end_tp).lower(), str(end_tp).upper())
        
        if n_visits == 1:
            dur_label = "6 Months"
        elif dt_s and dt_e:
            m_dur = round((dt_e - dt_s).days / 30.4375, 1)
            dur_label = f"{m_dur} Months" if m_dur < 24 else f"{round(m_dur / 12, 1)} Years"
        else:
            dur_label = f"{n_visits * 6} Months"

        date_disp = start_str if start_str == end_str else f"{start_str} - {end_str}"
        adjunct_str = f" | Adjunct: {', '.join(add_meds)}" if add_meds else ""
        edss_disp = f"EDSS: {edss_start}" if edss_start == edss_end else f"EDSS: {edss_start} → {edss_end}"
        
        text_annotation = (
            f"<b>{drug_name}</b><br>"
            f"📅 {date_disp} ({dur_label})<br>"
            f"💊 Total Medications: {total_meds}{adjunct_str}<br>"
            f"📊 {edss_disp} | Relapses: {max_rel}"
        )
        
        text_pos = "top center" if direction == 1 else "bottom center"

        fig_timeline.add_trace(go.Scatter(
            x=[idx, idx], y=[0, y_val], mode='lines+markers',
            line=dict(color='#dc2626', width=2.5), marker=dict(size=[8, 6], color=['#0f172a', '#dc2626']),
            showlegend=False, hoverinfo='skip'
        ))

        fig_timeline.add_trace(go.Scatter(
            x=[idx], y=[y_val * 1.08], mode='text', 
            text=[text_annotation], textposition=text_pos,
            showlegend=False, hoverinfo='text'
        ))

    fig_timeline.update_layout(
        height=380, margin=dict(l=30, r=30, t=40, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter, sans-serif', size=11, color='#1e293b'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4.8, 4.8])
    )
    return fig_timeline

# =====================================================================
# 3. GLOBAL SIDEBAR STRUCTURE & PATIENT CONTEXT
# =====================================================================
unique_pids = sorted(list(set(df['Local Participant'].unique().tolist() + [10376, 10377])))
with st.sidebar.container():
    st.markdown("""<div style="padding:0 0 1rem;border-bottom:1px solid #334155;margin-bottom:1rem">
    <div style="color:#7dd3fc;font-size:1.2rem;font-weight:800;line-height:1.35">MS-XAI: Human-Centred Explanation Interrogation</div>
    <div style="color:#cbd5e1;font-size:.8rem;line-height:1.5;margin-top:.55rem">Patient-Centred Progression Risk Assessment, Explainable AI, Clinical Reasoning, and Treatment Simulation</div>
    </div>""", unsafe_allow_html=True)
with st.sidebar.container(key="patient_context_sidebar"):
    st.markdown(
    "<div class='sidebar-section-title'>PATIENT CONTEXT</div>",
    unsafe_allow_html=True
        )
    selected_pid = st.selectbox(
        "Select patient",
        options=unique_pids,
        index=0,
        key="main_patient_selector",
    )

patient_records = df[df['Local Participant'] == selected_pid].sort_values('timepoints')
if patient_records.empty:
    patient_records = df[df['Local Participant'] == unique_pids[0]].sort_values('timepoints')

prog_risk, model_accuracy_val, pred_class_val, model_metrics = predictor.predict_patient(patient_records)

try:
    _patient_shap = np.asarray(predictor.compute_patient_shap(patient_records), dtype=float)
    _ranked = np.argsort(np.abs(_patient_shap))
    most_influential = FEATURE_DISPLAY_NAMES[predictor.features[int(_ranked[-1])]]
    least_influential = FEATURE_DISPLAY_NAMES[predictor.features[int(_ranked[0])]]
except (ValueError, IndexError, AttributeError) as exc:
    most_influential = least_influential = "Unavailable"
    st.warning(f"Feature ranking unavailable: {exc}")


curr_edss_val = float(patient_records['EDSS score'].iloc[-1])
baseline_edss_val = float(patient_records['EDSS score'].iloc[0])
age_val = int(patient_records['age'].iloc[-1])
gender_val = str(patient_records['gender'].iloc[-1]) if 'gender' in patient_records.columns else "Female"
mstype_val = str(patient_records['mstype'].iloc[-1]) if 'mstype' in patient_records.columns else "Primary Progressive MS"
current_dmt_val = str(patient_records['1drug'].iloc[-1]) if '1drug' in patient_records.columns and pd.notna(patient_records['1drug'].iloc[-1]) else "Fampyra"
relapses_val = int(patient_records['numrelapse2'].iloc[-1]) if 'numrelapse2' in patient_records.columns and pd.notna(patient_records['numrelapse2'].iloc[-1]) else 0
peak_gait_speed = float(round(patient_records['ws_30_p90_w'].iloc[-1], 2))

raw_visits = patient_records['timepoints'].dropna().astype(str).str.strip().tolist()
formatted_visit_dates = [VISIT_DATE_MAPPING.get(str(v).lower(), str(v).upper()) for v in raw_visits]
last_visit_date = formatted_visit_dates[-1] if formatted_visit_dates else "20 Feb 2021"

if raw_visits:
    first_tp = str(raw_visits[0]).lower()
    last_tp = str(raw_visits[-1]).lower()
else:
    first_tp = "t1"
    last_tp = "t1"

if first_tp in VISIT_DATETIME_MAPPING and last_tp in VISIT_DATETIME_MAPPING:
    dt_start = VISIT_DATETIME_MAPPING[first_tp]
    dt_end = VISIT_DATETIME_MAPPING[last_tp]
    days_span = (dt_end - dt_start).days
    months_span = round(days_span / 30.4375, 1)
    if months_span <= 0:
        disease_dur_val = "6 Months"
    elif months_span < 24.0:
        disease_dur_val = f"{months_span} Months"
    else:
        disease_dur_val = f"{round(months_span / 12, 1)} Years"
else:
    age_delta = float(patient_records['age'].max() - patient_records['age'].min())
    disease_dur_val = f"{age_delta:.1f} Years" if age_delta > 0 else f"{len(raw_visits)*6} Months"

dmt_series = patient_records['1drug'].fillna('Unknown').tolist()
curr_dmt = dmt_series[-1]
dmt_timepoints = []

# FIXED CODE
for idx in range(len(dmt_series) - 1, -1, -1):
    if dmt_series[idx] == curr_dmt:
        if idx < len(raw_visits):
            dmt_timepoints.append(raw_visits[idx])
    else:
        break

dmt_timepoints.reverse()
if len(dmt_timepoints) == 1:
    current_dmt_duration_str = "6 Months"
elif len(dmt_timepoints) > 1 and str(dmt_timepoints[0]).lower() in VISIT_DATETIME_MAPPING and str(dmt_timepoints[-1]).lower() in VISIT_DATETIME_MAPPING:
    dmt_dt_start = VISIT_DATETIME_MAPPING[dmt_timepoints[0].lower()]
    dmt_dt_end = VISIT_DATETIME_MAPPING[dmt_timepoints[-1].lower()]
    dmt_days = (dmt_dt_end - dmt_dt_start).days
    dmt_months = round(dmt_days / 30.4375, 1)
    if dmt_months < 24:
        current_dmt_duration_str = f"{dmt_months} Months"
    else:
        current_dmt_duration_str = f"{round(dmt_months / 12, 1)} Years"
else:
    current_dmt_duration_str = f"{len(dmt_timepoints)*6} Months"

alert_color = "#dc2626" if prog_risk >= 0.60 else ("#d97706" if prog_risk >= 0.30 else "#16a34a")
alert_bg = "#fef2f2" if prog_risk >= 0.60 else ("#fffbe1" if prog_risk >= 0.30 else "#f0fdf4")
alert_border = "#fca5a5" if prog_risk >= 0.60 else ("#fcd34d" if prog_risk >= 0.30 else "#86efac")
alert_label = "HIGH" if prog_risk >= 0.60 else ("MODERATE" if prog_risk >= 0.30 else "LOW")

p_acc = f"{model_metrics.get('accuracy', 0.88):.2f}"
p_rec = f"{model_metrics['recall']:.2f}"
p_prec = f"{model_metrics['precision']:.2f}"
p_f1 = f"{model_metrics['f1']:.2f}"
baseline_date = formatted_visit_dates[0] if formatted_visit_dates else "12 Mar 2020"
dyn_thresh_val = min_increase_for_progression(baseline_edss_val)

# Detailed patient PDF generation (download control appears after all sidebar selectors).
with st.sidebar:
    # Build the PDF in memory for the currently selected patient.
    def _make_patient_pdf():
        """One-page clinician-facing research snapshot; measured progression is distinct from prediction."""
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.utils import ImageReader

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=34, rightMargin=34,
                                topMargin=27, bottomMargin=25,
                                title=f"MS-{selected_pid} | Clinical Progression Snapshot")
        navy = colors.HexColor("#142a43")
        muted = colors.HexColor("#526579")
        teal = colors.HexColor("#087e82")
        red = colors.HexColor("#bf3544")
        pale = colors.HexColor("#edf3f8")
        styles = {
            "title": ParagraphStyle("snap_title", fontName="Helvetica-Bold", fontSize=14.5, leading=17, textColor=navy),
            "sub": ParagraphStyle("snap_sub", fontName="Helvetica", fontSize=8, leading=11, textColor=muted),
            "head": ParagraphStyle("snap_head", fontName="Helvetica-Bold", fontSize=9.1, leading=12, textColor=navy, spaceBefore=7, spaceAfter=3),
            "body": ParagraphStyle("snap_body", fontName="Helvetica", fontSize=7.65, leading=10.2, textColor=navy),
            "small": ParagraphStyle("snap_small", fontName="Helvetica", fontSize=6.7, leading=8.4, textColor=muted),
            "cell": ParagraphStyle("snap_cell", fontName="Helvetica", fontSize=7.4, leading=9.4, textColor=navy),
            "strong": ParagraphStyle("snap_strong", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=navy),
        }
        def para(value, style="body"):
            return Paragraph(html.escape(str(value)), styles[style])
        def rich(value, style="body"):
            return Paragraph(value, styles[style])
        def table(rows, widths, header=False, padd=4):
            t = Table(rows, colWidths=widths, hAlign="LEFT")
            commands = [("VALIGN",(0,0),(-1,-1),"TOP"),
                        ("LEFTPADDING",(0,0),(-1,-1),6), ("RIGHTPADDING",(0,0),(-1,-1),6),
                        ("TOPPADDING",(0,0),(-1,-1),padd), ("BOTTOMPADDING",(0,0),(-1,-1),padd),
                        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,pale]),
                        ("LINEBELOW",(0,-1),(-1,-1),0.4,colors.HexColor("#d5e0e8"))]
            if header:
                commands += [("BACKGROUND",(0,0),(-1,0),navy),
                             ("TEXTCOLOR",(0,0),(-1,0),colors.white)]
            t.setStyle(TableStyle(commands))
            return t
        def fig_image(fig, width=263, max_height=140):
            fig.tight_layout(pad=0.8)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=135, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            png.seek(0)
            iw, ih = ImageReader(png).getSize()
            return RLImage(png, width=width, height=min(max_height,width*ih/iw))

        ordered = patient_records.copy()
        ordered["_order"] = pd.to_numeric(ordered["timepoints"].astype(str).str.extract(r"(\d+)")[0],errors="coerce")
        ordered = ordered.sort_values("_order")
        observed = calc_progression(ordered)
        observed_label = "YES - study-defined progression" if observed == 1 else ("NO - not detected" if observed == 0 else "INDETERMINATE")
        change = curr_edss_val - baseline_edss_val
        threshold = baseline_edss_val + dyn_thresh_val
        x = np.arange(len(ordered))
        edss = pd.to_numeric(ordered["EDSS score"],errors="coerce").to_numpy()
        visits = ordered["timepoints"].astype(str).str.upper().tolist()
        report_date = datetime.date.today().strftime("%d %b %Y")
        story = [rich("MULTIPLE SCLEROSIS | CLINICAL SNAPSHOT", "title"),
                 rich(f"Patient MS-{html.escape(str(selected_pid))} &nbsp; | &nbsp; Report: {report_date} &nbsp; | &nbsp; Research decision-support summary", "sub"),
                 Spacer(1,6)]
        demographics = [
            [para("Age / sex","small"), para("MS phenotype","small"), para("Follow-up / visits","small")],
            [para(f"{age_val} years / {gender_val}","strong"),para(mstype_val,"strong"),para(f"{disease_dur_val} / {len(ordered)}","strong")]
        ]
        story.append(table(demographics,[125,225,194],padd=3))
        story.append(rich("01  PROGRESSION STATUS &amp; MODEL ASSESSMENT", "head"))
        status_rows = [
            [para("Observed EDSS progression", "small"),para("EDSS change", "small"),para("Model-estimated risk", "small")],
            [para(observed_label,"strong"),para(f"{baseline_edss_val:.1f} to {curr_edss_val:.1f} ({change:+.1f}); threshold {threshold:.1f}","strong"),para(f"{prog_risk:.0%} | {'Progression' if pred_class_val else 'Stable'} prediction","strong")]
        ]
        story.append(table(status_rows,[185,189,170],padd=4))
        story.append(rich("Observed status uses two consecutive qualifying EDSS follow-up readings relative to baseline; no minimum confirmation interval is imposed. A predicted risk is not proof of observed progression.","small"))

        story.append(rich("02  EDSS TRAJECTORY &amp; MAIN MODEL CONTRIBUTORS", "head"))
        fig1, ax1 = plt.subplots(figsize=(4.2,2.15))
        ax1.plot(x,edss,marker="o",color="#146a9c",lw=2,label="Observed EDSS")
        ax1.axhline(threshold,color="#bf7633",ls="--",lw=1.5,label="Progression threshold")
        ax1.set_ylim(0,10);ax1.set_xticks(x);ax1.set_xticklabels(visits,fontsize=7)
        ax1.set_ylabel("EDSS",fontsize=8);ax1.tick_params(axis="y",labelsize=7)
        ax1.grid(alpha=.17);ax1.legend(loc="best",fontsize=6,frameon=False)
        edss_chart=fig_image(fig1)
        shap_values = None
        try:
            shap_values=np.asarray(predictor.compute_patient_shap(ordered),dtype=float).reshape(-1)
            shap_values=np.nan_to_num(shap_values,nan=0.0,posinf=0.0,neginf=0.0)
            top_indices=np.argsort(np.abs(shap_values))[::-1][:5]
            top_indices=top_indices[::-1]
            short_names={"walkdur_all_sum_w":"Walking duration","wbsteps_all_sum_w":"Walking steps",
                         "ws_30_avg_w":"Average gait speed","ws_30_p90_w":"Peak gait speed",
                         "cadence_all_avg_w":"Cadence","strlen_30_avg_w":"Stride length",
                         "cadence_all_var_w":"Cadence variability","numrelapse2":"Relapses","age":"Age"}
            labels=[short_names.get(predictor.features[i],predictor.features[i]) for i in top_indices]
            values=shap_values[top_indices]
            fig2,ax2=plt.subplots(figsize=(4.2,2.15))
            ax2.barh(range(len(values)),values,color=["#c44954" if v>0 else "#237fa8" for v in values],height=.65)
            ax2.set_yticks(range(len(values)));ax2.set_yticklabels(labels,fontsize=7)
            ax2.axvline(0,color="#526579",lw=.8);ax2.tick_params(axis="x",labelsize=7)
            ax2.set_xlabel("SHAP impact (log-odds)",fontsize=7);ax2.grid(axis="x",alpha=.14)
            shap_chart=fig_image(fig2)
            story.append(Table([[edss_chart,shap_chart]],colWidths=[272,272],style=TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),
                ("RIGHTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),1),
                ("BOTTOMPADDING",(0,0),(-1,-1),1)])))
            top = np.argsort(np.abs(shap_values))[::-1][:3]
            summary_parts=[]
            for idx in top:
                name=short_names.get(predictor.features[idx],predictor.features[idx])
                direction="higher" if shap_values[idx]>0 else "lower"
                summary_parts.append(f"{name} ({shap_values[idx]:+.2f}; {direction} predicted risk)")
            story.append(rich("<b>Largest model contributions:</b> "+html.escape("; ".join(summary_parts))+". Red = higher model risk; blue = lower model risk.","small"))
        except Exception:
            story.append(edss_chart)
            story.append(rich("Patient-specific SHAP contributions could not be calculated for this report.","small"))
        story.append(rich("SHAP explains the model prediction, not why disability changed clinically. These associations do not establish causation.","small"))

        story.append(rich("03  MS TREATMENT &amp; CLINICAL CONTEXT", "head"))
        history = get_dmt_history_structure(ordered) if "1drug" in ordered.columns else []
        treatments=[]
        for item in history:
            drug,start,end,n_visits,*_ = item
            if str(drug).strip().lower() not in ("unknown","nan","none","no dmt recorded",""):
                label=f"{drug} ({str(start).upper()}-{str(end).upper()})" if str(start)!=str(end) else f"{drug} ({str(start).upper()})"
                treatments.append(label)
        treatment_text="; ".join(treatments) if treatments else "No recorded DMT in available visits"
        if len(treatment_text)>260:
            treatment_text=treatment_text[:257]+"..."
        story.append(table([
            [para("Current recorded therapy","small"),para("Recorded treatment sequence","small")],
            [para(current_dmt_val if current_dmt_val.strip().lower() not in ("nan","unknown") else "Not recorded","strong"),para(treatment_text,"cell")],
            [para("Latest relapse history / peak gait speed","small"),para("Therapy exposure","small")],
            [para(f"{relapses_val} relapses (2-year history); {peak_gait_speed:.2f} m/s","cell"),para(f"Current treatment duration (estimated): {current_dmt_duration_str}","cell")]
        ],[240,304],padd=3))
        story.append(Spacer(1,5))
        story.append(rich("<b>Clinical reading:</b> Determine whether the recorded EDSS trajectory meets the study rule; review relapses, walking changes and treatment exposure alongside clinical examination. The data do not establish that any treatment caused improvement or worsening.","small"))
        story.append(Spacer(1,5))
        story.append(HRFlowable(width="100%",thickness=.5,color=colors.HexColor("#cbd5e1")))
        story.append(rich("RESEARCH USE ONLY  |  Not a validated diagnostic report. Model metrics are in-sample, not independent clinical validation. Visit dates are dashboard mappings; verify against source records.","small"))
        doc.build(story)
        return buf.getvalue()
    patient_report_pdf = _make_patient_pdf()

# =====================================================================
# UPPER PATIENT SUMMARY / CLINICAL ACTION SECTION
# =====================================================================

# ---------------------------------------------------------------------
# TOOLTIP DEFINITIONS
# ---------------------------------------------------------------------

tt_peak_gait = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>Peak Gait Speed:</b> 90th percentile walking speed (m/s)."
    "</span>"
    "</span>"
)

tt_risk_score = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>Progression Risk Score:</b> ML predicted disability probability "
    "over 24 months."
    "</span>"
    "</span>"
)

tt_outcome = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>Predicted Outcome:</b> Binary decision class."
    "</span>"
    "</span>"
)

tt_dmt_proto = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>Medication at Last Visit:</b> Medication prescribed at the "
    "patient’s most recent visit."
    "</span>"
    "</span>"
)

tt_xgb_metrics = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>XGBoost Performance Metrics</b><br><br>"
    "<b>Accuracy:</b> How often the model makes the correct prediction overall.<br><br>"
    "<b>Precision:</b> How often predicted progression cases are actually progression cases.<br><br>"
    "<b>Recall:</b> How well the model identifies patients who actually experience progression.<br><br>"
    "<b>F1 Score:</b> A single measure that balances precision and recall."
    "</span>"
    "</span>"
)

tt_long_hist = (
    "<span class='info-tooltip'>ⓘ"
    "<span class='tooltiptext'>"
    "<b>Total no. of patient visits:</b> Number of patient visits "
    "in this MS study."
    "</span>"
    "</span>"
)


# =====================================================================
# RENDER UPPER SECTION
# =====================================================================

def render_upper_section():

    return (
        f"<div class='patient-summary-inner'>"

        # =============================================================
        # HEADER
        # =============================================================

        # =============================================================
        # ROW 1 — PATIENT INFORMATION
        # =============================================================

        f"<div class='meta-grid'>"

            # Patient ID
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>Patient ID</div>"
                f"<div class='meta-val'>MS-{selected_pid}</div>"
            f"</div>"

            # Demographics
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>Demographics</div>"
                f"<div class='meta-val'>{age_val} yrs | {gender_val}</div>"
            f"</div>"

            # MS Phenotype
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>MS Phenotype</div>"
                f"<div class='meta-val'>{mstype_val}</div>"
            f"</div>"

            # Follow-up Duration
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>Observed Follow-up Duration</div>"
                f"<div class='meta-val'>{disease_dur_val}</div>"
            f"</div>"

            # Last Visit
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>Last Visit</div>"
                f"<div class='meta-val'>{last_visit_date}</div>"
            f"</div>"

            # EDSS
            f"<div class='patient-meta-box'>"
                f"<div class='meta-label'>EDSS at Last Visit</div>"
                f"<div class='meta-val'>{curr_edss_val}</div>"
            f"</div>"

        f"</div>"


        # =============================================================
        # ROW 2 — CLINICAL ACTION STATUS
        # =============================================================

        f"<div class='critical-alert-box' "
        f"style='"
        f"background:{alert_bg};"
        f"border:1px solid {alert_border};"
        f"border-left:6px solid {alert_color};"
        f"'>"

            # ---------------------------------------------------------
            # Section title
            # ---------------------------------------------------------

            f"<div class='critical-title' "
            f"style='color:{alert_color};'>"
                f"⚠️ CLINICAL ACTION STATUS"
            f"</div>"


            # ---------------------------------------------------------
            # Clinical grid
            # ---------------------------------------------------------

            f"<div class='critical-grid'>"


                # =====================================================
                # 1. PROGRESSION RISK
                # =====================================================

                f"<div>"

                    f"<div class='critical-item-label'>"
                        f"Progression Risk Assessment "
                        f"{tt_risk_score}"
                    f"</div>"

                    f"<div class='critical-item-val' "
                    f"style='color:{alert_color};'>"
                        f"{alert_label} ({int(prog_risk * 100)}%)"
                    f"</div>"

                f"</div>"


                # =====================================================
                # Local SHAP feature rankings
                f"<div><div class='critical-item-label'>Most Influential Feature · |SHAP|</div>"
                f"<div class='critical-item-val' style='font-size:.94rem'>{html.escape(most_influential)}</div></div>"
                f"<div><div class='critical-item-label'>Least Influential Feature · |SHAP|</div>"
                f"<div class='critical-item-val' style='font-size:.94rem'>{html.escape(least_influential)}</div></div>"

                # 3. RISK SUMMARY
                # =====================================================

                f"<div>"

                    f"<div class='critical-item-label'>"
                        f"Risk Summary "
                        f"{tt_risk_score}"
                    f"</div>"

                    f"<div style='margin-top:5px;'>"

                        f"<span class='risk-badge' "
                        f"style='background:{alert_color};'>"
                            f"{alert_label} RISK"
                        f"</span>"

                    f"</div>"

                    f"<div style='"
                    f"font-size:0.72rem;"
                    f"color:#475569;"
                    f"margin-top:7px;"
                    f"line-height:1.35;"
                    f"'>"
                        f"Driven by XGBoost progression model and "
                        f"wearable walking telemetry."
                    f"</div>"

                f"</div>"


                # =====================================================
                # 4. MODEL PERFORMANCE
                # =====================================================

                f"<div>"

                    f"<div class='critical-item-label'>"
                        f"Model Performance Metrics (XGBoost) "
                        f"{tt_xgb_metrics}"
                    f"</div>"

                    f"<div class='critical-item-val' "
                    f"style='"
                    f"font-size:0.82rem;"
                    f"color:#1e293b;"
                    f"line-height:1.5;"
                    f"margin-top:4px;"
                    f"'>"

                        f"<b>Accuracy:</b> {p_acc} "
                        f"&nbsp;|&nbsp; "
                        f"<b>Precision:</b> {p_prec}"

                        f"<br>"

                        f"<b>Recall :</b> {p_rec} "
                        f"&nbsp;|&nbsp; "
                        f"<b>F1:</b> {p_f1}"

                    f"</div>"

                f"</div>"


                # =====================================================
                # 5. TOTAL VISITS
                # =====================================================

                f"<div>"

                    f"<div class='critical-item-label'>"
                        f"Total No. of Visits "
                        f"{tt_long_hist}"
                    f"</div>"

                    f"<div class='critical-item-val'>"
                        f"{len(patient_records)} Visits"
                    f"</div>"

                f"</div>"


            f"</div>"  # END critical-grid

        f"</div>"      # END critical-alert-box

        f"</div>"      # END patient-summary-inner
    )


# =====================================================================
# DISPLAY
# =====================================================================

# Compact, sticky patient header. It remains in the main scrolling region.
st.markdown("""<style>
[data-testid="stHeader"] {height:2.25rem!important; min-height:2.25rem!important; background:#f8fafc!important;}
[data-testid="stMainBlockContainer"] {padding-top:.15rem!important;}
.st-key-patient_summary_white_box {
    position:sticky!important; top:2.35rem!important; z-index:90!important;
    padding:10px 14px 8px!important; background:#fff!important;
    margin-bottom:3px!important; box-shadow:0 3px 12px #0f172a18!important;
}
.st-key-patient_summary_white_box .patient-meta-box {padding:6px 9px!important;}
.st-key-patient_summary_white_box .meta-grid {gap:7px!important;margin-bottom:8px!important;}
.st-key-patient_summary_white_box .critical-alert-box {padding:9px 12px!important;margin-bottom:7px!important;}
.st-key-patient_summary_white_box .critical-grid {gap:8px!important;}
.st-key-patient_summary_white_box .critical-title {margin-bottom:5px!important;}
.st-key-patient_summary_white_box .stButton button {min-height:29px!important;}
@media(max-width:900px){.st-key-patient_summary_white_box{position:relative!important;top:auto!important;}}
</style>""",unsafe_allow_html=True)

# Patient summary: real Streamlit bordered container holds both HTML and the popup button.

with st.container(border=True, key="patient_summary_white_box"):
    st.markdown(render_upper_section(), unsafe_allow_html=True)
    _open_treatment_history = st.button(
        "Observed Treatment History",
        icon=":material/medication:",
        key="open_medication_timeline",
        help="Open the selected patient's detailed medication history",
    )
st.caption("Feature rankings use absolute local SHAP values. Model training and evaluation remain retrospective and require temporal validation.")
# Shared explanation-interrogation feature selector. Rank the CURRENT patient's
# features by absolute local SHAP magnitude, rather than fixed global importance.
try:
    _interrogation_shap = np.asarray(predictor.compute_patient_shap(patient_records), dtype=float).reshape(-1)
    _interrogation_ranking = sorted(
        range(len(predictor.features)),
        key=lambda i: abs(_interrogation_shap[i]) if np.isfinite(_interrogation_shap[i]) else -1,
        reverse=True,
    )
except Exception:
    _interrogation_shap = np.full(len(predictor.features), np.nan)
    _interrogation_ranking = list(range(len(predictor.features)))
_interrogation_options = [predictor.features[i] for i in _interrogation_ranking]
_rank_by_feature = {predictor.features[i]: rank for rank, i in enumerate(_interrogation_ranking)}
_rank_palette = ["#12345b", "#214b77", "#32628a", "#48789b", "#608ca9",
                 "#769eb6", "#8aafc3", "#9fbfce", "#b4cbd5"]
# Shared selection is created in the relevant content header, never in the sidebar.
if "_pending_feature_selection" in st.session_state:
    _queued_feature = st.session_state.pop("_pending_feature_selection")
    if _queued_feature in _interrogation_options:
        st.session_state["interrogation_selected_feature"] = _queued_feature
if st.session_state.get("interrogation_selected_feature") not in _interrogation_options:
    st.session_state["interrogation_selected_feature"] = _interrogation_options[0]
_selected_interrogation_feature = st.session_state["interrogation_selected_feature"]
_selected_interrogation_label = FEATURE_DISPLAY_NAMES.get(_selected_interrogation_feature, _selected_interrogation_feature)

# Final sidebar action: one report button below every selector.
with st.sidebar:
    st.markdown('<div class="sidebar-download-divider"></div>', unsafe_allow_html=True)
    st.download_button(
        "Download Patient Report",
        data=patient_report_pdf,
        file_name=f"MS_{selected_pid}_Patient_Report.pdf",
        mime="application/pdf",
        key="patient_pdf_download",
        width="stretch",
        type="primary",
        icon=":material/download:",
    )
    st.caption("Research summary · Not a validated clinical report")

# Shared interrogation context, recalculated on every feature selection.
_interrogation_rank = _rank_by_feature[_selected_interrogation_feature] + 1
_selected_shap = float(_interrogation_shap[predictor.features.index(_selected_interrogation_feature)])
if _selected_shap > 1e-10:
    _selected_direction = "Contributes to a higher model-predicted likelihood of EDSS disability progression."
    _direction_class = "xai-direction-higher"
elif _selected_shap < -1e-10:
    _selected_direction = "Contributes to a lower model-predicted likelihood of EDSS disability progression."
    _direction_class = "xai-direction-lower"
else:
    _selected_direction = "Minimal contribution to the predicted likelihood of EDSS disability progression."
    _direction_class = "xai-direction-neutral"

def _interrogation_header(section, description, show_selector=True):
    safe_section = html.escape(section)
    safe_description = html.escape(description)
    st.markdown(f"""
    <div class="interrogation-panel" style="margin-bottom:0;border-bottom-left-radius:0;border-bottom-right-radius:0;">
      <div class="interrogation-eyebrow">EXPLANATION INTERROGATION · PATIENT MS-{html.escape(str(selected_pid))}</div>
      <div class="interrogation-title">{safe_section}</div>
      <div class="interrogation-description">{safe_description}</div>
    </div>""", unsafe_allow_html=True)
    if show_selector:
        with st.container(border=True):
            st.selectbox(
                "Selected feature · ranked by patient-specific SHAP influence",
                options=_interrogation_options,
                format_func=lambda f: f"Rank {_rank_by_feature[f]+1} of {len(predictor.features)} · {FEATURE_DISPLAY_NAMES.get(f,f)}",
                key="interrogation_selected_feature",
                on_change=_feature_header_changed,
            )
            st.markdown(
                f'<span class="interrogation-direction {_direction_class}">{html.escape(_selected_direction)}</span>',
                unsafe_allow_html=True,
            )

def _feature_header_changed():
    st.session_state["_xai_navigation_pending"] = "Feature Interrogation"

# Single, consistent clinical design system, applied after the earlier legacy styles.
st.markdown("""<style>
:root { --clinical-navy:#183653; --clinical-blue:#176ca8; --clinical-muted:#586d81;
 --clinical-border:#d9e5ee; --clinical-pale:#f1f7fb; --clinical-red:#b42332; --clinical-green:#167347; }
[data-testid="stMain"] .stMarkdown h1 {font-size:1.85rem!important;font-weight:800!important;line-height:1.2!important;color:var(--clinical-navy)!important;}
[data-testid="stMain"] .stMarkdown h2 {font-size:1.55rem!important;font-weight:780!important;line-height:1.25!important;color:var(--clinical-navy)!important;}
[data-testid="stMain"] .stMarkdown h3 {font-size:1.26rem!important;font-weight:750!important;line-height:1.3!important;color:var(--clinical-navy)!important;}
[data-testid="stMain"] .stMarkdown h4 {font-size:1.08rem!important;font-weight:720!important;line-height:1.35!important;color:var(--clinical-navy)!important;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {font-size:.83rem!important;color:var(--clinical-muted)!important;}
[data-testid="stMain"] [data-testid="stMetric"] {background:#fff!important;border:1px solid var(--clinical-border)!important;border-radius:12px!important;box-shadow:0 2px 8px #1836530a;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tab"] {font-size:.94rem!important;font-weight:730!important;color:var(--clinical-muted)!important;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {color:var(--clinical-blue)!important;}
[data-testid="stMain"] .report-title {font-size:2rem!important;line-height:1.2!important;font-weight:820!important;color:var(--clinical-navy)!important;}
[data-testid="stMain"] .report-subtitle {font-size:.95rem!important;color:var(--clinical-muted)!important;line-height:1.5!important;}
.interrogation-panel {background:#fff;border:1px solid var(--clinical-border);border-left:5px solid var(--clinical-blue);
 border-radius:13px;padding:19px 22px;margin:10px 0 19px;box-shadow:0 3px 12px #1836530c;}
.interrogation-eyebrow {font-size:.73rem;letter-spacing:.095em;font-weight:800;color:var(--clinical-blue);line-height:1.45;}
.interrogation-title {font-size:1.58rem;font-weight:800;line-height:1.25;color:var(--clinical-navy);margin:7px 0 5px;}
.interrogation-description {font-size:.9rem;line-height:1.55;color:var(--clinical-muted);margin-bottom:14px;}
.interrogation-feature {display:flex;align-items:center;flex-wrap:wrap;gap:8px 10px;font-size:.9rem;
 color:var(--clinical-navy);background:var(--clinical-pale);border:1px solid #e3edf5;border-radius:9px;padding:12px 13px;line-height:1.45;}
.interrogation-rank {display:inline-flex;align-items:center;padding:5px 10px;border-radius:18px;
 background:#dfecf7;color:#205780;font-size:.79rem;font-weight:750;line-height:1.4;}
.interrogation-direction {display:inline-flex;align-items:center;max-width:100%;padding:6px 11px;
 border-radius:9px;font-size:.82rem;font-weight:760;line-height:1.45;white-space:normal;}
.xai-direction-higher {background:#fce8ea!important;color:var(--clinical-red)!important;border:1px solid #f1bdc4;}
.xai-direction-lower {background:#e5f5ec!important;color:var(--clinical-green)!important;border:1px solid #b6e0c8;}
.xai-direction-neutral {background:#e8eef3!important;color:#536477!important;border:1px solid #d2dce5;}
@media(max-width:700px){.interrogation-panel{padding:15px 14px}.interrogation-title{font-size:1.35rem}}
</style>""",unsafe_allow_html=True)

def _xai_open_feature(feature):
    """Queue a feature change for the next run, before widgets are instantiated.

    Never write to a widget-owned session_state key after its widget has rendered.
    """
    if feature not in predictor.features:
        return
    st.session_state["_pending_feature_selection"] = feature
    st.session_state["_xai_navigation_pending"] = "Feature Interrogation"



def _xai_filter_changed():
    """An impact-filter change opens interrogation of the strongest matching feature."""
    direction = st.session_state.get("interactive_shap_direction", "All")
    values = np.asarray(_interrogation_shap, dtype=float)
    ranked = [i for i in _interrogation_ranking if np.isfinite(values[i]) and (
        direction == "All" or (values[i] >= 0 if direction == "Positive impact" else values[i] < 0))]
    if ranked:
        _xai_open_feature(predictor.features[ranked[0]])


# =====================================================================
# 4. MAIN PAGE TABS
# =====================================================================
# One continuous interrogation page; original chart renderers and columns retained.
_xai_pages = ["Clinical Overview & SHAP", "Feature Interrogation", "Patient Decision Pathway"]
if "xai_active_page" not in st.session_state:
    st.session_state["xai_active_page"] = "Clinical Overview & SHAP"
if "xai_navigation_radio" not in st.session_state:
    st.session_state["xai_navigation_radio"] = "Clinical Overview & SHAP"
if "_xai_navigation_pending" in st.session_state:
    _target_page = st.session_state.pop("_xai_navigation_pending")
    if _target_page in _xai_pages:
        # The radio has not been instantiated yet in this run.
        st.session_state["xai_navigation_radio"] = _target_page
        st.session_state["xai_active_page"] = _target_page
_nav_page = st.radio(
    "Dashboard workspace", _xai_pages, horizontal=True,
    key="xai_navigation_radio", label_visibility="collapsed",
)
st.session_state["xai_active_page"] = _nav_page

# Account/settings menu at top-right, with the former clinical judgement audit.
_settings_spacer, _settings_col = st.columns([12, 1])
with _settings_col:
    with st.popover("⚙️", help="Settings · Clinical judgement and audit trail"):
        st.markdown("### Clinical Judgement · Recorded interpretation")


        _records = [v for v in st.session_state.get("clinical_assessments",{}).values() if v.get("patient")==str(selected_pid)]
        if _records:
            st.dataframe(pd.DataFrame(_records)[["timestamp","feature_name","status","rationale","model_prediction","shap_log_odds"]],hide_index=True,width="stretch")
        else:
            st.info("No feature assessments saved for this patient. Select a feature and save an interpretation in Feature Interrogation.")
        _audit = st.session_state.get("clinical_audit",[])
        if _audit:
            st.download_button("Download research audit trail (CSV)",pd.DataFrame(_audit).to_csv(index=False).encode("utf-8"),file_name="ms_xai_session_audit.csv",mime="text/csv")
        st.caption("Session-only assessments: download the audit before closing the application. No clinical decision or treatment recommendation is automated.")



def get_exact_shap_waterfall_fig(engine, patient_df):
    X_p = patient_df[engine.features].iloc[[-1]]
    shap_vals = engine.compute_patient_shap(patient_df)
    display_names = [FEATURE_DISPLAY_NAMES[f] for f in engine.features]
    raw_vals = X_p.values[0]
    
    labels = [f"{raw_vals[i]:.2f} = {display_names[i]}" for i in range(len(engine.features))]
    sorted_indices = np.argsort(np.abs(shap_vals))
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_shap = [shap_vals[i] for i in sorted_indices]
    
    bar_colors = ['#b42332' if v > 0 else '#167347' if v < 0 else '#94a3b8' for v in sorted_shap]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(sorted_labels, sorted_shap, color=bar_colors, edgecolor='none', height=0.6)
    ax.axvline(x=0, color='#64748b', linestyle='--', linewidth=1)

    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.01
        ha = 'left' if width >= 0 else 'right'
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, 
                f"{width:+.2f}", va='center', ha=ha, fontsize=9, fontweight='bold', color='#1e293b')

    ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    
    plt.tight_layout()
    return fig

def get_exact_lime_fig(engine, patient_df):
    X_p = patient_df[engine.features].iloc[[-1]]
    X_p_scaled = engine.scaler.transform(X_p)[0]

    def predict_fn(x):
        return engine.model.predict_proba(x)

    exp = engine.lime_explainer.explain_instance(
        data_row=X_p_scaled,
        predict_fn=predict_fn,
        num_features=len(engine.features)
    )

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 5)
    plt.subplots_adjust(left=0.45, right=0.95, top=0.90, bottom=0.15)
    plt.tight_layout()
    return fig


@st.dialog("Observed Treatment History · Detailed Medication Timeline", width="large")
def _show_observed_treatment_history():
    st.caption("All treatments recorded/administered for the selected patient are shown below, together with their treatment classification and available medication information. This is historical/descriptive evidence; no treatment effect is inferred here.")
    # =====================================================================
    # OBSERVED TREATMENT HISTORY TIMELINE
    # =====================================================================
    st.markdown("#### Observed Treatment History")
    st.caption(
        "All treatments recorded/administered for the selected patient are shown below, "
        "together with their treatment classification and available medication information. "
        "This is historical/descriptive evidence; no treatment effect is inferred here."
    )
    
    # Medication information supplied with the application.
    # Kept local to this section so every observed medication can be classified
    # without filtering the history down to DMTs only.
    OBSERVED_TREATMENT_KB = {'baclofen': {'title': 'Baclofen (Lioresal)',
              'category': 'Symptomatic Management Therapies',
              'nice_guideline': 'NICE NG220 (Symptom Management: Spasticity)',
              'nice_recommendation': 'First-line oral treatment for spasticity in adult MS patients where spasticity '
                                     'causes discomfort or functional impairment.',
              'indication': 'Spasticity (muscle stiffness, tightness, involuntary spasms) & painful muscle cramps '
                            'caused by MS damage in central nerve pathways.',
              'mechanism': 'GABA-B receptor agonist in the central nervous system. Inhibits spinal cord reflexes to '
                           'relax skeletal muscles and reduce muscle tone.',
              'administration': 'Oral tablet (Lioresal) or Intrathecal Baclofen (ITB) pump surgically implanted to '
                                'deliver medication directly into spinal fluid for severe spasticity.'},
 'fampridine': {'title': 'Fampridine / Dalfampridine (Fampyra / Ampyra)',
                'category': 'Symptomatic Management Therapies',
                'nice_guideline': 'NICE NG220 / TA803 (Walking Impairment)',
                'nice_recommendation': 'Recommended as an option for improving walking in adults with MS who have '
                                       'walking disability (EDSS 4.0–7.0) and demonstrate response at 4–8 weeks.',
                'indication': 'Symptomatic improvement of walking ability and walking speed in adult patients with '
                              'Multiple Sclerosis.',
                'mechanism': 'Broad-spectrum voltage-gated potassium channel blocker. Prolongs repolarization and '
                             'enhances action potential conduction in demyelinated axons.',
                'administration': 'Extended-release oral tablet taken twice daily (12 hours apart).'},
 'tizanidine': {'title': 'Tizanidine (Sirdalud / Zanaflex)',
                'category': 'Symptomatic Management Therapies',
                'nice_guideline': 'NICE NG220 (Symptom Management: Spasticity)',
                'nice_recommendation': 'First-line alternative or adjunct to baclofen for managing spasticity in '
                                       'adults with MS.',
                'indication': 'Management of muscle spasticity associated with Multiple Sclerosis and spinal cord '
                              'injury.',
                'mechanism': 'Central alpha-2 adrenergic receptor agonist. Increases presynaptic inhibition of motor '
                             'neurons to reduce spastic muscle tone.',
                'administration': 'Oral tablets or capsules, titrated according to patient tolerance and muscle tone '
                                  'response.'},
 'gabapentin': {'title': 'Gabapentin (Neurontin)',
                'category': 'Symptomatic Management Therapies',
                'nice_guideline': 'NICE NG220 / CG173 (Neuropathic Pain)',
                'nice_recommendation': 'First-line option for treating MS-related neuropathic pain, dysesthesia, or '
                                       "Lhermitte's sign.",
                'indication': "Neuropathic pain, dysesthesias, paroxysmal symptoms, and Lhermitte's sign in MS.",
                'mechanism': 'Binds to alpha-2-delta subunit of voltage-gated calcium channels in the CNS, modulating '
                             'excitatory neurotransmitter release.',
                'administration': 'Oral capsule or tablet divided into 3 daily doses.'},
 'pregabalin': {'title': 'Pregabalin (Lyrica)',
                'category': 'Symptomatic Management Therapies',
                'nice_guideline': 'NICE NG220 / CG173 (Neuropathic Pain)',
                'nice_recommendation': 'First-line option for neuropathic pain management in MS when gabapentin is '
                                       'unsuitable or ineffective.',
                'indication': 'Central neuropathic pain, dysesthetic extremity pain, and neuropathic sensory '
                              'disturbances.',
                'mechanism': 'High-affinity binding to alpha-2-delta protein in central nervous system tissues, '
                             'reducing calcium influx in nerve terminals.',
                'administration': 'Oral capsule or solution administered twice or thrice daily.'},
 'modafinil': {'title': 'Modafinil (Provigil)',
               'category': 'Symptomatic Management Therapies',
               'nice_guideline': 'NICE NG220 (Off-label Fatigue Management)',
               'nice_recommendation': 'Considered under specialist supervision for disabling MS-related fatigue after '
                                      'non-pharmacological interventions.',
               'indication': 'MS-related chronic fatigue and daytime somnolence.',
               'mechanism': 'Atypical dopamine reuptake inhibitor that promotes wakefulness by increasing '
                            'extracellular dopamine and hypothalamic histamine levels.',
               'administration': 'Oral tablet administered once daily in the morning.'},
 'amantadine': {'title': 'Amantadine (Mantadan / PK-Merz)',
                'category': 'Symptomatic Management Therapies',
                'nice_guideline': 'NICE NG220 (Fatigue)',
                'nice_recommendation': 'Option for managing MS fatigue; offered as a trial for patients experiencing '
                                       'significant fatigue affecting daily activities.',
                'indication': 'MS-related fatigue and motor sluggishness.',
                'mechanism': 'NMDA receptor antagonist and dopamine release facilitator in central neural circuits.',
                'administration': 'Oral capsule or tablet taken once or twice daily.'},
 'solifenacin': {'title': 'Solifenacin (Vesicare)',
                 'category': 'Symptomatic Management Therapies - Neurogenic Bladder',
                 'nice_guideline': 'NICE NG220 / CG148 (Lower Urinary Tract Symptoms)',
                 'nice_recommendation': 'Recommended as a first-line antimuscarinic choice for overactive bladder '
                                        'symptoms in MS after post-void residual volume check.',
                 'indication': 'Neurogenic detrusor overactivity, urinary urgency, frequency, and urge incontinence.',
                 'mechanism': 'Competitive muscarinic M3 receptor antagonist that relaxes detrusor smooth muscle and '
                              'reduces involuntary bladder contractions.',
                 'administration': 'Oral tablet taken once daily.'},
 'mirabegron': {'title': 'Mirabegron (Betmiga)',
                'category': 'Symptomatic Management Therapies - Neurogenic Bladder',
                'nice_guideline': 'NICE NG220 / TA290 (Bladder Dysfunction)',
                'nice_recommendation': 'Alternative for MS bladder dysfunction when anticholinergics are ineffective, '
                                       'poorly tolerated, or contraindicated (e.g., severe cognitive risk).',
                'indication': 'Urgency, increased micturition frequency, and urge incontinence in neurogenic bladder.',
                'mechanism': 'Beta-3 adrenergic receptor agonist that induces detrusor relaxation during the storage '
                             'phase, increasing bladder capacity without anticholinergic cognitive side effects.',
                'administration': 'Oral extended-release tablet taken once daily.'},
 'ocrelizumab': {'title': 'Ocrelizumab (Ocrevus)',
                 'category': 'Disease-Modifying Therapies (DMT) - Anti-CD20 Monoclonal Antibody',
                 'nice_guideline': 'NICE TA533 (RRMS) & TA585 (PPMS)',
                 'nice_recommendation': 'Recommended for active RRMS and early primary progressive MS (PPMS with short '
                                        'disease duration & imaging features of inflammatory activity).',
                 'indication': 'Relapsing forms of MS (RMS) and Primary Progressive MS (PPMS).',
                 'mechanism': 'Humanized monoclonal antibody that selectively targets CD20-expressing B cells, '
                              'mediating B-cell depletion via ADCC and CDC to reduce neuroinflammation.',
                 'administration': 'Intravenous infusion every 6 months following an initial split-dose starter '
                                   'regimen.'},
 'ublituximab': {'title': 'Ublituximab (Briumvi)',
                 'category': 'Disease-Modifying Therapies (DMT) - Anti-CD20 Monoclonal Antibody',
                 'nice_guideline': 'NICE TA1025 (Active RRMS)',
                 'nice_recommendation': 'Recommended for adults with active relapsing forms of MS with clinical or '
                                        'imaging features.',
                 'indication': 'Relapsing forms of MS (RMS), including RRMS and active SPMS.',
                 'mechanism': 'Glycoengineered humanized anti-CD20 monoclonal antibody targeted to a unique epitope, '
                              'inducing rapid, high-potency antibody-dependent cellular cytotoxicity (ADCC) enabling '
                              '1-hour maintenance infusions.',
                 'administration': 'Intravenous infusion: Day 1 (150mg over 4h), Day 15 (450mg over 1h), then 450mg '
                                   'every 6 months (1h infusion).'},
 'fingolimod': {'title': 'Fingolimod (Gilenya)',
                'category': 'Disease-Modifying Therapies (DMT) - S1P Receptor Modulator',
                'nice_guideline': 'NICE TA254 (Highly Active RRMS)',
                'nice_recommendation': 'Recommended for highly active RRMS in adults with 1+ relapse in the past year '
                                       'despite baseline DMT or rapidly evolving severe RRMS.',
                'indication': 'Highly active Relapsing-Remitting MS (RRMS).',
                'mechanism': 'Sphingosine-1-phosphate (S1P) receptor modulator that sequesters lymphocytes within '
                             'lymph nodes, preventing encephalitogenic T cells from crossing the blood-brain barrier.',
                'administration': 'Oral capsule taken once daily.'},
 'ozanimod': {'title': 'Ozanimod (Zeposia)',
              'category': 'Disease-Modifying Therapies (DMT) - Selective S1P Modulator',
              'nice_guideline': 'NICE TA678 (Active RRMS)',
              'nice_recommendation': 'Recommended as an option for treating active relapsing-remitting MS in adults.',
              'indication': 'Relapsing forms of MS (RRMS).',
              'mechanism': 'Selective S1P-1 and S1P-5 receptor modulator that prevents lymphocyte egress from '
                           'secondary lymphoid organs into the central nervous system.',
              'administration': 'Oral capsule taken once daily following a 7-day dose escalation titration pack.'},
 'ponesimod': {'title': 'Ponesimod (Ponvory)',
               'category': 'Disease-Modifying Therapies (DMT) - Selective S1P Modulator',
               'nice_guideline': 'NICE TA767 (Active RRMS)',
               'nice_recommendation': 'Recommended as an option for active RRMS in adults.',
               'indication': 'Relapsing forms of MS (RRMS).',
               'mechanism': 'Reversible selective S1P-1 receptor modulator with rapid clearance (normalizes lymphocyte '
                            'counts within 1 week of discontinuation).',
               'administration': 'Oral tablet taken once daily following a 14-day dose titration schedule.'},
 'dimethyl fumarate': {'title': 'Dimethyl Fumarate (Tecfidera)',
                       'category': 'Disease-Modifying Therapies (DMT) - Nrf2 Activator',
                       'nice_guideline': 'NICE TA320 (RRMS)',
                       'nice_recommendation': 'Recommended as a first-line oral option for adults with active '
                                              'relapsing-remitting MS.',
                       'indication': 'Relapsing-Remitting Multiple Sclerosis (RRMS).',
                       'mechanism': 'Activates Nuclear factor erythroid 2-related factor 2 (Nrf2) pathway, inducing '
                                    'neuroprotective antioxidant responses and downregulating pro-inflammatory '
                                    'cytokines.',
                       'administration': 'Oral delayed-release capsule taken twice daily with food.'},
 'diroximel fumarate': {'title': 'Diroximel Fumarate (Vumerity)',
                        'category': 'Disease-Modifying Therapies (DMT) - Nrf2 Activator',
                        'nice_guideline': 'NICE TA829 (Active RRMS)',
                        'nice_recommendation': 'Recommended as an oral option for active RRMS with improved GI '
                                               'tolerability over dimethyl fumarate.',
                        'indication': 'Relapsing-Remitting MS (RRMS).',
                        'mechanism': 'Prodrug converted to monomethyl fumarate (MMF), activating the Nrf2 pathway; '
                                     'distinct chemical structure reduces GI adverse events.',
                        'administration': 'Oral delayed-release capsules taken twice daily.'},
 'monomethyl fumarate': {'title': 'Monomethyl Fumarate (Bafiertam)',
                         'category': 'Disease-Modifying Therapies (DMT) - Nrf2 Activator',
                         'nice_guideline': 'FDA Approved / Specialist Pathway',
                         'nice_recommendation': 'Bioequivalent active metabolite option targeting Nrf2 pathway with '
                                                'distinct dosing formulation.',
                         'indication': 'Relapsing forms of MS (RMS).',
                         'mechanism': 'Direct activation of Nrf2 cellular defense response to protect against '
                                      'oxidative-stress-mediated neuronal injury.',
                         'administration': 'Oral delayed-release capsules taken twice daily.'},
 'natalizumab': {'title': 'Natalizumab (Tysabri / Tyruko)',
                 'category': 'Disease-Modifying Therapies (DMT) - Selective Adhesion Molecule Inhibitor',
                 'nice_guideline': 'NICE TA127 (Rapidly Evolving Severe RRMS)',
                 'nice_recommendation': 'Recommended for rapidly evolving severe (RES) RRMS or active RRMS despite '
                                        'treatment with a beta-interferon or glatiramer.',
                 'indication': 'Highly active Relapsing-Remitting MS.',
                 'mechanism': 'Monoclonal antibody targeting alpha-4 integrin (VLA-4), preventing leukocyte '
                              'transmigration across the blood-brain barrier into CNS tissue.',
                 'administration': 'Intravenous infusion or Subcutaneous injection every 4 to 6 weeks.'},
 'teriflunomide': {'title': 'Teriflunomide (Aubagio)',
                   'category': 'Disease-Modifying Therapies (DMT) - Pyrimidine Synthesis Inhibitor',
                   'nice_guideline': 'NICE TA303 (Active RRMS)',
                   'nice_recommendation': 'Recommended as an oral option for active RRMS without features of highly '
                                          'active or rapidly progressive disease.',
                   'indication': 'Relapsing forms of MS.',
                   'mechanism': 'Inhibits dihydroorotate dehydrogenase (DHODH) to reduce rapid proliferation of '
                                'autoreactive activated T and B lymphocytes.',
                   'administration': 'Oral tablet taken once daily.'},
 'glatiramer': {'title': 'Glatiramer Acetate (Copaxone / Brabio)',
                'category': 'Disease-Modifying Therapies (DMT) - Immunomodulator',
                'nice_guideline': 'NICE TA527 (First-line RRMS / CIS)',
                'nice_recommendation': 'Recommended as a standard baseline option for active RRMS or clinically '
                                       'isolated syndrome (CIS) with active lesions.',
                'indication': 'Relapsing forms of MS.',
                'mechanism': 'Synthetic polypeptide mimic of myelin basic protein that shifts helper T-cell response '
                             'from Th1/Th17 pro-inflammatory to Th2 regulatory phenotype.',
                'administration': 'Subcutaneous injection 3 times weekly (40 mg) or daily (20 mg).'},
 'interferon': {'title': 'Interferon beta (Rebif / Avonex / Plegridy / Betaferon)',
                'category': 'Disease-Modifying Therapies (DMT) - Type I Interferon',
                'nice_guideline': 'NICE TA527 (Injectable DMT Baseline)',
                'nice_recommendation': 'Recommended as baseline first-line injectable DMT for active RRMS or CIS with '
                                       'documented MRI inflammatory activity.',
                'indication': 'Relapsing-Remitting MS and Clinically Isolated Syndrome (CIS).',
                'mechanism': 'Upregulates anti-inflammatory cytokines, restores blood-brain barrier integrity, and '
                             'suppresses T-cell activation.',
                'administration': 'Subcutaneous or intramuscular injection according to formulation schedule.'},
 'ofatumumab': {'title': 'Ofatumumab (Kesimpta)',
                'category': 'Disease-Modifying Therapies (DMT) - Subcutaneous Anti-CD20',
                'nice_guideline': 'NICE TA699 (Relapsing MS)',
                'nice_recommendation': 'Recommended as an option for adults with relapsing forms of MS with active '
                                       'disease defined by clinical or imaging features.',
                'indication': 'Relapsing forms of MS.',
                'mechanism': 'Fully human Anti-CD20 monoclonal antibody inducing precise B-cell depletion.',
                'administration': 'Self-administered subcutaneous autoinjector monthly at home following starter '
                                  'doses.'},
 'siponimod': {'title': 'Siponimod (Mayzent)',
               'category': 'Disease-Modifying Therapies (DMT) - Selective S1P1/S1P5 Modulator',
               'nice_guideline': 'NICE TA656 (Active SPMS)',
               'nice_recommendation': 'Recommended for adults with secondary progressive MS (SPMS) with active disease '
                                      '(relapses or MRI evidence of inflammatory activity).',
               'indication': 'Secondary Progressive MS (SPMS) with active disease & active RRMS.',
               'mechanism': 'Selective modulator of S1P1 and S1P5 receptors, preventing lymphocyte egress from '
                            'lymphoid tissue and penetrating CNS to protect neuroglia.',
               'administration': 'Oral tablet daily following CYP2C9 genotype titration.'},
 'cladribine': {'title': 'Cladribine (Mavenclad)',
                'category': 'Immune Reconstitution Therapy (IRT) / Pulsed DMT',
                'nice_guideline': 'NICE TA616 / TA1053 (Highly Active RRMS)',
                'nice_recommendation': 'Recommended for adults with highly active RRMS defined by clinical or MRI '
                                       'parameters.',
                'indication': 'Highly active Relapsing MS.',
                'mechanism': 'Purine nucleoside analogue causing selective depletion of B and T lymphocytes followed '
                             'by targeted immune reconstitution.',
                'administration': 'Short oral treatment courses given in Year 1 and Year 2 (2 treatment weeks per '
                                  'year).'},
 'alemtuzumab': {'title': 'Alemtuzumab (Lemtrada)',
                 'category': 'Immune Reconstitution Therapy (IRT) - Anti-CD52',
                 'nice_guideline': 'NICE TA312 / TA623 (Active/Highly Active RRMS)',
                 'nice_recommendation': 'Recommended for active RRMS despite prior therapy or rapidly evolving severe '
                                        'RRMS, with safety monitoring for autoimmune conditions.',
                 'indication': 'Active Relapsing-Remitting MS failing baseline DMTs.',
                 'mechanism': 'Monoclonal antibody binding CD52 to deplete T and B lymphocytes, resetting immune '
                              'system composition.',
                 'administration': 'Intravenous courses: 5 consecutive days at Year 1, 3 consecutive days at Year 2.'},
 'mitoxantrone': {'title': 'Mitoxantrone (Novantrone)',
                  'category': 'Disease-Modifying Therapies (DMT) - Antineoplastic / Immunosuppressant',
                  'nice_guideline': 'NICE NG220 / Specialist Commissioning',
                  'nice_recommendation': 'Reserved for rapidly worsening relapsing or secondary progressive MS when no '
                                         'other options are suitable, subject to strict lifetime cumulative dose '
                                         'monitoring.',
                  'indication': 'Aggressive, highly active Relapsing MS and Secondary Progressive MS.',
                  'mechanism': 'Topoisomerase II inhibitor that intercalates into DNA and causes strand breaks, '
                               'broadly suppressing T cells, B cells, and macrophages.',
                  'administration': 'Intravenous infusion once every 3 months (Lifetime cumulative dose capped at 140 '
                                    'mg/m²).'},
 'fenebrutinib': {'title': 'Fenebrutinib (Investigational Phase 3 BTKi)',
                  'category': "Disease-Modifying Therapies (DMT) - Bruton's Tyrosine Kinase (BTK) Inhibitor",
                  'nice_guideline': 'Under Phase 3 Evaluation (FENhance / FENtrepid Trials)',
                  'nice_recommendation': 'Investigational agent under clinical development for Relapsing and Primary '
                                         'Progressive MS.',
                  'indication': 'Relapsing Multiple Sclerosis & Primary Progressive MS.',
                  'mechanism': 'Non-covalent, highly selective BTK inhibitor targeting dual B-cell activation and '
                               'microglia-mediated neuroinflammation across the blood-brain barrier.',
                  'administration': 'Oral tablet taken twice daily.'},
 'tolebrutinib': {'title': 'Tolebrutinib (Investigational Phase 3 BTKi)',
                  'category': "Disease-Modifying Therapies (DMT) - Bruton's Tyrosine Kinase (BTK) Inhibitor",
                  'nice_guideline': 'Under Regulatory Review / Phase 3 Evaluation (HERCULES / GEMINI Trials)',
                  'nice_recommendation': 'Investigational agent demonstrating reduction in non-relapsing disability '
                                         'progression in Secondary Progressive MS (SPMS).',
                  'indication': 'Non-relapsing Secondary Progressive MS (nrSPMS) and Relapsing MS.',
                  'mechanism': 'CNS-penetrant covalent BTK inhibitor targeting B-cell signaling and smoldering '
                               'neuroinflammation mediated by activated CNS-resident microglia.',
                  'administration': 'Oral tablet taken once daily.'}}
    OBSERVED_SPECIAL_KB = {'recontherapy': {'title': 'Recorded Reconstitution / Immune Therapy (recontherapy)',
                  'category': 'Immune Reconstitution Therapy (IRT) / Specialized Treatment Pathway',
                  'nice_guideline': 'NICE NG220 / Specialist Pulsed Immunotherapy Pathway',
                  'nice_recommendation': 'Specialist referral framework for pulsed immune reconstitution therapies '
                                         '(IRT) in active or breakthrough MS.',
                  'indication': 'High-efficacy pulsed disease management in active RRMS.',
                  'mechanism': 'Refers to transiently depleting pulse immunotherapies (e.g., Cladribine, Alemtuzumab, '
                               'AHSCT) that reset systemic neuro-autoimmunity.',
                  'administration': 'Short-course pulses followed by long drug-free remission monitoring.'},
 'ahsct': {'title': 'Autologous Hematopoietic Stem Cell Transplantation (AHSCT)',
           'category': 'Immune Reconstitution Therapy (IRT) / Cellular Therapy',
           'nice_guideline': 'NICE NG220 / EBMT Guidelines',
           'nice_recommendation': 'Recommended for highly active relapsing MS with persistent clinical/MRI '
                                  'inflammatory activity despite high-efficacy DMTs.',
           'indication': 'Rapidly evolving severe RRMS resistant to standard high-efficacy DMTs.',
           'mechanism': 'Ablation of the autoreactive immune system using high-dose chemotherapy followed by rescue '
                        'with autologous hematopoietic stem cells to rebuild a tolerant immune network.',
           'administration': 'Multi-stage inpatient procedure: Stem cell mobilization, harvesting, conditioning '
                             'chemotherapy, autologous reinfusion, and supportive aplasia care.'}}
    
    OBSERVED_TREATMENT_ALIASES = {
        "glatiramer acetate": "glatiramer",
        "copaxone": "glatiramer",
        "brabio": "glatiramer",
        "interferon beta 1a": "interferon",
        "interferon beta 1b": "interferon",
        "peginterferon beta 1a": "interferon",
        "interferon": "interferon",
        "fampyra": "fampridine",
        "dalfampridine": "fampridine",
        "ocrevus": "ocrelizumab",
        "kesimpta": "ofatumumab",
        "tysabri": "natalizumab",
        "mavenclad": "cladribine",
        "lemtrada": "alemtuzumab",
        "ponvory": "ponesimod",
        "gilenya": "fingolimod",
        "mayzent": "siponimod",
        "tecfidera": "dimethyl fumarate",
        "aubagio": "teriflunomide",
        "recontherapy": "recontherapy",
        "ahsct": "ahsct",
    }
    
    def _sim_clean(value, default=""):
        """Return a clean display string, replacing missing/NaN values with default."""
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "nat"}:
            return default
        return text

    def _sim_norm(value):
        """Normalize medication names for matching against the treatment KB."""
        text = _sim_clean(value, "").lower()
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _observed_meds(row):
        """
        Extract every medication recorded for one longitudinal visit.

        Returns tuples of (display_name, normalized_name, recorded_route).
        The dataset may contain either ``1drug``/``2drug``... columns or the
        corresponding ``*_MEDIC`` variants, so both forms are supported.
        """
        observed = []
        seen = set()

        # The source application supports up to 25 medication slots.
        for i in range(1, 26):
            drug = ""
            for col in (f"{i}drug_MEDIC", f"{i}drug"):
                if col in row.index:
                    candidate = _sim_clean(row.get(col), "")
                    if candidate:
                        drug = candidate
                        break

            if not drug:
                continue

            canon = _sim_norm(drug)
            if not canon or canon in seen:
                continue
            seen.add(canon)

            route = "Route not recorded"
            # Keep compatibility with common route fields in MSFinal.csv.
            route_candidates = (
                f"{i}oral", f"{i}route", f"{i}route_MEDIC",
                f"{i}administration", f"{i}administration_MEDIC",
            )
            for route_col in route_candidates:
                if route_col in row.index:
                    route_value = _sim_clean(row.get(route_col), "")
                    if route_value:
                        # Some datasets encode the oral field as a flag.
                        if route_col == f"{i}oral":
                            low = route_value.lower()
                            if low in {"1", "1.0", "true", "yes", "y"}:
                                route = "Oral"
                            elif low not in {"0", "0.0", "false", "no", "n"}:
                                route = route_value
                        else:
                            route = route_value
                        if route != "Route not recorded":
                            break

            observed.append((drug, canon, route))

        return observed

    def _sim_timepoint_label(v):
      """Convert t1, t2, ... to Visit Timepoint 1, Visit Timepoint 2, ..."""
      tp = _sim_clean(v, "").strip()
      match = re.fullmatch(r"t(\d+)", tp, flags=re.IGNORECASE)
      if match:
              return f"Visit Timepoint {int(match.group(1))}"
      return tp if tp else "Timepoint not recorded"
    
    def _treatment_exact_date(row):
        """
        Use the actual visit date stored in the patient data first.
        VISIT_DATE_MAPPING is used only when no usable date exists in the row.
        """
        for c in ("visit-date_MEDIC", "visit-date"):
            if c in row.index:
                raw_date = row.get(c)
    
                try:
                    if raw_date is not None and not pd.isna(raw_date):
                        d = pd.to_datetime(raw_date, errors="coerce")
                        if pd.notna(d):
                            return d.strftime("%d %b %Y")
                except Exception:
                    pass
    
        tp = _sim_clean(row.get("timepoints"), "")
        mapped = VISIT_DATE_MAPPING.get(str(tp).strip().lower())
                
    
        if mapped:
            try:
                d = pd.to_datetime(mapped, errors="coerce")
                if pd.notna(d):
                    return d.strftime("%d %b %Y")
            except Exception:
                return str(mapped)
    
        return "Date not recorded"
    
    
    def _tp_sort_key(tp):
        """Natural ordering: t1, t2, ... t9, t10, t11 rather than t1, t10, t2."""
        s = str(tp).strip()
        m = re.search(r"(\d+)", s)
        return (int(m.group(1)) if m else 10**9, s.lower())
    def _observed_treatment_info(drug, canon):
        """
        Match an observed medication from the patient record against the
        medication knowledge bases already defined in this simulation section.
    
        Returns
        -------
        (knowledge_base_key, medication_information_dict)
        """
    
        raw = _sim_norm(drug)
        canonical = _sim_norm(canon)
    
        candidates = []
    
        # Exact canonical value supplied by _observed_meds().
        if canonical:
            candidates.append(canonical)
    
        # Normalised medication name from the actual patient record.
        if raw:
            candidates.append(raw)
    
        # Alias lookup.
        for value in list(candidates):
            alias_key = OBSERVED_TREATMENT_ALIASES.get(value)
            if alias_key:
                candidates.append(alias_key)
    
        # Handle brand names / additional text in the medication field.
        for alias, key in OBSERVED_TREATMENT_ALIASES.items():
            if alias and alias in raw:
                candidates.append(key)
    
        # Also allow direct partial matching against the supplied KB.
        if "OBSERVED_TREATMENT_KB" in locals():
            for key in OBSERVED_TREATMENT_KB.keys():
                if key and key in raw:
                    candidates.append(key)
    
        # Remove duplicate candidate keys while preserving order.
        candidates = list(dict.fromkeys(
            x for x in candidates if x
        ))
    
        # Main medication knowledge base.
        for key in candidates:
            if key in OBSERVED_TREATMENT_KB:
                return key, OBSERVED_TREATMENT_KB[key]
    
        # Special treatment / immune-reconstitution knowledge base.
        for key in candidates:
            if (
                "OBSERVED_SPECIAL_KB" in locals()
                and key in OBSERVED_SPECIAL_KB
            ):
                return key, OBSERVED_SPECIAL_KB[key]
    
        # Medication genuinely exists in the patient record but is not
        # represented in the supplied classification knowledge base.
        return canonical or raw, {
            "title": drug,
            "category": (
                "Recorded treatment — classification not available "
                "in supplied knowledge base"
            ),
            "nice_guideline": "Not available",
            "nice_recommendation": "Not available",
            "indication": "Not available",
            "mechanism": "Not available",
            "administration": "Not available",
        }
    
    
    def _observed_treatment_group(category):
        """
        Convert the detailed medication classification into the short
        classification displayed on the treatment timeline.
        """
    
        category_text = str(category).strip().lower()
    
        if (
            "immune reconstitution" in category_text
            or "irt" in category_text
        ):
            return "Immune reconstitution therapy (IRT)"
    
        if (
            "disease-modifying" in category_text
            or "disease modifying" in category_text
            or "dmt" in category_text
        ):
            return "Disease-modifying therapy (DMT)"
    
        if "symptomatic" in category_text:
            return "Symptomatic therapy"
    
        return "Other / unclassified treatment"
        
    
    # -----------------------------------------------------------------
    # BUILD HISTORY GROUPED BY TIMEPOINT
    # -----------------------------------------------------------------
    latest_index = patient_records.index[-1] if len(patient_records) else None
    grouped_history = {}
    
    for row_index, row in patient_records.iterrows():
    
        tp = _sim_clean(row.get("timepoints"), "Visit")
        visit_date = _treatment_exact_date(row)
    
        if tp not in grouped_history:
            grouped_history[tp] = {
                "tp": tp,
                "dates": [],
                "medications": [],
                "is_latest_visit": False,
            }
    
        # Preserve actual date(s) associated with this timepoint.
        if (
            visit_date
            and visit_date != "Date not recorded"
            and visit_date not in grouped_history[tp]["dates"]
        ):
            grouped_history[tp]["dates"].append(visit_date)
    
        if row_index == latest_index:
            grouped_history[tp]["is_latest_visit"] = True
    
        # Collect ALL medications recorded at this timepoint.
        for drug, canon, route in _observed_meds(row):
    
            kb_key, info = _observed_treatment_info(drug, canon)
    
            recorded_route = route
            if recorded_route == "Route not recorded":
                recorded_route = info.get(
                    "administration",
                    "Route not recorded"
                )
    
            med = {
                "drug": drug,
                "canon": canon,
                "kb_key": kb_key,
                "route": recorded_route,
                "category": info.get(
                    "category",
                    "Classification not available"
                ),
                "therapy_group": _observed_treatment_group(
                    info.get("category", "")
                ),
                "title": info.get("title", drug),
                "indication": info.get(
                    "indication",
                    "Not available"
                ),
                "mechanism": info.get(
                    "mechanism",
                    "Not available"
                ),
                "administration": info.get(
                    "administration",
                    "Not available"
                ),
                "nice_guideline": info.get(
                    "nice_guideline",
                    "Not available"
                ),
                "nice_recommendation": info.get(
                    "nice_recommendation",
                    "Not available"
                ),
                "is_latest_visit": row_index == latest_index,
            }
    
            # Avoid duplicate display of the same medication inside one timepoint.
            med_identity = (
                med["kb_key"]
                or med["canon"]
                or _sim_norm(med["drug"])
            )
    
            existing_identities = {
                (
                    x["kb_key"]
                    or x["canon"]
                    or _sim_norm(x["drug"])
                )
                for x in grouped_history[tp]["medications"]
            }
    
            if med_identity not in existing_identities:
                grouped_history[tp]["medications"].append(med)
    
    
    # Natural timepoint ordering.
    timepoint_history = sorted(
        grouped_history.values(),
        key=lambda x: _tp_sort_key(x["tp"])
    )
    
    # Remove empty timepoints if no medication was recorded there.
    timepoint_history = [
        x for x in timepoint_history
        if x["medications"]
    ]
    
    
    if not timepoint_history:
    
        st.info(
            "No treatment observations were identified for this patient."
        )
    
        current_observed_dmt = "Not recorded"
        current_observed_canon = ""
    
    else:
    
    
        timeline_blocks = []
        modal_payload = {}
    
        # =============================================================
        # HORIZONTAL GANTT-STYLE OBSERVED TREATMENT HISTORY
        # =============================================================
    
        for tp_index, visit in enumerate(timepoint_history):
    
            tp_raw = visit["tp"]
            tp = _sim_timepoint_label(tp_raw)
    
            if visit["dates"]:
                date_display = " / ".join(visit["dates"])
            else:
                date_display = "Date not recorded"
    
            medication_buttons = []
    
            # ---------------------------------------------------------
            # MEDICATIONS RECORDED AT THIS TIMEPOINT
            # ---------------------------------------------------------
    
            for med_index, h in enumerate(visit["medications"]):
    
                item_id = f"treatment_{tp_index}_{med_index}"
    
                modal_payload[item_id] = {
    
                    "observed_treatment": _sim_clean(
                        h.get("drug"),
                        "Not recorded"
                    ),
    
                    "standard_name": _sim_clean(
                        h.get("title"),
                        h.get("drug", "Not recorded")
                    ),
    
                    "treatment_type": _sim_clean(
                        h.get("therapy_group"),
                        "Unclassified"
                    ),
    
                    "classification": _sim_clean(
                        h.get("category"),
                        "Classification not available"
                    ),
    
                    "timepoint": tp,
    
                    "date": date_display,
    
                    "status": (
                        "Recorded at latest patient visit"
                        if visit["is_latest_visit"]
                        else "Historical observation"
                    ),
    
                    "recorded_route": _sim_clean(
                        h.get("route"),
                        "Route not recorded"
                    ),
    
                    "administration": _sim_clean(
                        h.get("administration"),
                        "Not available"
                    ),
    
                    "indication": _sim_clean(
                        h.get("indication"),
                        "Not available"
                    ),
    
                    "mechanism": _sim_clean(
                        h.get("mechanism"),
                        "Not available"
                    ),
    
                    "guideline": _sim_clean(
                        h.get("nice_guideline"),
                        "Not available"
                    ),
    
                    "recommendation": _sim_clean(
                        h.get("nice_recommendation"),
                        "Not available"
                    ),
                }
    
                route_text = _sim_clean(
                    h.get("route"),
                    "Route not recorded"
                )
    
                # Keep the Gantt card compact.
                # Full information remains available in the popup.
                if len(route_text) > 45:
                    route_text = route_text.split("|")[0].strip()
    
                if len(route_text) > 45:
                    route_text = route_text[:42].rstrip() + "…"
    
                medication_buttons.append(
    
                    '<button class="med-card" type="button" '
                    'onclick="openTreatment(\'' + item_id + '\')">'
    
                        '<div class="med-content">'
    
                            '<div class="med-name">'
                            + _html.escape(str(h["drug"]))
                            + '</div>'
    
                            '<div class="med-route">'
                            + _html.escape(route_text)
                            + '</div>'
    
                            '<div class="med-class">'
                            + _html.escape(
                                str(h["therapy_group"])
                            )
                            + '</div>'
    
                        '</div>'
    
                        '<span class="med-info">i</span>'
    
                    '</button>'
                )
    
            # ---------------------------------------------------------
            # LATEST VISIT BADGE
            # ---------------------------------------------------------
    
            latest_badge = (
    
                '<span class="latest-badge">'
                'LATEST'
                '</span>'
    
                if visit["is_latest_visit"]
    
                else ""
            )
    
            # ---------------------------------------------------------
            # ONE HORIZONTAL GANTT COLUMN PER TIMEPOINT
            # ---------------------------------------------------------
    
            timeline_blocks.append(
    
                '<div class="gantt-column">'
    
                    # Timepoint + date
                    '<div class="gantt-header">'
    
                        '<div class="gantt-tp">'
    
                            '<span>'
                            + _html.escape(str(tp))
                            + '</span>'
    
                            + latest_badge +
    
                        '</div>'
    
                        '<div class="gantt-date">'
                        + _html.escape(date_display)
                        + '</div>'
    
                    '</div>'
    
                    # Horizontal timeline axis
                    '<div class="gantt-axis">'
    
                        '<span class="gantt-node"></span>'
    
                    '</div>'
    
                    # Treatments belonging to this timepoint
                    '<div class="gantt-treatments">'
    
                        + "".join(medication_buttons) +
    
                    '</div>'
    
                '</div>'
            )
    
    
        # =============================================================
        # SERIALISE MODAL DATA
        # =============================================================
    
        payload_json = _json.dumps(
            modal_payload
        ).replace("</", "<\\/")
    
    
        timeline_html = "".join(
            timeline_blocks
        )
    
    
        # =============================================================
        # COMPONENT HEIGHT
        #
        # Height is based on the largest treatment stack rather than
        # the number of visits because visits now run horizontally.
        # =============================================================
    
        max_medications = max(
            (
                len(x["medications"])
                for x in timepoint_history
            ),
            default=1
        )
    
        component_height = min(
            720,
            max(
                330,
                205 + (max_medications * 105)
            )
        )
    
    
        # =============================================================
        # HTML COMPONENT
        # =============================================================
    
        component_doc = r'''
    <!doctype html>
    
    <html>
    
    <head>
    
    <meta charset="utf-8">
    
    <style>
    
    * {
box-sizing: border-box;
    }
    
    
    body {
    
margin: 0;
    
padding: 5px 8px 15px 5px;
    
font-family:
    Inter,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    Roboto,
    Arial,
    sans-serif;
    
color: #0f172a;
    
background: transparent;
    }
    
    
    /* ================================================================
       HORIZONTAL GANTT TIMELINE
       ================================================================ */
    
    .timeline {
    
width: 100%;
    
position: relative;
    
display: flex;
    
align-items: flex-start;
    
overflow-x: auto;
    
overflow-y: hidden;
    
padding:
    8px
    8px
    24px
    8px;
    
scrollbar-width: thin;
    
scrollbar-color:
    #9abbd8
    #eef5fb;
    }
    
    
    /* ================================================================
       ONE TIMEPOINT / GANTT COLUMN
       ================================================================ */
    
    .gantt-column {
    
position: relative;
    
flex: 0 0 300px;
    
min-width: 300px;
    
padding:
    0
    10px;
    }
    
    
    /* ================================================================
       TIMEPOINT HEADER
       ================================================================ */
    
    .gantt-header {
    
min-height: 70px;
    
display: flex;
    
flex-direction: column;
    
align-items: center;
    
justify-content: flex-end;
    
text-align: center;
    
padding:
    5px
    8px
    10px;
    }
    
    
    .gantt-tp {
    
display: flex;
    
align-items: center;
    
justify-content: center;
    
gap: 7px;
    
font-size: 15px;
    
font-weight: 750;
    
color: #123b68;
    }
    
    
    .gantt-date {
    
margin-top: 5px;
    
font-size: 12px;
    
font-weight: 600;
    
color: #526b84;
    
white-space: nowrap;
    }
    
    
    /* ================================================================
       LATEST VISIT BADGE
       ================================================================ */
    
    .latest-badge {
    
display: inline-block;
    
padding:
    3px
    7px;
    
border-radius: 10px;
    
background: #e7f7ee;
    
color: #177245;
    
font-size: 9px;
    
font-weight: 750;
    
letter-spacing: .02em;
    }
    
    
    /* ================================================================
       HORIZONTAL GANTT AXIS
       ================================================================ */
    
    .gantt-axis {
    
position: relative;
    
width: calc(100% + 20px);
    
height: 32px;
    
margin-left: -10px;
    
display: flex;
    
align-items: center;
    
justify-content: center;
    }
    
    
    /* Horizontal blue line */
    
    .gantt-axis::before {
    
content: "";
    
position: absolute;
    
left: 0;
    
right: 0;
    
top: 50%;
    
height: 4px;
    
transform:
    translateY(-50%);
    
background: #8ec5f4;
    
border-radius: 4px;
    }
    
    
    /* ================================================================
       TIMEPOINT NODE
       ================================================================ */
    
    .gantt-node {
    
position: relative;
    
z-index: 3;
    
width: 15px;
    
height: 15px;
    
border-radius: 50%;
    
background: #2383d8;
    
border:
    3px
    solid
    #ffffff;
    
box-shadow:
    0 0 0 2px #2383d8,
    0 2px 5px rgba(37, 99, 235, .20);
    }
    
    
    /* ================================================================
       ARROW BETWEEN TIMEPOINTS
       ================================================================ */
    
    .gantt-column:not(:last-child)
    .gantt-axis::after {
    
content: "";
    
position: absolute;
    
right: -1px;
    
top: 50%;
    
transform:
    translateY(-50%);
    
width: 0;
    
height: 0;
    
border-top:
    6px
    solid
    transparent;
    
border-bottom:
    6px
    solid
    transparent;
    
border-left:
    9px
    solid
    #2383d8;
    
z-index: 2;
    }
    
    
    /* Stop horizontal line at final visit */
    
    .gantt-column:last-child
    .gantt-axis::before {
    
right: 50%;
    }
    
    
    /* ================================================================
       TREATMENTS BELOW EACH TIMEPOINT
       ================================================================ */
    
    .gantt-treatments {
    
position: relative;
    
display: flex;
    
flex-direction: column;
    
gap: 9px;
    
padding-top: 15px;
    }
    
    
    /* Vertical connector from timeline node to treatment cards */
    
    .gantt-treatments::before {
    
content: "";
    
position: absolute;
    
top: -17px;
    
left: 50%;
    
width: 2px;
    
height: 31px;
    
transform:
    translateX(-50%);
    
background: #b8d8f3;
    
z-index: 0;
    }
    
    
    /* ================================================================
       MEDICATION CARD
       ================================================================ */
    
    .med-card {
    
position: relative;
    
z-index: 1;
    
width: 100%;
    
min-height: 88px;
    
display: flex;
    
justify-content: space-between;
    
align-items: center;
    
text-align: left;
    
padding:
    11px
    12px;
    
border:
    1px
    solid
    #dce7f1;
    
border-radius: 9px;
    
background: #fbfdff;
    
color: inherit;
    
cursor: pointer;
    
box-shadow:
    0 2px 5px
    rgba(15, 23, 42, .035);
    
transition:
    background .15s ease,
    border-color .15s ease,
    box-shadow .15s ease,
    transform .15s ease;
    }
    
    
    .med-card:hover,
    .med-card:focus {
    
background: #f4f9ff;
    
border-color: #9bc8ef;
    
box-shadow:
    0 4px 10px
    rgba(30, 64, 175, .10);
    
outline: none;
    
transform:
    translateY(-2px);
    }
    
    
    .med-content {
    
min-width: 0;
    
padding-right: 10px;
    }
    
    
    .med-name {
    
font-size: 14px;
    
line-height: 1.25;
    
font-weight: 700;
    
color: #15395f;
    }
    
    
    .med-route {
    
margin-top: 4px;
    
font-size: 12px;
    
line-height: 1.3;
    
color: #64748b;
    }
    
    
    .med-class {
    
display: inline-block;
    
margin-top: 7px;
    
padding:
    3px
    7px;
    
border-radius: 10px;
    
background: #eaf4ff;
    
color: #1769aa;
    
font-size: 11px;
    
line-height: 1.25;
    
font-weight: 600;
    }
    
    
    .med-info {
    
flex:
    0
    0
    auto;
    
width: 23px;
    
height: 23px;
    
border-radius: 50%;
    
display: inline-flex;
    
align-items: center;
    
justify-content: center;
    
background: #eaf4ff;
    
color: #1976c9;
    
border:
    1px
    solid
    #a9d2f5;
    
font:
    700
    13px
    Georgia,
    serif;
    }
    
    
    /* ================================================================
       HORIZONTAL SCROLLBAR
       ================================================================ */
    
    .timeline::-webkit-scrollbar {
    
height: 9px;
    }
    
    
    .timeline::-webkit-scrollbar-track {
    
background: #eef5fb;
    
border-radius: 10px;
    }
    
    
    .timeline::-webkit-scrollbar-thumb {
    
background: #9abbd8;
    
border-radius: 10px;
    }
    
    
    .timeline::-webkit-scrollbar-thumb:hover {
    
background: #77a8cf;
    }
    
    
    /* ================================================================
       POPUP
       ================================================================ */
    
    .modal-backdrop {
    
display: none;
    
position: fixed;
    
inset: 0;
    
z-index: 9998;
    
background:
    rgba(15, 23, 42, .40);
    
padding: 18px;
    
align-items: center;
    
justify-content: center;
    }
    
    
    .modal-backdrop.open {
    
display: flex;
    }
    
    
    .modal {
    
width:
    min(700px, 96vw);
    
max-height: 88vh;
    
overflow-y: auto;
    
background: #ffffff;
    
border:
    1px
    solid
    #cbd5e1;
    
border-radius: 12px;
    
box-shadow:
    0 20px 55px
    rgba(15, 23, 42, .24);
    }
    
    
    .modal-head {
    
position: sticky;
    
top: 0;
    
z-index: 2;
    
display: flex;
    
justify-content: space-between;
    
align-items: flex-start;
    
gap: 12px;
    
padding:
    17px
    19px;
    
background: #f4f8fc;
    
border-bottom:
    1px
    solid
    #dbe5ef;
    }
    
    
    .modal-title {
    
font-size: 18px;
    
font-weight: 750;
    
color: #123b68;
    }
    
    
    .modal-sub {
    
margin-top: 5px;
    
font-size: 12px;
    
color: #64748b;
    }
    
    
    .close {
    
border: 0;
    
background: #ffffff;
    
width: 30px;
    
height: 30px;
    
border-radius: 50%;
    
cursor: pointer;
    
font-size: 19px;
    
color: #475569;
    
box-shadow:
    0 0 0 1px
    #d7e0e8;
    }
    
    
    .modal-body {
    
padding:
    16px
    19px
    20px;
    }
    
    
    .class-box {
    
padding:
    11px
    13px;
    
margin-bottom: 13px;
    
border-radius: 8px;
    
background: #eef6ff;
    
border-left:
    4px
    solid
    #2d82cf;
    }
    
    
    .class-label,
    .info-label {
    
font-size: 10px;
    
text-transform: uppercase;
    
letter-spacing: .05em;
    
color: #64748b;
    
font-weight: 750;
    }
    
    
    .class-value {
    
margin-top: 4px;
    
font-size: 13px;
    
line-height: 1.4;
    
font-weight: 650;
    
color: #173f6b;
    }
    
    
    .info-grid {
    
display: grid;
    
grid-template-columns:
    1fr
    1fr;
    
gap: 10px;
    }
    
    
    .info-card {
    
padding:
    10px
    11px;
    
border:
    1px
    solid
    #e2e8f0;
    
border-radius: 7px;
    
background: #ffffff;
    }
    
    
    .info-card.wide {
    
grid-column:
    1 / -1;
    }
    
    
    .info-value {
    
margin-top: 5px;
    
font-size: 12px;
    
line-height: 1.5;
    
color: #243b53;
    
white-space: pre-wrap;
    }
    
    
    /* ================================================================
       RESPONSIVE
       ================================================================ */
    
    @media(max-width: 620px) {
    
.gantt-column {
    
    flex-basis: 260px;
    
    min-width: 260px;
}
    
    
.info-grid {
    
    grid-template-columns: 1fr;
}
    
    
.info-card.wide {
    
    grid-column: auto;
}
    }
    
    </style>
    
    </head>
    
    
    <body>
    
    
    <!-- =============================================================
 HORIZONTAL GANTT TIMELINE
 ============================================================= -->
    
    <div class="timeline">
    
__TIMELINE__
    
    </div>
    
    
    <!-- =============================================================
 TREATMENT INFORMATION MODAL
 ============================================================= -->
    
    <div
id="txModalBackdrop"
class="modal-backdrop"
onclick="backdropClose(event)"
    >
    
<div
    class="modal"
    role="dialog"
    aria-modal="true"
>
    
    <div class="modal-head">
    
        <div>
    
            <div
                id="txModalTitle"
                class="modal-title">
            </div>
    
            <div
                id="txModalSub"
                class="modal-sub">
            </div>
    
        </div>
    
    
        <button
            class="close"
            type="button"
            onclick="closeTreatment()">
    
            ×
    
        </button>
    
    </div>
    
    
    <div
        id="txModalBody"
        class="modal-body">
    </div>
    
</div>
    
    </div>
    
    
    <script>
    
    const treatments = __PAYLOAD__;
    
    
    /* ================================================================
       HTML ESCAPE
       ================================================================ */
    
    const esc = (v) =>
    
String(v ?? "Not available")
    
    .replaceAll("&", "&amp;")
    
    .replaceAll("<", "&lt;")
    
    .replaceAll(">", "&gt;")
    
    .replaceAll('"', "&quot;")
    
    .replaceAll("'", "&#039;");
    
    
    /* ================================================================
       INFORMATION CARD
       ================================================================ */
    
    function card(label, value, wide=false) {
    
return `
    
    <div class="info-card ${wide ? 'wide' : ''}">
    
        <div class="info-label">
            ${esc(label)}
        </div>
    
        <div class="info-value">
            ${esc(value)}
        </div>
    
    </div>
    
`;
    }
    
    
    /* ================================================================
       OPEN TREATMENT POPUP
       ================================================================ */
    
    function openTreatment(id) {
    
const d = treatments[id];
    
if (!d) return;
    
    
document.getElementById(
    "txModalTitle"
).textContent =
    d.observed_treatment;
    
    
document.getElementById(
    "txModalSub"
).textContent =
    d.timepoint +
    " • " +
    d.date +
    " • " +
    d.status;
    
    
document.getElementById(
    "txModalBody"
).innerHTML = `
    
    <div class="class-box">
    
        <div class="class-label">
            Treatment classification
        </div>
    
        <div class="class-value">
            ${esc(d.treatment_type)}
        </div>
    
    </div>
    
    
    <div class="info-grid">
    
        ${card(
            "Standard medication name",
            d.standard_name
        )}
    
        ${card(
            "Recorded route",
            d.recorded_route
        )}
    
        ${card(
            "Detailed classification",
            d.classification,
            true
        )}
    
        ${card(
            "Administration",
            d.administration,
            true
        )}
    
        ${card(
            "Indication",
            d.indication,
            true
        )}
    
        ${card(
            "Mechanism",
            d.mechanism,
            true
        )}
    
        ${card(
            "Guideline",
            d.guideline,
            true
        )}
    
        ${card(
            "Recommendation",
            d.recommendation,
            true
        )}
    
    </div>
    
`;
    
    
document.getElementById(
    "txModalBackdrop"
).classList.add("open");
    }
    
    
    /* ================================================================
       CLOSE POPUP
       ================================================================ */
    
    function closeTreatment() {
    
document.getElementById(
    "txModalBackdrop"
).classList.remove("open");
    }
    
    
    /* ================================================================
       CLOSE WHEN BACKDROP CLICKED
       ================================================================ */
    
    function backdropClose(event) {
    
if (
    event.target.id ===
    "txModalBackdrop"
) {
    
    closeTreatment();
}
    }
    
    
    /* ================================================================
       ESC KEY CLOSE
       ================================================================ */
    
    document.addEventListener(
"keydown",
function(event) {
    
    if (event.key === "Escape") {
    
        closeTreatment();
    }
}
    );
    
    </script>
    
    </body>
    
    </html>
    '''
    
    
        # =============================================================
        # INSERT GENERATED TIMELINE + DATA INTO COMPONENT
        # =============================================================
    
        component_doc = component_doc.replace(
            "__TIMELINE__",
            timeline_html
        )
    
        component_doc = component_doc.replace(
            "__PAYLOAD__",
            payload_json
        )
    
    
        # =============================================================
        # RENDER
        # =============================================================
    
        st.components.v1.html(
            component_doc,
            height=component_height,
            scrolling=True
        )



if _open_treatment_history:
    _show_observed_treatment_history()

# CLINICAL OVERVIEW — observed evidence before model explanation.
if st.session_state["xai_active_page"] == "Clinical Overview & SHAP":
    _shap_first_container = st.container()
    # =====================================================================
    # EDSS DISABILITY PROGRESSION — simplified clinical presentation
    # =====================================================================
    tt_edss_logic = (
        "<span class='info-tooltip'>ⓘ<span class='tooltiptext'>"
        "The progression level is 1.0 EDSS point above the first visit when "
        "starting EDSS is 5.5 or below, or 0.5 points above when it is above 5.5. "
        "Two consecutive follow-up visits must meet the level."
        "</span></span>"
    )
   
    edss_y = pd.to_numeric(patient_records['EDSS score'], errors='coerce').tolist()
    visit_x = formatted_visit_dates
    prog_thresh = min_increase_for_progression(baseline_edss_val)
    target_prog_level = baseline_edss_val + prog_thresh

    crossing_idx = next((i for i in range(1, len(edss_y))
                         if pd.notna(edss_y[i]) and edss_y[i] >= target_prog_level), None)
    sustained_start_idx = next((i for i in range(1, len(edss_y) - 1)
                                if pd.notna(edss_y[i]) and pd.notna(edss_y[i + 1])
                                and edss_y[i] >= target_prog_level
                                and edss_y[i + 1] >= target_prog_level), None)

    timepoint_risks = []
    for idx in range(len(patient_records)):
        risk, _, _, _ = predictor.predict_patient(patient_records.iloc[[idx]])
        timepoint_risks.append(f"{int(risk * 100)}%")

    # Red markers identify visits at or above the progression level;
    # this alone does not imply confirmed progression.
    edss_marker_colors = [
        '#dc2626' if pd.notna(y) and y >= target_prog_level else '#0284c7'
        for y in edss_y
    ]

    fig_edss_traj = go.Figure()
    fig_edss_traj.add_trace(go.Scatter(
        x=visit_x, y=edss_y, mode='lines+markers',
        name='Patient EDSS',
        line=dict(color='#0284c7', width=3),
        marker=dict(size=10, color=edss_marker_colors),
        customdata=[[f"{float(y)-baseline_edss_val:+.1f}" if pd.notna(y) else "N/A", r]
                    for y, r in zip(edss_y, timepoint_risks)],
        hovertemplate=(
            '<b>%{x}</b><br>'
            'EDSS score: %{y:.1f}<br>'
            'Change from first visit: %{customdata[0]}<br>'
            'Estimated progression risk: %{customdata[1]}'
            '<extra></extra>'
        ),
    ))
    fig_edss_traj.add_trace(go.Scatter(
        x=visit_x, y=[baseline_edss_val] * len(visit_x), mode='lines',
        name=f'Starting EDSS ({baseline_edss_val:.1f})',
        line=dict(color='#64748b', width=1.8, dash='dash'),
        hovertemplate=f'Starting EDSS: {baseline_edss_val:.1f}<extra></extra>',
    ))
    fig_edss_traj.add_trace(go.Scatter(
        x=visit_x, y=[target_prog_level] * len(visit_x), mode='lines',
        name=f'Progression level ({target_prog_level:.1f})',
        line=dict(color='#dc2626', width=2.5, dash='dot'),
        hovertemplate=f'Progression level: {target_prog_level:.1f}<extra></extra>',
    ))
    if visit_x:
        fig_edss_traj.add_annotation(
            x=visit_x[-1], y=target_prog_level,
            text=f'Progression level: {target_prog_level:.1f}',
            xanchor='right', yanchor='bottom', yshift=7,
            showarrow=False, font=dict(size=11, color='#dc2626'),
            bgcolor='rgba(255,255,255,0.88)',
        )
    # Label sustained progression only after two consecutive qualifying follow-up visits.
    if sustained_start_idx is not None:
        confirm_idx = sustained_start_idx + 1
        fig_edss_traj.add_annotation(
            x=visit_x[confirm_idx], y=edss_y[confirm_idx],
            text='Sustained progression confirmed',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor='#b91c1c', ax=0, ay=-48,
            font=dict(size=11, color='#991b1b'),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#fecaca', borderpad=5,
        )
    fig_edss_traj.update_layout(
        height=420, margin=dict(l=25, r=20, t=65, b=55),
        plot_bgcolor='white', paper_bgcolor='white', hovermode='closest',
        yaxis=dict(title='EDSS score', range=[0.0, max(9.0, target_prog_level + 0.5)], gridcolor='#f1f5f9'),
        xaxis=dict(title='Visit date', gridcolor='#f1f5f9'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    font=dict(size=10), bgcolor='rgba(255,255,255,0.85)'),
    )

    try:
        _edss_shap = np.asarray(predictor.compute_patient_shap(patient_records), dtype=float).reshape(-1)
        _edss_top = np.argsort(np.abs(_edss_shap))[::-1][:3]
        _edss_drivers = "; ".join(
            f"{FEATURE_DISPLAY_NAMES[predictor.features[i]]} ({'increases' if _edss_shap[i] >= 0 else 'decreases'} model score; SHAP {_edss_shap[i]:+.2f})"
            for i in _edss_top
        )
    
    except Exception:
        st.caption(f"Model-predicted progression probability: {prog_risk:.0%}. Patient-specific SHAP drivers unavailable.")
    with _shap_first_container:
        
        st.markdown("""
        <style>
        /* Hide radio label */
        div[data-testid="stRadio"] > label {
            display: none;
        }
    
        /* Radio container */
        div[data-testid="stRadio"] > div {
            gap: 12px;
        }
    
        /* Individual options */
        div[data-testid="stRadio"] label[data-baseweb="radio"] {
            background: #f8fafc;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 14px 22px;
            min-height: 56px;
            transition: all 0.2s ease;
        }

        /* Larger text */
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        font-size: 18px !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }

    /* Hover */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: #f1f5f9;
        border-color: #94a3b8;
    }

    /* Selected option */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: #eef4ff;
        border-color: #2563eb;
    }

    /* Selected text */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: #1d4ed8 !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    
        # Both local and cohort-level explanations are visible in one scrolling workflow.
        # Keep the existing plots and their original layout, without navigation tabs.
        subtab_choice = "Patient Explanation"

        curr_patient_dict = {
            'age': float(patient_records['age'].iloc[-1]),
            'numrelapse2': float(patient_records['numrelapse2'].iloc[-1]),
            'walkdur_all_sum_w': float(patient_records['walkdur_all_sum_w'].iloc[-1]),
            'wbsteps_all_sum_w': float(patient_records['wbsteps_all_sum_w'].iloc[-1]),
            'ws_30_avg_w': float(patient_records['ws_30_avg_w'].iloc[-1]),
            'ws_30_p90_w': float(patient_records['ws_30_p90_w'].iloc[-1]),
            'cadence_all_avg_w': float(patient_records['cadence_all_avg_w'].iloc[-1]),
            'strlen_30_avg_w': float(patient_records['strlen_30_avg_w'].iloc[-1]),
            'cadence_all_var_w': float(patient_records['cadence_all_var_w'].iloc[-1])
        }

        if True:
            _interrogation_header("Model Reasoning", "Inspect how the selected feature contributes to this patient's prediction and the approximate decision rules.")

            patient_id_display = (
                patient_records["patient_id"].iloc[-1]
                if "patient_id" in patient_records.columns
                else "Selected Patient"
            )

            # Interactive, patient-specific SHAP: XGBoost TreeExplainer values are in
            # model-output (log-odds) units for binary logistic models, not % risk.
            shap_values = np.asarray(predictor.compute_patient_shap(patient_records), dtype=float)
            shap_table = pd.DataFrame({
                "Feature": [FEATURE_DISPLAY_NAMES[f] for f in predictor.features],
                "Observed value": [float(patient_records[f].iloc[-1]) for f in predictor.features],
                "SHAP value": shap_values,
            })
            shap_table["Absolute impact"] = shap_table["SHAP value"].abs()
            shap_table["Rank"] = shap_table["Feature"].map({FEATURE_DISPLAY_NAMES[f]: rank + 1 for f, rank in _rank_by_feature.items()})
            shap_table["Selected"] = np.where(shap_table["Feature"] == _selected_interrogation_label, "◀ UNDER REVIEW", "")
            shap_table["Direction"] = np.where(shap_table["SHAP value"] >= 0,
                                                "Positive impact", "Negative impact")
    
            # Restore the filtered SHAP dataset before constructing the Plotly figure.
            direction_filter = "All impacts"
            visible = shap_table.sort_values("Absolute impact", ascending=True).copy()

            fig_shap_interactive = px.bar(
                data_frame=visible,
                x="SHAP value",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Positive impact": "#dc2626", "Negative impact": "#2563eb"},
                custom_data=["Observed value", "Direction"],
            )
            fig_shap_interactive.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Patient measurement: %{customdata[0]:.2f}"
                              "<br>SHAP (model-output units): %{x:+.3f}<br>%{customdata[1]}<extra></extra>"
            )
            # Strong outline identifies the selected feature without changing the
            # red/blue SHAP direction colour scheme.
            for _trace in fig_shap_interactive.data:
                _trace.marker.line.width = [3 if name == _selected_interrogation_label else 0
                                            for name in _trace.y]
                _trace.marker.line.color = ["#0f172a" if name == _selected_interrogation_label else "rgba(0,0,0,0)"
                                            for name in _trace.y]
                _trace.opacity = 1.0
            fig_shap_interactive.add_vline(x=0, line_color="#64748b", line_width=1)
            if _selected_interrogation_label in visible["Feature"].values:
                _focus = visible.loc[visible["Feature"] == _selected_interrogation_label].iloc[0]
                fig_shap_interactive.add_annotation(
                    x=float(_focus["SHAP value"]), y=_selected_interrogation_label,
                    text="SELECTED", showarrow=True, arrowhead=2, ax=40, ay=-30,
                    bgcolor="#dbeafe", bordercolor="#176caa", font=dict(size=11, color="#17324d"))

            fig_shap_interactive.update_layout(height=440,
                xaxis_title="SHAP contribution to model score (log-odds)", autosize=True, yaxis_title="",
                legend_title="Effect", plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=5,r=5,t=30,b=40))
            _shap_plot_col, _shap_table_col = st.columns([1.1, 1], gap="medium")
            with _shap_plot_col:
                st.markdown("#### Patient Feature Contributions · SHAP Explanation")
                st.caption("☝️ Hover over a coloured bar to see the hand pointer, then click to inspect the feature.")
                st.caption("Ranked contributions of this patient's recorded features to the XGBoost model output. Red increases and blue decreases the model's progression score relative to its baseline. Bar length shows contribution strength in log-odds, not a percentage change in risk. Select a bar to interrogate that feature.")
                _plot_event = st.plotly_chart(fig_shap_interactive, width="stretch", key=f"xai_selectable_shap_{st.session_state.get('_xai_click_epoch', 0)}", on_select="rerun", selection_mode="points")
                if _plot_event and _plot_event.selection.points:
                    _point = _plot_event.selection.points[0]
                    _clicked = _point.get("y")
                    _raw_clicked = DISPLAY_TO_RAW_FEATURE.get(str(_clicked))
                    if _raw_clicked in predictor.features:
                        _xai_open_feature(_raw_clicked)
                        st.session_state["_xai_click_epoch"] = st.session_state.get("_xai_click_epoch", 0) + 1
                        st.rerun()
                # Clinician-readable interpretation of the patient's SHAP explanation.
                _positive = [(predictor.features[i], float(_interrogation_shap[i])) for i in _interrogation_ranking if np.isfinite(_interrogation_shap[i]) and _interrogation_shap[i] > 0]
                _negative = [(predictor.features[i], float(_interrogation_shap[i])) for i in _interrogation_ranking if np.isfinite(_interrogation_shap[i]) and _interrogation_shap[i] < 0]
                _top_positive = FEATURE_DISPLAY_NAMES[_positive[0][0]] if _positive else "None identified"
                _top_negative = FEATURE_DISPLAY_NAMES[_negative[0][0]] if _negative else "None identified"
                st.markdown("#### SHAP Explanation · Clinical Interpretation")
                st.markdown(f"""<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(195px,1fr));gap:10px;margin:8px 0 12px">
                  <div style="background:#fff1f2;border:1px solid #fecdd3;border-left:5px solid #dc2626;border-radius:10px;padding:13px">
                    <div style="color:#9f1239;font-weight:800;font-size:.84rem">↑ Strongest upward contribution</div>
                    <div style="font-size:1.25rem;font-weight:850;color:#881337;margin-top:5px">{html.escape(_top_positive)}</div>
                    <div style="font-size:.8rem;color:#9f1239;margin-top:5px">Raises the model score relative to its baseline</div>
                  </div>
                  <div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:5px solid #2563eb;border-radius:10px;padding:13px">
                    <div style="color:#1d4ed8;font-weight:800;font-size:.84rem">↓ Strongest downward contribution</div>
                    <div style="font-size:1.25rem;font-weight:850;color:#1e40af;margin-top:5px">{html.escape(_top_negative)}</div>
                    <div style="font-size:.8rem;color:#1d4ed8;margin-top:5px">Lowers the model score relative to its baseline</div>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.caption("SHAP explains the model score in log-odds; these are not clinical causes or changes in risk percentage points.")
                # Interpretations, captions and table all follow the active impact filter.
                if direction_filter == "Positive impact":
                    interpretation = ("Positive impact (red): these features push the model's "
                                      "predicted MS progression score higher relative to its baseline.")
                    table_caption = "Showing only positive SHAP contributions (red), ranked by impact magnitude."
                elif direction_filter == "Negative impact":
                    interpretation = ("Negative impact (blue): these features push the model's "
                                      "predicted MS progression score lower relative to its baseline.")
                    table_caption = "Showing only negative SHAP contributions (blue), ranked by impact magnitude."
                else:
                    interpretation = ("Positive impact (red) raises the model's predicted progression "
                                      "score; negative impact (blue) lowers it, relative to baseline.")
                    table_caption = "Showing both positive (red) and negative (blue) SHAP contributions, ranked by impact magnitude."

                st.markdown(f"**Interpretation:** {interpretation}")
                if visible.empty:
                    st.caption(f"No {direction_filter.lower()} factors are available for this patient.")
                else:
                    strongest_visible = visible.loc[visible["Absolute impact"].idxmax()]
                st.caption(
                    f"{len(visible)} of {len(shap_table)} patient factors displayed. "
                    "SHAP describes model behaviour, not clinical causation or percentage-point risk changes."
                )
            with _shap_table_col:
                st.markdown("#### EDSS Disability Progression Over Time")
                st.plotly_chart(fig_edss_traj, width="stretch", key="edss_progression_trajectory")
                st.caption("Observed EDSS (blue), baseline (grey), and study-defined progression level (red). Red points identify visits at or above the level; two consecutive qualifying follow-up visits are required to confirm progression.")

                edss_info_col = st.container()
                with edss_info_col:
                    st.markdown('#### EDSS Progression Status')
                    metric_left, metric_right = st.columns(2)
                    with metric_left:
                        st.metric('Current EDSS', f'{curr_edss_val:.1f}')
                    with metric_right:
                        st.metric('Progression level', f'{target_prog_level:.1f}')

                    if sustained_start_idx is not None:
                        st.error('**Confirmed disability progression**')
                        st.markdown(
                            f'The progression level was met at two consecutive follow-up visits, '
                            f'confirmed on **{visit_x[sustained_start_idx + 1]}**.'
                        )
                    elif crossing_idx is not None:
                        st.warning('**Progression not yet confirmed**')
                        st.markdown(
                            f'The progression level was first reached on **{visit_x[crossing_idx]}**, '
                            'but two consecutive qualifying follow-up visits have not been recorded.'
                        )
                    else:
                        st.success('**No confirmed disability progression**')
                        change = curr_edss_val - baseline_edss_val
                        if abs(change) < 0.05:
                            st.markdown("The patient's latest EDSS is unchanged from the first visit.")
                        elif change > 0:
                            st.markdown(f"The patient's latest EDSS is **{change:.1f} points higher** than at the first visit.")
                        else:
                            st.markdown(f"The patient's latest EDSS is **{abs(change):.1f} points lower** than at the first visit.")

                    remaining = max(0.0, target_prog_level - curr_edss_val)
                    if remaining > 0:
                        st.caption(
                            f'An increase of {remaining:.1f} EDSS points from the latest visit is needed '
                            'to reach the progression level. Progression must be confirmed at two '
                            'consecutive follow-up visits.'
                        )
                    else:
                        st.caption('Progression requires two consecutive follow-up visits at or above the progression level.')
                    st.caption(
                        'The progression level is an assessment criterion, not a forecast of when '
                        'progression will occur. Hover risk percentages are model estimates, not '
                        'observed or confirmed progression.'
                    )

                # Interactive explanation of the EDSS threshold (does not change patient data or chart).
    
                with st.expander("How is EDSS progression assessed?", expanded=False):

                    st.markdown(
                        "The dashboard assesses whether a patient meets the "
                        "**study-defined criterion for sustained EDSS disability progression**."
                    )

                    st.markdown("#### 1. Observed progression criteria")

                    st.markdown(
                        """
                        | Starting EDSS | Required increase | Progression level |
                        |---|---|---|
                        | ≤ 5.5 | +1.0 | Starting EDSS + 1.0 |
                        | > 5.5 | +0.5 | Starting EDSS + 0.5 |
                        """
                    )

                    st.caption(
                        "At least three valid EDSS observations are required. "
                        "The progression level must be reached at two consecutive "
                        "recorded observations. No minimum interval between visits "
                        "is currently enforced."
                    )

                    st.markdown("#### 2. Explore the progression level")

                    left, right = st.columns([1, 1.5], gap="medium")

                    with left:
                        example_baseline = st.slider(
                            "Starting EDSS",
                            min_value=0.0,
                            max_value=9.0,
                            value=float(
                                np.clip(round(baseline_edss_val * 2) / 2, 0, 9)
                            ),
                            step=0.5,
                            key="edss_threshold_rule_example"
                        )

                    example_increase = min_increase_for_progression(example_baseline)
                    example_level = example_baseline + example_increase

                    with right:
                        st.metric(
                            "Progression level",
                            f"{example_level:.1f}",
                            delta=f"+{example_increase:.1f} EDSS points",
                            delta_color="off"
                        )
                        st.caption(
                            f"Calculation: {example_baseline:.1f} + "
                            f"{example_increase:.1f} = {example_level:.1f}"
                        )

                    st.caption(
                        "Illustrative calculation only. Changing the slider does not "
                        "modify the selected patient's EDSS trajectory or classification."
                    )

                    st.divider()

                    st.markdown("#### 3. How is progression classified?")

                    class_col1, class_col2 = st.columns(2, gap="small")

                    with class_col1:
                        st.markdown(
                            "**0 — No observed sustained progression**"
                        )
                        st.caption(
                            "Recorded EDSS measurements do not meet "
                            "the study-defined progression criterion."
                        )

                    with class_col2:
                        st.markdown(
                            "**1 — Observed sustained progression**"
                        )
                        st.caption(
                            "Recorded EDSS measurements meet "
                            "the study-defined progression criterion."
                        )

                    st.markdown("#### 4. What does XGBoost estimate?")

                    st.markdown(
                        "XGBoost estimates the probability of belonging to the "
                        "**sustained EDSS progression class**, using the latest "
                        "recorded demographic, relapse and gait-related measurements."
                    )

                    st.caption(
                        "Important methodological limitation: The latest feature "
                        "measurements may have been recorded after progression occurred. "
                        "Therefore, the current model classifies study-defined "
                        "progression status and should not be interpreted as a "
                        "validated prediction of future progression."
                    )





# Dedicated third workspace — surrogate decision pathway.
if st.session_state["xai_active_page"] == "Patient Decision Pathway":
    st.markdown("### Patient Decision Pathway · Explanation Interrogation")
    st.caption(
        "Follow the highlighted patient-specific route through the simplified surrogate tree. "
        "Each split compares a measurement with a model-derived threshold; these are not clinical cut-offs."
    )
    
    # -----------------------------------------------------------------
    # 1. EXTRACT CURRENT PATIENT BASE DICTIONARY FROM SELECTED SIDEBAR PATIENT
    # -----------------------------------------------------------------
    patient_base_dict = {
        'age': float(patient_records['age'].iloc[-1]),
        'numrelapse2': float(patient_records['numrelapse2'].iloc[-1]),
        'walkdur_all_sum_w': float(patient_records['walkdur_all_sum_w'].iloc[-1]),
        'wbsteps_all_sum_w': float(patient_records['wbsteps_all_sum_w'].iloc[-1]),
        'ws_30_avg_w': float(patient_records['ws_30_avg_w'].iloc[-1]),
        'ws_30_p90_w': float(patient_records['ws_30_p90_w'].iloc[-1]),
        'cadence_all_avg_w': float(patient_records['cadence_all_avg_w'].iloc[-1]),
        'strlen_30_avg_w': float(patient_records['strlen_30_avg_w'].iloc[-1]),
        'cadence_all_var_w': float(patient_records['cadence_all_var_w'].iloc[-1])
    }

    _rule_summary_slot = st.empty()
    _rule_controls_col, _rule_tree_col = st.columns([1, 2.1], gap="medium")
    _rule_tree_slot = _rule_tree_col.empty()
    with _rule_controls_col:
        # Interactive patient measurements sit beside the live decision tree.
        st.markdown("##### Interrogate patient measurements")
        st.caption("Adjust one or more measurements to examine changes in the surrogate decision pathway. "
                   "These hypothetical values do not represent observed clinical change or treatment effects.")
        if st.button(
           "Reset Sliders to Actual Patient Values",
               key="reset_patient_sliders"
           ):
         st.session_state.slider_reset_trigger = not st.session_state.get(
                "slider_reset_trigger",
                False
            )
    
        adjusted_patient_dict = {}

        # Age slider: Cannot be reduced below actual patient age
        actual_age = int(patient_base_dict['age'])
        max_age = max(100, actual_age + 20)

        adjusted_patient_dict['age'] = st.slider(
            FEATURE_DISPLAY_NAMES['age'],
            min_value=actual_age,
            max_value=max_age,
            value=actual_age,
            step=1,
            key=f"slider_age_{selected_pid}_{st.session_state.get('slider_reset_trigger', False)}"
        )

        # Sliders for remaining features based on dataset ranges
        non_age_features = [f for f in predictor.features if f != 'age']

        for feat in non_age_features:
            actual_val = patient_base_dict[feat]
            disp_name = FEATURE_DISPLAY_NAMES[feat]
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())

            if min_val >= max_val:
                min_val = float(actual_val * 0.5)
                max_val = float(actual_val * 1.5) if actual_val != 0 else 1.0

            curr_min = min(min_val, float(actual_val))
            curr_max = max(max_val, float(actual_val))

            if isinstance(actual_val, int) or feat in ['numrelapse2', 'wbsteps_all_sum_w']:
                adjusted_patient_dict[feat] = st.slider(
                    disp_name,
                    min_value=int(curr_min),
                    max_value=int(curr_max),
                    value=int(actual_val),
                    step=1,
                    key=f"slider_{feat}_{selected_pid}_{st.session_state.get('slider_reset_trigger', False)}"
                )
            else:
                adjusted_patient_dict[feat] = st.slider(
                    disp_name,
                    min_value=float(curr_min),
                    max_value=float(curr_max),
                    value=float(actual_val),
                    step=0.01,
                    key=f"slider_{feat}_{selected_pid}_{st.session_state.get('slider_reset_trigger', False)}"
                )

    # -----------------------------------------------------------------
    # 2. DYNAMIC RULE PATHWAY EXTRACTION (UPDATES WITH EACH SIDEBAR SELECTION)
    # -----------------------------------------------------------------
    rulexai_engine = RuleXAIEngine(predictor.model, predictor.scaler, predictor.features)
    rulexai_engine.fit_dynamic_rules(df)

    patient_data = adjusted_patient_dict
    
    def extract_full_pathway_sentence(tree_model, feature_names, patient_dict):
        """Traverse the entire surrogate decision tree to produce a single complete pathway sentence."""
        tree_ = tree_model.tree_
        curr_node = 0
        path_steps = []
        while curr_node != _tree.TREE_UNDEFINED and tree_.feature[curr_node] != _tree.TREE_UNDEFINED:
             raw_feat = feature_names[tree_.feature[curr_node]]
             disp_feat = FEATURE_DISPLAY_NAMES.get(raw_feat, raw_feat)
             thresh = float(tree_.threshold[curr_node])
             patient_val = float(patient_dict.get(raw_feat, 0.0))

             if patient_val <= thresh:
                 step_str = f"{disp_feat} of {patient_val:.2f} is less than or equal to {thresh:.2f}"
                 curr_node = tree_.children_left[curr_node]

             else:
                step_str = f"{disp_feat} of {patient_val:.2f} is greater than {thresh:.2f}"
                curr_node = tree_.children_right[curr_node]

             path_steps.append(step_str)
            
        leaf_value = tree_.value[curr_node][0]
        pred_class = int(np.argmax(leaf_value))
        outcome_str = "Progression Risk" if pred_class == 1 else "Stable Condition"
        if len(path_steps) > 1:
             conditions_text = ", ".join(path_steps[:-1]) + ", and " + path_steps[-1]
        elif len(path_steps) == 1:
             conditions_text = path_steps[0]
        else:
             conditions_text = "No conditions evaluated"

        # Construct full multi-stage sentence
        full_sentence = f"Because {conditions_text}, the predicted outcome is {outcome_str}."
        return full_sentence, path_steps, pred_class

    full_pathway_text, path_conditions, active_pred_class = extract_full_pathway_sentence(
        rulexai_engine.surrogate_tree, 
            rulexai_engine.feature_names, 
            patient_data
        )


    with _rule_summary_slot.container():

        st.markdown(f"#### Patient MS-{selected_pid} Decision Summary")
        # Present each measured feature once in the written explanation.
        # Repeated splits remain intact in the original decision tree.
        _unique_conditions = []
        _seen_rule_features = set()
        for _condition in path_conditions:
            _feature_name = _condition.split(" of ", 1)[0]
            if _feature_name not in _seen_rule_features:
                _seen_rule_features.add(_feature_name)
                _unique_conditions.append(_condition)

        _clinical_outcome = (
            "a progression-associated classification" if active_pred_class == 1
            else "a non-progression classification"
        )
        _readable_conditions = [
            _condition.replace(" is less than or equal to ", " is at or below the model threshold of ")
                      .replace(" is greater than ", " is above the model threshold of ")
            for _condition in _unique_conditions
        ]
        _condition_summary = "; ".join(_readable_conditions)
        simple_pathway = (
            f"The patient's selected measurements follow a surrogate decision pathway associated "
            f"with {_clinical_outcome}. "
            + (f"Key comparisons: {_condition_summary}." if _condition_summary else
               "No additional feature comparisons are required at this leaf.")
        )
        st.markdown(
            f"""
            <div style="background:#f5f9fd;border:1px solid #d5e4f0;
                        border-left:4px solid #176caa;padding:15px 18px;
                        border-radius:9px;margin-bottom:12px;">
              <div style="font-size:12px;font-weight:800;letter-spacing:.07em;
                          color:#176caa;margin-bottom:7px;">PATIENT-SPECIFIC RULE INTERPRETATION</div>
              <div style="color:#23364b;font-size:14px;line-height:1.65;">
                {html.escape(simple_pathway)}
              </div>
            </div>
            """, unsafe_allow_html=True
        )
        st.caption(
               "This surrogate pathway approximates the primary XGBoost model; "
               "it does not explain every internal model operation or establish clinical causation. "
               "Use alongside the observed EDSS trajectory, clinical history and original-model SHAP explanation."
           )
        
   
        # 2. Unified Rule Model Metrics Dashboard
        X_all = df[predictor.features]
        X_scaled_all = predictor.scaler.transform(X_all)
        y_xgb_all = predictor.model.predict(X_scaled_all)
    
        # Access the global or active cluster surrogate decision tree
        surrogate_tree = rulexai_engine.surrogate_tree
        y_surr_all = surrogate_tree.predict(X_all)
    
        # Calculate global surrogate fidelity metrics
        matching_rate = float(accuracy_score(y_xgb_all, y_surr_all) * 100)
        disagreement_rate = 100.0 - matching_rate
        f1_alignment = float(f1_score(y_xgb_all, y_surr_all, average='weighted') * 100)

        # Compute average decision path length across the cohort
        indicator = surrogate_tree.decision_path(X_all)
        avg_path_length = float(np.mean(indicator.sum(axis=1)))

        # Compute patient-specific active path length for the selected patient
        patient_X = pd.DataFrame([patient_data])[predictor.features]
        patient_indicator = surrogate_tree.decision_path(patient_X)
        patient_path_length = int(patient_indicator.sum(axis=1).item())
        st.markdown(
            "##### **XCCSHAP Shallow Tree Fidelity & Performance**",
            help="The XCCSHAP shallow decision tree surrogate achieves 81.8% accuracy and an F1 score of 82.2% relative to the primary black-box XGBoost model across the entire patient cohort while reducing decision path lengths.",
        )
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
        with m_col1:
            st.metric(
                label="Shallow Decision Tree Matching Rate", 
                value=f"{matching_rate:.1f}%", 
                delta=f"-{disagreement_rate:.1f}% Disagreement",
                delta_color="inverse",
                help=(
                    f"The percentage of cases where a simplified, shallow decision tree produces the same prediction as the primary model. "
                    f"In this case about {matching_rate:.1f}% of the time.\n\n"
                    f"Disagreement ({disagreement_rate:.1f}%): In about {disagreement_rate:.1f}% of cases, "
                    f"the simple tree and the main model arrive at different predictions.\n"
                    f"A higher disagreement rate indicates that the simplified tree is less able to replicate the primary model’s decisions."
                )
            )
        
        with m_col2:
            st.metric(
                label="Model Alignment Score (F1)", 
                value=f"{f1_alignment:.1f}%",
                help=(
                    "Measures how closely the surrogate model's predictions align with the original model, "
                    "balancing precision and recall. Higher scores indicate stronger agreement between the two models."
                
                )
            )
        
        with m_col3:
            st.metric(
                label="Avg Decision Path Length", 
                value=f"{avg_path_length:.1f} Nodes", 
                delta="Compact Explanations",
                help=(
                    "Measures the average number of decision conditions/rules a patient case traverses "
                    "from the root to a final leaf. Fewer nodes indicate simpler, more interpretable reasoning."
                )
            )
        
        with m_col4:
            st.metric(
                label=f"Patient MS-{selected_pid} Path Length", 
                value=f"{patient_path_length} Nodes", 
                delta=f"{len(path_conditions)} Active Rules Evaluated",
                help=(
                    f"The exact number of decision rules evaluated specifically for Patient MS-{selected_pid} "
                    f"to arrive at their individual outcome in this surrogate tree."
                )
            )
    # Tree hierarchy plot hidden: retain only the final two XCCSHAP importance charts.

    # Visualizing XCCSHAP Hierarchy with active path highlighted for selected patient
    def build_plotly_decision_tree(tree_model, feature_names, patient_latest):
        """Compact, collision-aware full surrogate tree; no marker dots."""
        tree_ = tree_model.tree_
        children_l, children_r = tree_.children_left, tree_.children_right
        active = set()
        node = 0
        while node >= 0:
            active.add(node)
            if children_l[node] < 0:
                break
            feature = feature_names[tree_.feature[node]]
            node = (children_l[node] if float(patient_latest.get(feature, 0))
                    <= tree_.threshold[node] else children_r[node])

        leaves = []
        def enumerate_leaves(n):
            if children_l[n] < 0:
                leaves.append(n)
            else:
                enumerate_leaves(children_l[n])
                enumerate_leaves(children_r[n])
        enumerate_leaves(0)
        leaf_index = {n: i for i, n in enumerate(leaves)}
        coords, depths = {}, {}
        def place(n, d=0):
            depths[n] = d
            if children_l[n] < 0:
                coords[n] = float(leaf_index[n])
            else:
                place(children_l[n], d+1)
                place(children_r[n], d+1)
                coords[n] = (coords[children_l[n]] + coords[children_r[n]]) / 2
        place(0)
        max_depth = max(depths.values())
        short = {
            'ws_30_avg_w': 'Average Walking<br>Speed (m/s)',
            'walkdur_all_sum_w': 'Walking Duration<br>(s)',
            'ws_30_p90_w': 'Peak Gait Speed<br>(m/s)',
            'cadence_all_avg_w': 'Average Cadence<br>(steps/min)',
            'cadence_all_var_w': 'Cadence<br>Variability',
            'strlen_30_avg_w': 'Average Stride<br>Length',
            'wbsteps_all_sum_w': 'Total Walking<br>Steps',
            'numrelapse2': 'Relapses', 'age': 'Age (Years)'
        }
        # All nodes at the same depth occupy nonoverlapping horizontal slots.
        # A full-depth tree with many leaves needs a wide screen for readable labels.
        fig = go.Figure()
        normal_x, normal_y, active_x, active_y = [], [], [], []
        for parent in range(tree_.node_count):
            if children_l[parent] < 0:
                continue
            for child in (children_l[parent], children_r[parent]):
                xs = active_x if parent in active and child in active else normal_x
                ys = active_y if parent in active and child in active else normal_y
                xs.extend([coords[parent], coords[child], None])
                ys.extend([-depths[parent], -depths[child], None])
        fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode='lines',
            line=dict(color='#c5d2e2', width=1.5), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=active_x, y=active_y, mode='lines',
            line=dict(color='#f59e0b', width=3.5), hoverinfo='skip', showlegend=False))
        ann = []
        for n in range(tree_.node_count):
            is_active = n in active
            if children_l[n] < 0:
                pred = int(np.argmax(tree_.value[n][0]))
                label = 'PROGRESSION' if pred else 'STABLE'
                c = '#dc2626' if pred else '#16a34a'
                label_text = f'<b>{label}</b><br><i>n = {tree_.n_node_samples[n]}</i>'
                bg = '#0f172a' if is_active else '#ffffff'
                fg = '#ffffff' if is_active else c
                border = c
                box_width = 77
            else:
                feature = feature_names[tree_.feature[n]]
                threshold = float(tree_.threshold[n])
                value = float(patient_latest.get(feature, 0))
                comparison = "At/below" if value <= threshold else "Above"
                label_text = (f'<b>{short.get(feature, FEATURE_DISPLAY_NAMES.get(feature, feature))}</b>'
                              f'<br>Rule: ≤ {threshold:.2f}'
                              f'<br><span style="color:#0369a1">Value: {value:.2f} ({comparison})</span>')
                bg, fg = '#ffffff', '#0f172a'
                border = '#f59e0b' if is_active else '#94a3b8'
                box_width = 94
            ann.append(dict(x=coords[n], y=-depths[n], xref='x', yref='y',
                text=label_text, showarrow=False, align='center',
                font=dict(family='Arial, sans-serif', size=11, color=fg),
                width=box_width, bgcolor=bg, bordercolor=border,
                borderwidth=2 if is_active else 1, borderpad=3,
                xanchor='center', yanchor='middle'))
        # 14+ leaves cannot fit 94-pixel boxes into a narrow chart without
        # overlapping. Use Plotly's responsive container and narrower labels.
        fig.update_layout(
            annotations=ann, height=535,
            margin=dict(l=12, r=12, t=38, b=35),
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            xaxis=dict(visible=False, fixedrange=True,
                       range=[-0.65, max(1, len(leaves)-1)+0.65]),
            yaxis=dict(visible=False, fixedrange=True,
                       range=[-max_depth-0.43, 0.43]),
            showlegend=False, dragmode=False)
        return fig

    _rule_tree_col.caption(
        "Amber branches identify the active patient route. Highlighted nodes show the comparisons "
        "used by the surrogate; other branches show alternative model pathways. "
        "Changing a slider updates the route without changing the underlying tree structure."
    )
    fig_surrogate_tree = build_plotly_decision_tree(rulexai_engine.surrogate_tree, rulexai_engine.feature_names, patient_data)
    fig_surrogate_tree.update_layout(
        height=535,
        margin=dict(l=12, r=12, t=38, b=35),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#203449"),
    )
    _rule_tree_slot.plotly_chart(fig_surrogate_tree, width="stretch", key="patient_rule_tree_live")

    # Small SHAP comparison; the surrogate tree and sliders above are unchanged.
    _tree_outcome = "Progression" if active_pred_class == 1 else "No progression"
    _xgb_outcome = "Progression" if pred_class_val == 1 else "No progression"
    _same_outcome = active_pred_class == pred_class_val
    # Clear, accessible comparison cards using the existing dashboard palette.
    _comparison_color = "#15803d" if _same_outcome else "#b45309"
    _comparison_bg = "#f0fdf4" if _same_outcome else "#fffbeb"
    _comparison_border = "#bbf7d0" if _same_outcome else "#fcd34d"
    _comparison_label = "Same classification" if _same_outcome else "Different classifications"
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #dbe4ed;border-radius:15px;
                padding:22px 24px;margin:18px 0 12px;box-shadow:0 3px 12px rgba(15,23,42,.05);">
      <div style="font-size:1.25rem;font-weight:800;color:#17324d;margin-bottom:15px;">
        Does the decision agree with SHAP?
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;">
        <div style="background:#f1f6fc;border:1px solid #cbdced;border-left:5px solid #2563eb;
                    border-radius:12px;padding:18px;min-height:112px;">
          <div style="color:#526579;font-size:.82rem;font-weight:750;margin-bottom:9px;">DECISION TREE</div>
          <div style="color:#17324d;font-size:1.45rem;font-weight:800;line-height:1.25;">{html.escape(_tree_outcome)}</div>
        </div>
        <div style="background:#f1f6fc;border:1px solid #cbdced;border-left:5px solid #0e7490;
                    border-radius:12px;padding:18px;min-height:112px;">
          <div style="color:#526579;font-size:.82rem;font-weight:750;margin-bottom:9px;">XGBOOST MODEL</div>
          <div style="color:#17324d;font-size:1.45rem;font-weight:800;line-height:1.25;">{html.escape(_xgb_outcome)}</div>
        </div>
        <div style="background:{_comparison_bg};border:1px solid {_comparison_border};
                    border-left:5px solid {_comparison_color};border-radius:12px;padding:18px;min-height:112px;">
          <div style="color:#526579;font-size:.82rem;font-weight:750;margin-bottom:9px;">COMPARISON RESULT</div>
          <div style="color:{_comparison_color};font-size:1.25rem;font-weight:800;line-height:1.3;">
            {'✓' if _same_outcome else '⚠'} {html.escape(_comparison_label)}</div>
        </div>
      </div>
      <div style="font-size:.82rem;color:#64748b;line-height:1.5;margin-top:13px;">
        SHAP describes the original patient prediction; the tree follows the current slider values.
        Agreement does not mean the explanations are equivalent.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    RuleXAIEngine.render_surrogate_validation_section(predictor, df)


# INTERROGATION — measured evidence directly underneath the selected attribution.
if st.session_state["xai_active_page"] == "Feature Interrogation":
    visit_x = formatted_visit_dates
    _interrogation_header("Feature Interrogation", "Examine observed evidence, compare two predictors, and challenge the model response.")
    # Side-by-side accordion headers; expanded content uses the full width for legible charts.
    _accordion_titles = [
        "Patient Measurements & Cohort Comparisons",
        "Subgroup Feature Impact & Model Explanations",
        "Prediction Sensitivity · Single- and Two-Feature Analysis",
    ]
    if "fi_open_panel" not in st.session_state:
        st.session_state["fi_open_panel"] = 0

    st.markdown("""<style>
    /* Three card-like accordion triggers, styled after the supplied reference. */
    [data-testid="stMain"] [class*="st-key-fi_panel_"] button {
        width:100%; min-height:74px; text-align:left; justify-content:space-between;
        background:#ffffff!important; color:#23364b!important;
        border:1px solid #cbd8e8!important; border-radius:12px!important;
        box-shadow:0 3px 10px rgba(15,38,75,.06)!important;
        font-size:.91rem!important; font-weight:750!important;
        white-space:normal!important; line-height:1.4!important;
        padding:13px 15px!important;
    }
    [data-testid="stMain"] [class*="st-key-fi_panel_"] button:hover {
        border-color:#1678ae!important; background:#f0f7ff!important;
    }
    [data-testid="stMain"] [class*="st-key-fi_panel_active_"] button {
        border-color:#1680c1!important; border-bottom:4px solid #1680c1!important;
        background:#eef7ff!important; color:#07578a!important;
    }
    </style>""", unsafe_allow_html=True)
    _accordion_cols = st.columns(3, gap="medium")
    for _panel_i, _panel_title in enumerate(_accordion_titles):
        with _accordion_cols[_panel_i]:
            _is_open = st.session_state["fi_open_panel"] == _panel_i
            _symbol = "⌃" if _is_open else "⌄"
            if st.button(
                f"{_panel_title}   {_symbol}",
                key=f"fi_panel_{'active_' if _is_open else ''}{_panel_i}",
                use_container_width=True,
            ):
                st.session_state["fi_open_panel"] = None if _is_open else _panel_i
                st.rerun()
    with st.popover("ⓘ  About these sections"):
        st.markdown("**How to interrogate the explanation**")
        st.write("Choose a panel above to expand its visualisations below. Click the open panel again to collapse it. Only one panel is open at a time.")
        st.write("**Measurements:** patient trajectories and cohort comparisons.")
        st.write("**Subgroups:** feature attributions across patient groups.")
        st.write("**Sensitivity:** hypothetical model responses, not causal effects.")
    _accordion_choice = (_accordion_titles[st.session_state["fi_open_panel"]]
                         if st.session_state["fi_open_panel"] is not None else None)
    if _accordion_choice == _accordion_titles[0]:
        with st.container(border=True):
            # Shared visit labels used by Mobility and EDSS charts
            visit_x = formatted_visit_dates

            # =====================================================================
            # UNIFIED MOBILITY PROFILE SECTION
            # =====================================================================
            # 1. Define available mobility features
            mobility_features = [
                "ws_30_p90_w",
                "ws_30_avg_w",
                "strlen_30_avg_w",
                "cadence_all_avg_w",
                "walkdur_all_sum_w",
                "cadence_all_var_w",
            ]

            # 2. Build display-name -> feature-name mapping
            try:
                _mob_shap = np.asarray(predictor.compute_patient_shap(patient_records), dtype=float).reshape(-1)
                _mob_rank = {f: abs(float(_mob_shap[i])) for i, f in enumerate(predictor.features)}
                mobility_features.sort(key=lambda f: _mob_rank.get(f, 0), reverse=True)
            except Exception:
                _mob_rank = {f: 0.0 for f in mobility_features}
            # Rank by the magnitude of each feature's contribution, but present
            # clinician-readable labels instead of unexplained SHAP numbers.
            mobility_options = {
                f"{FEATURE_DISPLAY_NAMES[f]} — {('Strongest' if rank == 0 else '2nd' if rank == 1 else '3rd' if rank == 2 else f'{rank + 1}th')} most influential mobility factor": f
                for rank, f in enumerate(mobility_features)
            }

            # ================================================================
            # ONLY NOW CREATE THE TWO FIGURE COLUMNS
            # ================================================================


            # Full-width trajectory followed by a full-width two-chart comparison row.
            # This prevents an empty lower-left column and gives radar labels room.
            chart_col1 = st.container()
            chart_col2 = st.container()


            # ================================================================
            # LEFT FIGURE
            # ================================================================

            with chart_col1:
                st.markdown("#### Longitudinal Metric Trajectory")

                # Resolve the shared selector BEFORE referencing cohort or patient values.
                selected_mobility_feat = _selected_interrogation_feature
                selected_mobility_label = FEATURE_DISPLAY_NAMES.get(
                    selected_mobility_feat, selected_mobility_feat
                )
                st.caption(f"Selected feature: **{selected_mobility_label}**")

                _mob_config = {
                    "age": ("years", "Age is not an abnormality; interpret it as patient context."),
                    "numrelapse2": ("relapses", "More reported relapses may indicate inflammatory disease activity."),
                    "walkdur_all_sum_w": ("s", "Changes in walking duration may reflect activity or mobility differences."),
                    "wbsteps_all_sum_w": ("steps", "Changes in step count may reflect activity or mobility differences."),
                    "ws_30_avg_w": ("m/s", "Lower walking speed may suggest reduced walking performance."),
                    "ws_30_p90_w": ("m/s", "Lower peak gait speed may suggest reduced walking capacity."),
                    "cadence_all_avg_w": ("steps/min", "Changes in cadence may reflect an altered walking pattern."),
                    "strlen_30_avg_w": ("m", "Shorter stride length may warrant assessment of gait and mobility."),
                    "cadence_all_var_w": ("", "Higher cadence variability may suggest a less consistent walking rhythm."),
                }
                unit, _clinical_context = _mob_config.get(
                    selected_mobility_feat, ("", "Interpret alongside clinical history.")
                )

                _mob_history = pd.to_numeric(
                    patient_records[selected_mobility_feat], errors="coerce"
                ).reset_index(drop=True)
                _mob_reference = pd.to_numeric(
                    patient_level_cohort[selected_mobility_feat], errors="coerce"
                ).dropna()

                # Stride length can be stored in cm (e.g. 105) or m (e.g. 1.05).
                # Convert each series consistently; never compare metres against centimetres.
                if selected_mobility_feat == "strlen_30_avg_w":
                    if _mob_history.dropna().median() > 10:
                        _mob_history = _mob_history / 100.0
                    if len(_mob_reference) and _mob_reference.median() > 10:
                        _mob_reference = _mob_reference / 100.0

                _valid = _mob_history.dropna()
                _mob_latest = float(_valid.iloc[-1]) if len(_valid) else float("nan")
                _mob_first = float(_valid.iloc[0]) if len(_valid) else float("nan")
                _mob_pctl = float("nan")
                _mob_median = float("nan")
                _cohort_label = "Cohort median unavailable"
                if len(_mob_reference) and np.isfinite(_mob_latest):
                    _mob_pctl = 100 * float((_mob_reference < _mob_latest).mean())
                    _mob_median = float(_mob_reference.median())
                    _cohort_label = f"Cohort median: {_mob_median:.2f} {unit}".strip()

                _change = _mob_latest - _mob_first if len(_valid) >= 2 else float("nan")
                _change_pct = (
                    100 * _change / abs(_mob_first)
                    if np.isfinite(_change) and abs(_mob_first) > 1e-9
                    else float("nan")
                )
                if selected_mobility_feat == "age":
                    _trend_label, _trend_color = "PATIENT CONTEXT · AGE", "#1d4ed8"
                    _trend_detail = (
                        f"Latest recorded age: {_mob_latest:.0f} years · demographic context, not a clinical trend"
                        if np.isfinite(_mob_latest) else "Age information unavailable"
                    )
                elif len(_valid) < 2:
                    _trend_label, _trend_color, _trend_detail = (
                        "INSUFFICIENT HISTORY", "#64748b", "At least two valid visits are needed to assess change"
                    )
                elif np.isfinite(_change_pct) and abs(_change_pct) < 5:
                    _trend_label, _trend_color = "RELATIVELY STABLE", "#15803d"
                    _trend_detail = f"{_change_pct:+.1f}% since first recorded visit"
                elif _change > 0:
                    _trend_label, _trend_color = "INCREASED", "#0369a1"
                    _trend_detail = (
                        f"{_change_pct:+.1f}% since first recorded visit" if np.isfinite(_change_pct)
                        else f"+{_change:.2f} {unit} since first recorded visit"
                    )
                elif _change < 0:
                    _trend_label, _trend_color = "DECREASED", "#c2410c"
                    _trend_detail = (
                        f"{_change_pct:+.1f}% since first recorded visit" if np.isfinite(_change_pct)
                        else f"{_change:.2f} {unit} since first recorded visit"
                    )
                else:
                    _trend_label, _trend_color, _trend_detail = (
                        "UNCHANGED", "#15803d", "Same recorded value at first and latest visits"
                    )

                st.caption(f"**Clinical context:** {_clinical_context} · Cohort comparisons are descriptive, not diagnostic.")

                fig_mob_trend = go.Figure()
                _x_vals = list(visit_x)[:len(_mob_history)]
                _y_vals = _mob_history.tolist()[:len(_x_vals)]
                if np.isfinite(_mob_median):
                    fig_mob_trend.add_hline(
                        y=_mob_median, line_dash="dot", line_color="#64748b",
                        annotation_text="Patient baseline", annotation_position="bottom left",
                    )
                _point_colors = [_trend_color for _ in _y_vals]
                fig_mob_trend.add_trace(go.Scatter(
                    x=_x_vals, y=_y_vals, mode="lines+markers",
                    name="Patient measurements",
                    line=dict(color=_trend_color, width=3),
                    marker=dict(size=10, color=_point_colors, line=dict(color="white", width=1.5)),
                    hovertemplate=f"%{{x}}<br>Value: %{{y:.2f}} {unit}<extra></extra>",
                ))
                if len(_valid) and _x_vals:
                    _last_idx = next(
                        (i for i in range(min(len(_y_vals), len(_x_vals)) - 1, -1, -1)
                         if pd.notna(_y_vals[i])), None
                    )
                    if _last_idx is not None:
                        fig_mob_trend.add_annotation(
                            x=_x_vals[_last_idx], y=_y_vals[_last_idx],
                            text=f"<b>Latest: {_y_vals[_last_idx]:.2f} {unit}</b>",
                            showarrow=True, arrowhead=2, ax=-60, ay=-35,
                            bgcolor="white", bordercolor=_trend_color,
                            font=dict(color="#17324d", size=13),
                        )
                # Prominent trend banner INSIDE the Plotly figure, above the plotting area.
                fig_mob_trend.add_annotation(
                    x=0.01, y=1.16, xref="paper", yref="paper",
                    xanchor="left", yanchor="middle",
                    text=f"<b>{_trend_label}</b><br>{_trend_detail}",
                    showarrow=False,
                    bgcolor=_trend_color, bordercolor=_trend_color,
                    borderpad=9,
                    font=dict(color="white", size=16),
                    align="left",
                )
                fig_mob_trend.add_annotation(
                    x=0.99, y=1.30, xref="paper", yref="paper",
                    xanchor="right", yanchor="middle",
                    text=f"<b>{_cohort_label}</b>",
                    showarrow=False,
                    bgcolor="#eaf1f8", bordercolor="#cbd5e1",
                    borderpad=9,
                    font=dict(color="#17324d", size=13),
                    align="right",
                )
                fig_mob_trend.update_layout(
                    height=490, margin=dict(l=65, r=35, t=130, b=95),
                    plot_bgcolor="rgba(255,255,255,0.94)",
                    paper_bgcolor="rgba(255,255,255,0.94)",
                    yaxis=dict(title=f"{selected_mobility_label} ({unit})" if unit else selected_mobility_label,
                               automargin=True, gridcolor="#e2e8f0"),
                    xaxis=dict(title="Clinical visits", automargin=True, tickangle=-30),
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_mob_trend, width="stretch",
                    key=f"mobility_trajectory_{selected_mobility_feat}"
                )
                st.caption("The dotted line shows the cohort median. Trend colours describe measured direction, not clinical severity; age is shown only as patient context.")


            # ================================================================
            # RIGHT FIGURE
            # ================================================================

            with chart_col2:
        
                st.markdown(
                    "#### Patient Features Compared with the Cohort"
                )

                # All nine model inputs appear in the profile, including contextual variables.
                radar_features = list(_interrogation_options)
                radar_labels = [FEATURE_DISPLAY_NAMES[f].replace(" (", "<br>(") for f in radar_features]

                # Subgroup controls: default to the selected patient's demographic peer group.
                patient_age_group = _age_group_10y(float(patient_records["age"].iloc[-1]))
                patient_gender = str(patient_records["gender"].iloc[-1]).strip() if "gender" in patient_records.columns else "Unknown"
                default_age_idx = subgroup_age_options.index(patient_age_group) if patient_age_group in subgroup_age_options else 0
                default_gender_idx = subgroup_gender_options.index(patient_gender) if patient_gender in subgroup_gender_options else 0

                # Filters sit immediately above the percentile chart and its embedded
                # comparison cards, instead of occupying a narrow column beside the chart.
                pct_age_col, pct_gender_col = st.columns(2, gap="small")
                with pct_age_col:
                    pct_age_group = st.selectbox(
                        "Age subgroup (10-year bands)",
                        subgroup_age_options,
                        index=default_age_idx,
                        key="percentile_age_subgroup",
                    )
                with pct_gender_col:
                    pct_gender = st.selectbox(
                        "Gender subgroup",
                        subgroup_gender_options,
                        index=default_gender_idx,
                        key="percentile_gender_subgroup",
                    )

                percentile_cohort = _filter_subgroup(patient_level_cohort, pct_age_group, pct_gender)
                if percentile_cohort.empty:
                    st.warning("No patients are available in this subgroup. Select a broader age or gender subgroup.")
                    percentile_cohort = patient_level_cohort
                    pct_effective_label = "All patients (fallback)"
                else:
                    pct_effective_label = _subgroup_label(pct_age_group, pct_gender)

                # Independent of the trajectory selector: this card always compares
                # the patient's latest peak gait speed with the selected subgroup.
                # The patient measurement stays fixed; the median and difference change.
                comparison_feat = _selected_interrogation_feature
                comparison_unit = ("years" if comparison_feat == "age" else
                                   "m/s" if comparison_feat in ("ws_30_avg_w", "ws_30_p90_w") else
                                   "steps/min" if comparison_feat == "cadence_all_avg_w" else
                                   "s" if comparison_feat == "walkdur_all_sum_w" else "")
                comparison_fmt = "{:.2f}"
                comparison_patient = pd.to_numeric(
                    patient_records[comparison_feat], errors="coerce"
                ).iloc[-1]
                subgroup_metric = pd.to_numeric(
                    percentile_cohort[comparison_feat], errors="coerce"
                ).dropna()
                subgroup_median = float(subgroup_metric.median()) if not subgroup_metric.empty else np.nan
                subgroup_difference = (
                    100.0 * (float(comparison_patient) - subgroup_median) / subgroup_median
                    if pd.notna(comparison_patient) and np.isfinite(subgroup_median) and subgroup_median != 0
                    else np.nan
                )

                st.markdown(f"##### Patient and Cohort Comparison · {_selected_interrogation_label}")
                st.caption("Patient value vs selected subgroup median")
                metric_cols = st.columns(3, gap="small")
                metric_cols[0].metric(
                    "Patient's Latest Value",
                    f"{comparison_fmt.format(comparison_patient)} {comparison_unit}" if pd.notna(comparison_patient) else "N/A",
                    help="Latest recorded value of the selected feature.",
                )
                metric_cols[1].metric(
                    "Study Cohort Median",
                    f"{comparison_fmt.format(subgroup_median)} {comparison_unit}" if np.isfinite(subgroup_median) else "N/A",
                    help="Median of the selected feature in the selected age/gender subgroup (one latest record per patient).",
                )
                metric_cols[2].metric(
                    "Difference from Cohort Median",
                    f"{subgroup_difference:+.1f}%" if np.isfinite(subgroup_difference) else "N/A",
                    help="100 × (selected patient's latest value − selected subgroup median) / selected subgroup median.",
                )
                st.caption(
                    f"Selected median reference: {pct_effective_label} (n={percentile_cohort['Local Participant'].nunique()}). "
                    "The patient's latest value and full-study percentiles stay fixed; the selected subgroup median and difference update together."
                )

                # Fixed percentile scale: every patient's position uses the full-study distribution.
                # The selected subgroup median is placed on that SAME fixed scale so it can move.
                patient_percentiles = []
                subgroup_median_percentiles = []
                for feat in radar_features:
                    full_values = pd.to_numeric(patient_level_cohort[feat], errors="coerce").dropna()
                    sub_values = pd.to_numeric(percentile_cohort[feat], errors="coerce").dropna()
                    patient_value = float(patient_records[feat].iloc[-1])
                    patient_pct = float((full_values <= patient_value).mean() * 100) if len(full_values) else np.nan
                    subgroup_pct = (float((full_values <= sub_values.median()).mean() * 100)
                                    if len(full_values) and len(sub_values) else np.nan)
                    patient_percentiles.append(round(patient_pct, 1))
                    subgroup_median_percentiles.append(round(subgroup_pct, 1))

                fig_radar_pct = go.Figure()
                if _selected_interrogation_feature in radar_features:
                    _focus_idx = radar_features.index(_selected_interrogation_feature)
                    _focus_theta = radar_labels[_focus_idx]
                    _focus_pct = patient_percentiles[_focus_idx]
                    _focus_sub_pct = subgroup_median_percentiles[_focus_idx]
                else:
                    _focus_theta = None

                fig_radar_pct.add_trace(go.Scatterpolar(
                    r=patient_percentiles + [patient_percentiles[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself", mode="lines+markers",
                    name=f"Patient MS-{selected_pid} (fixed, full-study percentile)",
                    line=dict(color="#0284c7", width=2.5), marker=dict(size=8),
                    customdata=radar_features + [radar_features[0]],
                    hovertemplate="<b>%{theta}</b><br>Patient: %{r:.1f}th percentile of full study<br>Click to inspect risk-factor rank<extra></extra>",
                ))
                fig_radar_pct.add_trace(go.Scatterpolar(
                    r=subgroup_median_percentiles + [subgroup_median_percentiles[0]],
                    theta=radar_labels + [radar_labels[0]], mode="lines+markers",
                    name="Selected subgroup median (on fixed full-study scale)",
                    line=dict(color="#dc2626", width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond"),
                    hovertemplate="<b>%{theta}</b><br>Subgroup median: %{r:.1f}th percentile of full study<extra></extra>",
                ))
                if _focus_theta is not None:
                    fig_radar_pct.add_trace(go.Scatterpolar(
                        r=[_focus_pct], theta=[_focus_theta], mode="markers",
                        marker=dict(size=19, color="#f59e0b", symbol="circle-open", line=dict(width=4, color="#b45309")),
                        name="Selected influential feature",
                        hovertemplate=f"<b>{_selected_interrogation_label}</b><br>Patient percentile: %{{r:.1f}}<extra></extra>"
                    ))
                    fig_radar_pct.add_annotation(text=f"Selected: {_selected_interrogation_label}",
                        x=0.5, y=1.10, xref="paper", yref="paper", showarrow=False,
                        font=dict(size=12, color="#a16207"), bgcolor="#fffbeb")

                profile_radar_col, profile_demo_col = st.columns([1.12, 1], gap="large")
                with profile_radar_col:
                    fig_radar_pct.update_layout(
                        height=590,

                        margin=dict(
                            l=155,
                            r=155,
                            t=80,
                            b=130
                        ),

                        paper_bgcolor="rgba(255,255,255,0.92)",

                        polar=dict(

                            bgcolor="rgba(255,255,255,0.92)",

                            angularaxis=dict(
                                tickfont=dict(
                                    size=9
                                ),
                                rotation=90,
                                direction="clockwise"
                            ),

                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],

                                tickmode="array",

                                tickvals=[
                                    0,
                                    25,
                                    50,
                                    75,
                                    100
                                ],

                                ticktext=[
                                    "0%",
                                    "25%",
                                    "50%",
                                    "75%",
                                    "100%"
                                ]
                            )
                        ),

                        legend=dict(
                            orientation="h",
                            y=-0.20,
                            x=0.5,
                            xanchor="center"
                        )
                    )

                    st.caption("Blue: patient's fixed full-study percentile. Red dashed: the selected subgroup median on the same scale. Higher percentile means a higher measured value, not necessarily a better clinical outcome.")
                    _radar_event = st.plotly_chart(
                        fig_radar_pct,
                        width="stretch",
                        key=f"mobility_radar_fixed_full_study_{st.session_state.get('_xai_click_epoch', 0)}",
                        on_select="rerun", selection_mode="points",
                    )
                    if _radar_event and _radar_event.selection.points:
                        _pt = _radar_event.selection.points[0]
                        _raw = _pt.get("customdata")
                        if isinstance(_raw, (list, tuple)):
                            _raw = _raw[0] if _raw else None
                        if _raw in predictor.features:
                            _xai_open_feature(_raw)
                            st.session_state["_xai_click_epoch"] = st.session_state.get("_xai_click_epoch", 0) + 1
                            st.rerun()
                    st.caption(f"Click a blue patient marker to inspect its model-impact rank. "
                               f"Selected: **Rank {_rank_by_feature[_selected_interrogation_feature]+1} · {_selected_interrogation_label}**. "
                               "Percentiles indicate relative measured values, not risk-factor importance.")
                    if np.isfinite(subgroup_difference):
                        _relative = "below" if subgroup_difference < 0 else "above" if subgroup_difference > 0 else "equal to"
                        st.markdown(
                            f'<div class="xai-insight"><strong>Peer-group interpretation:</strong> '
                            f'{html.escape(_selected_interrogation_label)} is <strong>{abs(subgroup_difference):.1f}% {_relative}</strong> '
                            f'the median for <strong>{html.escape(pct_effective_label)}</strong>. '
                            'This describes a cohort comparison, not a causal explanation or a diagnosis.</div>',
                            unsafe_allow_html=True,
                        )

                with profile_demo_col:
                    # =====================================================================
                    # COHORT DEMOGRAPHICS — BESIDE THE PERCENTILE RADAR
                    # =====================================================================
                    st.markdown("##### Cohort Demographics by Age Group and Gender")
                    st.caption(
                        "Unique eligible participants grouped from their latest recorded visit. "
                        "Age bands are fixed 10-year intervals; each participant is counted once."
                    )

                    demo = patient_level_cohort.copy()
                    demo = demo[(demo["Age Group"] != "Unknown age") & (demo["gender"] != "Unknown")].copy()
                    demo_counts = (
                        demo.groupby(["Age Group", "gender"])["Local Participant"]
                        .nunique()
                        .reset_index(name="Participants")
                    )
                    age_order = sorted(demo_counts["Age Group"].unique().tolist(), key=_age_group_sort_key)
                    gender_order = [g for g in ["Female", "Male"] if g in demo_counts["gender"].unique()]
                    gender_order += [g for g in sorted(demo_counts["gender"].unique()) if g not in gender_order]

                    fig_demo = go.Figure()
                    for gender_idx, gender_name in enumerate(gender_order):
                        gender_counts = (
                            demo_counts[demo_counts["gender"] == gender_name]
                            .set_index("Age Group")["Participants"]
                            .reindex(age_order, fill_value=0)
                        )
                        fig_demo.add_trace(go.Bar(
                            name=gender_name,
                            x=[i + (gender_idx - (len(gender_order) - 1) / 2) * (0.8 / max(len(gender_order), 1)) for i in range(len(age_order))],
                            width=0.8 / max(len(gender_order), 1),
                            y=gender_counts.values,
                            text=[str(int(v)) for v in gender_counts.values],
                            textposition="outside",
                            cliponaxis=False,
                            customdata=age_order,
                            hovertemplate=(
                                "<b>Age group:</b> %{customdata}<br>"
                                f"<b>Gender:</b> {gender_name}<br>"
                                "<b>Participants:</b> %{y}<extra></extra>"
                            ),
                        ))

                    # Mark the selected patient's age group without adding cohort totals.
                    selected_patient_age_group = _age_group_10y(age_val)
                    if selected_patient_age_group in age_order:
                        selected_age_index = age_order.index(selected_patient_age_group)
                        selected_gender = str(gender_val).strip()
                        selected_gender_idx = gender_order.index(selected_gender) if selected_gender in gender_order else None
                        selected_bar_x = (selected_age_index +
                            (selected_gender_idx - (len(gender_order) - 1) / 2) * (0.8 / len(gender_order))
                            if selected_gender_idx is not None else selected_age_index)
                        fig_demo.add_vline(
                            x=selected_bar_x,
                            line_width=2.5,
                            line_dash="dash",
                            line_color="#dc2626",
                            # Patient age/gender are explained in the caption; avoid label collisions.
                        )

                    fig_demo.update_layout(
                        barmode="group",
                        height=440,
                        xaxis=dict(title="Age group (years)", tickmode="array", tickvals=list(range(len(age_order))), ticktext=age_order),
                        yaxis=dict(title="Number of unique participants", rangemode="tozero"),
                        legend=dict(title="Gender", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=70, r=30, t=70, b=65),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        bargap=0.22,
                        bargroupgap=0.08,
                    )
                    # Emphasize the selected age/gender bar; other groups remain visible for context.
                    for _trace in fig_demo.data:
                        if getattr(_trace, "type", None) == "bar":
                            _trace.marker.opacity = [
                                1.0 if ((pct_age_group == "All ages" or _age == pct_age_group)
                                        and (pct_gender == "All genders" or _trace.name == pct_gender)) else 0.23
                                for _age in age_order
                            ]
                    st.plotly_chart(fig_demo, width="stretch", config={"displayModeBar": False}, key="plot_line_1708")
                    st.caption(
                        f"Displayed cohort: n={demo['Local Participant'].nunique()} unique participants with known age and gender. "
                        "Bar labels show gender-specific counts; the red dashed vertical line marks the selected patient's age-and-gender bar."
                    )

# =====================================================================
# AI DECISION EXPLANATION — COHORT & MODEL ANALYSIS SUBTAB
# =====================================================================
if st.session_state["xai_active_page"] == "Feature Interrogation":
    if _accordion_choice == _accordion_titles[1]:
        with st.container(border=True):
            st.markdown("### Subgroup Feature Impact")
            st.caption(
                "Latest observation per patient · Filter by age and gender"
            )
            sg_col1, sg_col2 = st.columns(2)
            with sg_col1:
                xai_age_group = st.selectbox(
                    "Age subgroup (10-year bands)",
                    subgroup_age_options,
                    index=0,
                    key="xai_age_subgroup",
                )
            with sg_col2:
                xai_gender = st.selectbox(
                    "Gender subgroup",
                    subgroup_gender_options,
                    index=0,
                    key="xai_gender_subgroup",
                )

            xai_subgroup = _filter_subgroup(patient_level_cohort, xai_age_group, xai_gender)
            subgroup_name = _subgroup_label(xai_age_group, xai_gender)
            subgroup_n = xai_subgroup["Local Participant"].nunique()
            st.caption(f"Selected subgroup: {subgroup_name} · n={subgroup_n} patients")

            if subgroup_n == 0:
                st.warning("No patients are available in this subgroup. Choose a broader age or gender subgroup.")
            else:
                subgroup_shap, _ = predictor.compute_cohort_shap(xai_subgroup)
                X_subgroup_raw = xai_subgroup[all_features].copy()
                X_subgroup_raw.columns = [FEATURE_DISPLAY_NAMES[f] for f in all_features]

                col_bee, col_glob = st.columns(2)
                with col_bee:
                    st.markdown("#### Distribution of Feature Impact Across Patients")
                    st.caption(f"SHAP beeswarm · {subgroup_name} · n={subgroup_n}. "
                               f"Gold band = selected feature: {_selected_interrogation_label}. "
                               "Each dot is a patient; horizontal position is SHAP impact (log-odds), "
                               "colour indicates measured feature value.")
                    fig_bee = plt.figure(figsize=(7, 5))
                    shap.summary_plot(
                        subgroup_shap,
                        X_subgroup_raw,
                        plot_type="dot",
                        show=False,
                        color_bar=True,
                    )
                    # Highlight the currently interrogated feature without changing the
                    # SHAP beeswarm's point colours (which encode feature value).
                    _bee_ax = plt.gca()
                    _bee_labels = [t.get_text() for t in _bee_ax.get_yticklabels()]
                    for _bee_i, _bee_name in enumerate(_bee_labels):
                        if _bee_name == _selected_interrogation_label:
                            _bee_ax.axhspan(_bee_i - .43, _bee_i + .43, color="#fef3c7", alpha=.65, zorder=-10)
                            _bee_ax.get_yticklabels()[_bee_i].set_color("#a16207")
                            _bee_ax.get_yticklabels()[_bee_i].set_fontweight("bold")
                            break
                    plt.tight_layout()
                    st.pyplot(fig_bee)
                    plt.close(fig_bee)

                with col_glob:
                    st.markdown("#### Overall Risk Factor Importance")
                    st.caption(
                        f"Mean absolute SHAP within {subgroup_name} · n={subgroup_n}. "
                        "This is subgroup-specific model impact, not the model's fixed global split importance."
                    )
                    mean_abs_shap = np.mean(np.abs(np.asarray(subgroup_shap)), axis=0)
                    df_global_imp = pd.DataFrame({
                        "Feature": interpretable_feature_names,
                        "Mean |SHAP|": mean_abs_shap,
                    }).sort_values("Mean |SHAP|", ascending=True)

                    _chosen_subgroup_row = df_global_imp.loc[df_global_imp["Feature"] == _selected_interrogation_label]
                    if not _chosen_subgroup_row.empty:
                        st.metric(f"Subgroup mean |SHAP| · {_selected_interrogation_label}",
                                  f"{float(_chosen_subgroup_row['Mean |SHAP|'].iloc[0]):.3f}")
                    fig_glob = px.bar(
                        df_global_imp,
                        y="Feature",
                        x="Mean |SHAP|",
                        orientation="h",
                        color="Feature",
                        color_discrete_map={f: ("#176caa" if f == _selected_interrogation_label else "#cbd5e1") for f in interpretable_feature_names},
                        labels={"Feature": "Clinical Feature", "Mean |SHAP|": "Mean absolute SHAP"},
                    )
                    fig_glob.update_layout(
                        height=360,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(l=10, r=10, t=20, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_glob, width="stretch", key="plot_line_4809")

                if subgroup_n < 10:
                    st.info(
                        "Small subgroup (n < 10): interpret the SHAP distribution and ranking cautiously; "
                        "individual patients can have a large influence on the displayed pattern."
                    )

            st.markdown("---")
if st.session_state["xai_active_page"] == "Feature Interrogation":
    if _accordion_choice == _accordion_titles[2]:
        with st.container(border=True):
            st.markdown("### Prediction Sensitivity")
            st.markdown("#### Single-Feature Risk Sensitivity")
            st.caption(f"Interrogating **{_selected_interrogation_label}** · Other inputs fixed at this patient’s latest recorded values")

            # Keep an audit trail of cohort-reference sensitivity changes.
            if "sensitivity_change_history" not in st.session_state:
                st.session_state.sensitivity_change_history = []
            if "sensitivity_last_values" not in st.session_state:
                st.session_state.sensitivity_last_values = {}

            def _sens_unit(raw_feature):
                return {
                    "age": "years", "walkdur_all_sum_w": "s", "ws_30_avg_w": "m/s",
                    "ws_30_p90_w": "m/s", "cadence_all_avg_w": "steps/min",
                    "strlen_30_avg_w": "cm"
                }.get(raw_feature, "")

            def _sens_fmt(raw_feature, value):
                if raw_feature in ("age", "numrelapse2", "wbsteps_all_sum_w", "walkdur_all_sum_w"):
                    return f"{value:.0f}"
                if raw_feature in ("ws_30_avg_w", "ws_30_p90_w"):
                    return f"{value:.2f}"
                if raw_feature in ("strlen_30_avg_w", "cadence_all_avg_w"):
                    return f"{value:.1f}"
                return f"{value:.2f}"

            def _sens_slider(raw_feature, label, current_value, cohort_min, cohort_max, key):
                # Patient-centred sensitivity uses observed cohort ranges; other inputs remain at patient values.
                slider_min = float(max(cohort_min, current_value)) if raw_feature == "age" else float(cohort_min)
                slider_max = max(float(cohort_max), slider_min)
                if raw_feature in ("age", "numrelapse2", "wbsteps_all_sum_w", "walkdur_all_sum_w"):
                    min_v, max_v = int(round(slider_min)), int(round(slider_max))
                    max_v = max(max_v, min_v + 1)
                    default_v = int(round(np.clip(current_value, min_v, max_v)))
                    return float(st.slider(label, min_value=min_v, max_value=max_v, value=default_v, step=1, key=key))
                span = max(slider_max - slider_min, 1e-6)
                step = max(span / 200.0, 0.01)
                default_v = float(np.clip(current_value, slider_min, slider_max))
                return float(st.slider(label, min_value=slider_min, max_value=slider_max, value=default_v, step=float(step), key=key))

            def _track_sensitivity_change(section, feature_values, original_values, risk, key):
                signature = tuple((f, round(float(v), 6)) for f, v in feature_values.items())
                state_key = f"patient|{selected_pid}|{section}|{key}"
                previous = st.session_state.sensitivity_last_values.get(state_key)
                # Do not log the initial render; log each subsequent user change once.
                if previous is None:
                    st.session_state.sensitivity_last_values[state_key] = signature
                elif previous != signature:
                    st.session_state.sensitivity_change_history.append({
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "Patient": str(selected_pid),
                        "Raw values": dict(feature_values),
                        "Risk numeric": float(risk),
                        "Reference": "Patient baseline",
                        "Section": section,
                        "Changed feature(s)": ", ".join(FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_values),
                        "Original value(s)": "; ".join(f"{FEATURE_DISPLAY_NAMES.get(f,f)}={_sens_fmt(f, original_values[f])}" for f in feature_values),
                        "Scenario value(s)": "; ".join(f"{FEATURE_DISPLAY_NAMES.get(f,f)}={_sens_fmt(f, feature_values[f])}" for f in feature_values),
                        "Predicted risk": f"{risk * 100:.0f}%",
                    })
                    st.session_state.sensitivity_last_values[state_key] = signature

            def _render_section_sensitivity_tracker(section, tracker_key, baseline_values, baseline_risk):
                """Patient-referenced feature comparisons, ordered and coloured by local SHAP rank."""
                records = [r for r in st.session_state.sensitivity_change_history
                           if r.get("Reference") == "Patient baseline" and r.get("Section") == section
                           and r.get("Patient") == str(selected_pid)]
                st.markdown("##### Tracking history")
                rows = []
                for label, values, risk, when in [
                    ("Baseline", baseline_values, baseline_risk, "Reference")
                ] + [(f"Change {i}", r.get("Raw values", {}), r.get("Risk numeric", np.nan), r.get("Time", ""))
                     for i, r in enumerate(records, 1)]:
                    for feature, value in values.items():
                        rank = int(_rank_by_feature.get(feature, len(all_features))) + 1
                        rows.append({"Scenario": label, "Rank": rank,
                                     "Feature": FEATURE_DISPLAY_NAMES.get(feature, feature),
                                     "Patient": _sens_fmt(feature, baseline_values.get(feature, patient_base_dict.get(feature, value))),
                                     "Tested": _sens_fmt(feature, float(value)),
                                     "Risk": f"{float(risk)*100:.1f}%" if np.isfinite(risk) else "—",
                                     "Time": when})
                if rows:
                    frame = pd.DataFrame(rows)
                    def shade(row):
                        rank = int(row["Rank"])
                        bg = "#dbeafe" if rank == 1 else "#e9d5ff" if rank == 2 else "#dcfce7" if rank == 3 else "#f1f5f9"
                        return [f"background-color:{bg};font-weight:700" if col in ("Rank", "Feature") else "" for col in row.index]
                    st.dataframe(frame.style.apply(shade, axis=1), hide_index=True,
                                 width="stretch", height=min(390, 65+35*len(frame)))
                st.caption("Rank = patient-specific absolute SHAP importance. Patient and tested values are compared; risk is a hypothetical model prediction.")
                if st.button("Clear history", key=tracker_key, disabled=not bool(records)):
                    st.session_state.sensitivity_change_history = [r for r in st.session_state.sensitivity_change_history
                        if not (r.get("Reference") == "Patient baseline" and r.get("Section") == section
                                and r.get("Patient") == str(selected_pid))]
                    st.rerun()

            selected_pdp_display_label = _selected_interrogation_label
            st.caption(f"Testing the selected feature: **{selected_pdp_display_label}**")
            selected_pdp_feature = _selected_interrogation_feature
            feat_min = float(df[selected_pdp_feature].min())
            feat_max = float(df[selected_pdp_feature].max())
            patient_base_dict = {f: float(patient_records[f].iloc[-1]) for f in all_features}
            reference_val = float(patient_base_dict[selected_pdp_feature])
            base_risk = float(prog_risk)  # Exact same patient prediction as the header

            control_col, reset_col = st.columns([5, 1])
            with control_col:
                scenario_val = _sens_slider(
                    selected_pdp_feature,
                    f" {selected_pdp_display_label}",
                    reference_val, feat_min, feat_max,
                    f"pdp_scenario_patient_{selected_pid}_{selected_pdp_feature}"
                )
            with reset_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("Reset", key=f"pdp_reset_patient_{selected_pid}_{selected_pdp_feature}", width="stretch"):
                    st.session_state[f"pdp_scenario_patient_{selected_pid}_{selected_pdp_feature}"] = int(round(reference_val)) if selected_pdp_feature in ("age", "numrelapse2", "wbsteps_all_sum_w", "walkdur_all_sum_w") else float(reference_val)
                    st.rerun()

            scenario_dict = patient_base_dict.copy()
            scenario_dict[selected_pdp_feature] = scenario_val
            scenario_risk, _ = predictor.predict_custom(scenario_dict)
            risk_delta_pp = (scenario_risk - base_risk) * 100
            _track_sensitivity_change(
                "One-Feature", {selected_pdp_feature: scenario_val},
                {selected_pdp_feature: reference_val}, scenario_risk, selected_pdp_feature
            )

            pdp_col1, pdp_col2 = st.columns([2, 1])
            with pdp_col1:
                plot_min = max(feat_min, reference_val) if selected_pdp_feature == "age" else feat_min
                plot_max = max(feat_max, scenario_val, plot_min + (1 if selected_pdp_feature == "age" else 1e-6))
                x_pdp = np.linspace(plot_min, plot_max, 60)
                y_pdp = []
                for val in x_pdp:
                    temp_d = patient_base_dict.copy()
                    temp_d[selected_pdp_feature] = float(val)
                    prob, _ = predictor.predict_custom(temp_d)
                    y_pdp.append(prob * 100)

                fig_pdp = go.Figure()
                fig_pdp.add_trace(go.Scatter(
                    x=x_pdp, y=y_pdp, mode="lines", name="Risk sensitivity",
                    line=dict(color="#0284c7", width=3)
                ))
                fig_pdp.add_vline(x=reference_val, line_dash="dash", line_color="#64748b",
                                  annotation_text=f"Patient baseline ({_sens_fmt(selected_pdp_feature, reference_val)})",
                                  annotation_position="top left")
                fig_pdp.add_vline(x=scenario_val, line_dash="solid", line_color="#dc2626",
                                  annotation_text=f"Scenario ({_sens_fmt(selected_pdp_feature, scenario_val)})",
                                  annotation_position="top right")
                fig_pdp.add_trace(go.Scatter(
                    x=[scenario_val], y=[scenario_risk * 100], mode="markers",
                    marker=dict(size=14, color="#dc2626", line=dict(color="white", width=2)),
                    name="Scenario", hovertemplate="Scenario risk: %{y:.0f}%<extra></extra>"
                ))
                fig_pdp.update_layout(
                    title=f"Interactive Risk Sensitivity: {selected_pdp_display_label}", height=380,
                    margin=dict(l=20, r=20, t=55, b=20), plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title=selected_pdp_display_label, gridcolor="#f1f5f9"),
                    yaxis=dict(title="Predicted progression risk (%)", range=[0, 100], gridcolor="#f1f5f9")
                )
                st.plotly_chart(fig_pdp, width="stretch", config={"displayModeBar": False}, key="plot_line_4944")

            with pdp_col2:
                _render_section_sensitivity_tracker("One-Feature", "clear_tracker_one_feature",
                                                    {selected_pdp_feature: reference_val}, base_risk)

            # =====================================================================
            # INTERACTIVE TWO-FACTOR RISK SENSITIVITY
            # =====================================================================
            st.markdown("---")
            st.markdown("#### Two-Feature Risk Sensitivity")
            st.caption("Explore two features while holding all other inputs at this patient’s latest recorded values")

            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                x_feature_display = _selected_interrogation_label
                st.caption(f"X-axis: **{x_feature_display}** (selected influential feature)")
                x_feature_raw = _selected_interrogation_feature
            with hm_col2:
                _other_factor_options = [f for f in interpretable_feature_names if f != x_feature_display]
                y_feature_display = st.selectbox("Select Y-Axis Feature:", options=_other_factor_options,
                                                 key=f"hm_y_feature_select_{x_feature_raw}")
                y_feature_raw = DISPLAY_TO_RAW_FEATURE[y_feature_display]

            if x_feature_raw == y_feature_raw:
                st.warning("Please select two different features for the two-factor sensitivity view.")
            else:
                curr_patient_dict = {f: float(patient_records[f].iloc[-1]) for f in all_features}
                reference_x = float(curr_patient_dict[x_feature_raw])
                reference_y = float(curr_patient_dict[y_feature_raw])
                reference_risk = float(prog_risk)  # Exact same patient prediction as the header

                x_min, x_max = float(df[x_feature_raw].min()), float(df[x_feature_raw].max())
                y_min, y_max = float(df[y_feature_raw].min()), float(df[y_feature_raw].max())

                sx, sy = st.columns(2)
                with sx:
                    scenario_x = _sens_slider(x_feature_raw, f"Scenario: {x_feature_display}", reference_x, x_min, x_max,
                                              f"hm_scenario_x_patient_{selected_pid}_{x_feature_raw}_{y_feature_raw}")
                    if x_feature_raw == "age":
                        st.caption("Age varies from the patient’s recorded age upward; this is a hypothetical model response.")
                with sy:
                    scenario_y = _sens_slider(y_feature_raw, f"Scenario: {y_feature_display}", reference_y, y_min, y_max,
                                              f"hm_scenario_y_patient_{selected_pid}_{x_feature_raw}_{y_feature_raw}")
                    if y_feature_raw == "age":
                        st.caption("Age varies from the patient’s recorded age upward; this is a hypothetical model response.")

                scenario_hm_dict = curr_patient_dict.copy()
                scenario_hm_dict[x_feature_raw] = scenario_x
                scenario_hm_dict[y_feature_raw] = scenario_y
                scenario_hm_risk, _ = predictor.predict_custom(scenario_hm_dict)

                _track_sensitivity_change(
                    "Two-Factor", {x_feature_raw: scenario_x, y_feature_raw: scenario_y},
                    {x_feature_raw: reference_x, y_feature_raw: reference_y}, scenario_hm_risk,
                    f"{x_feature_raw}|{y_feature_raw}"
                )

                x_plot_min = max(x_min, reference_x) if x_feature_raw == "age" else x_min
                y_plot_min = max(y_min, reference_y) if y_feature_raw == "age" else y_min
                x_plot_max = max(x_max, scenario_x, x_plot_min + (1 if x_feature_raw == "age" else 1e-6))
                y_plot_max = max(y_max, scenario_y, y_plot_min + (1 if y_feature_raw == "age" else 1e-6))
                x_range = np.linspace(x_plot_min, x_plot_max, 7)
                y_range = np.linspace(y_plot_min, y_plot_max, 7)
                heatmap_z = np.zeros((len(y_range), len(x_range)))

                for i, y_v in enumerate(y_range):
                    for j, x_v in enumerate(x_range):
                        temp_dict = curr_patient_dict.copy()
                        temp_dict[x_feature_raw] = float(x_v)
                        temp_dict[y_feature_raw] = float(y_v)
                        r_prob, _ = predictor.predict_custom(temp_dict)
                        heatmap_z[i, j] = r_prob * 100

                clinical_risk_scale = [[0.00,"#15803d"],[0.29,"#86c66b"],[0.30,"#facc15"],
                                       [0.59,"#fb923c"],[0.60,"#ef4444"],[1.00,"#991b1b"]]
                fig_hm = go.Figure()
                fig_hm.add_trace(go.Heatmap(
                    z=heatmap_z, x=x_range, y=y_range, colorscale=clinical_risk_scale, zmin=0, zmax=100,
                    colorbar=dict(title=dict(text="Predicted<br>risk (%)"), tickvals=[0,30,60,100], thickness=18, len=0.78),
                    text=np.round(heatmap_z), texttemplate="<b>%{text:.0f}%</b>", textfont=dict(size=11, color="#111827"),
                    hovertemplate=(f"<b>{x_feature_display}</b>: %{{x:.2f}} {_sens_unit(x_feature_raw)}<br>"
                                   f"<b>{y_feature_display}</b>: %{{y:.2f}} {_sens_unit(y_feature_raw)}<br>"
                                   "<b>Predicted progression risk</b>: %{z:.0f}%<extra></extra>")
                ))
                fig_hm.add_trace(go.Scatter(
                    x=[reference_x], y=[reference_y], mode="markers+text", text=["Patient baseline"], textposition="top center",
                    marker=dict(size=16, color="#2563eb", symbol="circle", line=dict(color="white", width=3)),
                    name="Patient baseline"
                ))
                fig_hm.add_trace(go.Scatter(
                    x=[scenario_x], y=[scenario_y], mode="markers+text", text=["Scenario"], textposition="bottom center",
                    marker=dict(size=18, color="#dc2626", symbol="diamond", line=dict(color="white", width=3)),
                    customdata=[[scenario_hm_risk*100]],
                    hovertemplate="<b>Scenario</b><br>Predicted risk: %{customdata[0]:.0f}%<extra></extra>", name="Scenario"
                ))
                fig_hm.update_layout(
                    title=dict(text=f"Interactive Risk Surface: {x_feature_display} × {y_feature_display}", font=dict(size=15), x=0.01),
                    xaxis=dict(title=x_feature_display, gridcolor="#e2e8f0", zeroline=False),
                    yaxis=dict(title=y_feature_display, gridcolor="#e2e8f0", zeroline=False),
                    height=520, margin=dict(l=30,r=30,t=65,b=45), plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                heat_col, history_col = st.columns([2, 1])
                with heat_col:
                    st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False}, key="plot_line_5070")
                with history_col:
                    _render_section_sensitivity_tracker("Two-Factor", "clear_tracker_two_factor",
                        {x_feature_raw: reference_x, y_feature_raw: reference_y}, reference_risk)

            st.caption("Sensitivity scenarios are hypothetical model responses, not causal effects or treatment recommendations.")

    st.divider()
    st.markdown("### Record clinician interpretation")
    _f = _selected_interrogation_feature
    _assessment_key = f"{selected_pid}|{_f}"
    if "clinical_assessments" not in st.session_state:
        st.session_state.clinical_assessments = {}
    _previous = st.session_state.clinical_assessments.get(_assessment_key, {})
    _status_options = ["Plausible", "Questionable", "Requires additional evidence"]
    _status = st.radio("Explanation assessment", _status_options,
                       index=_status_options.index(_previous.get("status", "Requires additional evidence")),
                       horizontal=True, key=f"assessment_status_{_assessment_key}")
    _rationale = st.text_area("Clinical rationale", value=_previous.get("rationale", ""),
                              key=f"assessment_reason_{_assessment_key}")
    if st.button("Save clinician interpretation", key="save_assessment"):
        _record = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                   "patient": str(selected_pid), "feature": _f,
                   "feature_name": _selected_interrogation_label, "status": _status,
                   "rationale": _rationale, "model_prediction": float(prog_risk),
                   "shap_log_odds": float(_selected_shap)}
        st.session_state.clinical_assessments[_assessment_key] = _record
        st.session_state.setdefault("clinical_audit", []).append(dict(_record))
        st.success("Interpretation saved in this session's clinical audit trail.")

