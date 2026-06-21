"""
CREDIT RISK BANKING DASHBOARD - FINAL CLEAN VERSION
Professional Banking Analytics Platform with AI-Powered Insights
Sprint 2 - Interactive Data Application (Innomatics Research Labs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from io import StringIO
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Risk Dashboard", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# CSS STYLING - COMPLETELY FLAT, NO GRADIENTS
# ============================================================================

css_code = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Poppins:wght@400;600;700&display=swap');
* { font-family: 'Outfit', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif; font-weight: 700; }
.main { background: #0f2027; color: #ffffff; }
[data-testid="stSidebar"] { background: #1a2a3a; }
.main-header { background: #667eea; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 30px; box-shadow: none; }
.main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
.main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 10px; }
.metric-card { background: #5a6b7d; padding: 25px; border-radius: 12px; text-align: center; box-shadow: none; border: none; }
.metric-card.success { background: #2ecc71; }
.metric-card.danger { background: #e74c3c; }
.metric-card.info { background: #3498db; }
.metric-card.warning { background: #f39c12; }
.metric-value { font-size: 2.5rem; font-weight: 800; color: white; margin: 10px 0; }
.metric-label { font-size: 0.95rem; color: rgba(255,255,255,0.95); font-weight: 600; margin-bottom: 5px; }
.metric-delta { font-size: 0.85rem; color: rgba(255,255,255,0.85); margin-top: 8px; }
.icon-lg { font-size: 2rem; margin-bottom: 10px; display: block; }
.section-header { font-size: 1.5rem; font-weight: 700; color: white; margin-top: 30px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid #667eea; }
h1, h2, h3, h4, h5, h6 { color: white !important; }
p { color: rgba(255,255,255,0.9) !important; }
</style>
"""

st.markdown(css_code, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_risk_score(age, income, loan_amount, grade, employment_years):
    score = 0
    if age < 25: score += 30
    elif age < 35: score += 20
    elif age < 50: score += 10
    else: score += 5
    if income < 300000: score += 30
    elif income < 500000: score += 20
    elif income < 750000: score += 10
    else: score += 5
    loan_to_income = (loan_amount / income) * 100
    if loan_to_income > 50: score += 25
    elif loan_to_income > 35: score += 15
    elif loan_to_income > 20: score += 8
    else: score += 2
    grade_scores = {'A': 5, 'B': 15, 'C': 25, 'D': 35, 'E': 45, 'F': 50}
    score += grade_scores.get(grade, 30)
    if employment_years < 1: score += 20
    elif employment_years < 3: score += 10
    else: score += 5
    return min(score, 100)

def get_risk_level(score):
    if score < 25: return "LOW RISK", "#2ecc71"
    elif score < 50: return "MEDIUM RISK", "#f39c12"
    elif score < 75: return "HIGH RISK", "#e67e22"
    else: return "VERY HIGH RISK", "#e74c3c"

@st.cache_data
def load_data():
    return pd.read_csv('credit_risk_dataset.csv')

df = load_data()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""<h3 style="color: white; margin-top: 0;"><i class="fas fa-university"></i> CREDIT RISK DASHBOARD</h3>
    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Banking Analytics Platform</p>""", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio("Navigate:", ["Dashboard", "Analytics", "Patterns", "Risk", "Compare", "Scenarios", "Batch", "Advanced"], label_visibility="collapsed")
    
    st.divider()
    with st.expander("Information", expanded=True):
        st.markdown(f"""**Dataset Stats:**
        - Records: {df.shape[0]:,}
        - Features: {df.shape[1]}
        - Default Rate: {(df['loan_status'].sum()/len(df)*100):.1f}%
        **Classification Task** - Binary Prediction Model""")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.markdown("""<div class="main-header">
        <h1><i class="fas fa-chart-line"></i> Credit Risk Dashboard</h1>
        <p>Real-Time Loan Default Analysis & Insights</p>
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        st.markdown(f"""<div class="metric-card">
            <i class="fas fa-file-invoice icon-lg"></i>
            <div class="metric-label">Total Loans</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-delta">Records Analyzed</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        default_count = df['loan_status'].sum()
        default_pct = (default_count/len(df)*100)
        st.markdown(f"""<div class="metric-card danger">
            <i class="fas fa-exclamation-circle icon-lg"></i>
            <div class="metric-label">Defaults</div>
            <div class="metric-value">{default_count:,}</div>
            <div class="metric-delta">{default_pct:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        non_default = len(df) - df['loan_status'].sum()
        st.markdown(f"""<div class="metric-card success">
            <i class="fas fa-check-circle icon-lg"></i>
            <div class="metric-label">Non-Defaults</div>
            <div class="metric-value">{non_default:,}</div>
            <div class="metric-delta">{(non_default/len(df)*100):.1f}% repaid</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        avg_income = df['person_income'].mean()
        st.markdown(f"""<div class="metric-card info">
            <i class="fas fa-rupee-sign icon-lg"></i>
            <div class="metric-label">Avg Income</div>
            <div class="metric-value">₹{avg_income/100000:.1f}L</div>
            <div class="metric-delta">Annual</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="section-header"><i class="fas fa-chart-pie"></i> Loan Status Distribution</h3>', unsafe_allow_html=True)
        loan_counts = df['loan_status'].value_counts()
        fig, ax = plt.subplots(figsize=(9, 6), facecolor='#1a2a3a')
        ax.pie(loan_counts.values, labels=['Non-Default', 'Default'], autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'], startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Distribution', fontsize=14, fontweight='bold', color='white', pad=20)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header"><i class="fas fa-bolt"></i> Quick Insights</h3>', unsafe_allow_html=True)
        st.success(f"Total Loans: {df.shape[0]:,} records analyzed")
        st.error(f"High Risk: {df['loan_status'].sum():,} defaults detected")
        st.warning(f"Default Rate: {(df['loan_status'].sum()/len(df)*100):.1f}% - Action Required!")
        st.info(f"Recommendation: Deploy ML model for automated decisions")
    
    st.markdown("---")
    st.markdown('<h3 class="section-header"><i class="fas fa-calculator"></i> Quick Risk Calculator</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    with col1: calc_age = st.number_input("Age", 18, 80, 35)
    with col2: calc_income = st.number_input("Income (₹)", 100000, 5000000, 500000, 100000)
    with col3: calc_loan = st.number_input("Loan (₹)", 10000, 1000000, 300000, 50000)
    with col4: calc_grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F'])
    with col5: calc_emp = st.number_input("Emp (yrs)", 0, 50, 5)
    
    if st.button("Calculate Risk Score", use_container_width=True):
        risk_score = calculate_risk_score(calc_age, calc_income, calc_loan, calc_grade, calc_emp)
        risk_level, risk_color = get_risk_level(risk_score)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""<div class="metric-card" style="background: {risk_color};">
                <i class="fas fa-chart-bar icon-lg"></i>
                <div class="metric-label">Risk Assessment</div>
                <div class="metric-value">{risk_score:.0f}/100</div>
                <div class="metric-delta">{risk_level}</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            if risk_score < 25: st.success("APPROVE")
            elif risk_score < 50: st.warning("REVIEW")
            else: st.error("REJECT")

# PAGE 2: ANALYTICS
elif page == "Analytics":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-sliders-h"></i> Data Analytics</h1>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<h3 class="section-header"><i class="fas fa-filter"></i> Advanced Filters</h3>', unsafe_allow_html=True)
    
    f1, f2, f3, f4 = st.columns(4, gap="large")
    with f1: age_r = st.slider("Age", 18, 80, (25, 65))
    with f2: inc_r = st.slider("Income (₹)", 100000, 5000000, (200000, 1500000), 100000)
    with f3: grade_f = st.multiselect("Grade", ['A', 'B', 'C', 'D', 'E', 'F'], ['A', 'B', 'C', 'D', 'E', 'F'])
    with f4: emp_f = st.slider("Min Emp", 0, 50, 0)
    
    filtered_data = df[(df['person_age'] >= age_r[0]) & (df['person_age'] <= age_r[1]) &
                       (df['person_income'] >= inc_r[0]) & (df['person_income'] <= inc_r[1]) &
                       (df['loan_grade'].isin(grade_f)) & (df['person_emp_length'] >= emp_f)]
    
    st.info(f"Showing **{len(filtered_data)}** records (Filtered from {len(df)} total)")
    st.divider()
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<h3 class="section-header"><i class="fas fa-hashtag"></i> Numerical</h3>', unsafe_allow_html=True)
        num_cols = [col for col in filtered_data.select_dtypes(include=['int64', 'float64']).columns if col != 'loan_status']
        sel = st.selectbox("Select:", num_cols, key="num")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a2a3a')
        ax.hist(filtered_data[sel], bins=30, color='#667eea', alpha=0.8, edgecolor='white')
        ax.set_xlabel(sel, fontweight='bold', color='white')
        ax.set_ylabel('Count', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    
    with c2:
        st.markdown('<h3 class="section-header"><i class="fas fa-tags"></i> Categorical</h3>', unsafe_allow_html=True)
        cat_cols = filtered_data.select_dtypes(include=['object']).columns.tolist()
        sel_c = st.selectbox("Select:", cat_cols, key="cat")
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a2a3a')
        counts = filtered_data[sel_c].value_counts()
        ax.barh(counts.index, counts.values, color='#764ba2', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Count', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)

# PAGE 3: PATTERNS
elif page == "Patterns":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-project-diagram"></i> Key Patterns</h1>', unsafe_allow_html=True)
    st.divider()
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown('<h3 class="section-header"><i class="fas fa-building"></i> Default Rate by Grade</h3>', unsafe_allow_html=True)
        grade_def = df.groupby('loan_grade')['loan_status'].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a2a3a')
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8b0000']
        ax.bar(grade_def.index, grade_def.values, color=colors[:len(grade_def)], alpha=0.85, edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Default Rate (%)', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    
    with c2:
        st.markdown('<h3 class="section-header"><i class="fas fa-lightbulb"></i> Default Rate by Intent</h3>', unsafe_allow_html=True)
        intent_def = df.groupby('loan_intent')['loan_status'].mean() * 100
        intent_def = intent_def.sort_values()
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a2a3a')
        ax.barh(intent_def.index, intent_def.values, color='#667eea', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Default Rate (%)', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)

# PAGE 4: RISK
elif page == "Risk":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-radiation-alt"></i> Risk Assessment</h1>', unsafe_allow_html=True)
    st.divider()
    df_a = df.copy()
    df_a['income_cat'] = pd.qcut(df_a['person_income'], q=4, labels=['Low', 'Med', 'High', 'V.High'], duplicates='drop')
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<h3 class="section-header"><i class="fas fa-coins"></i> Risk by Income</h3>', unsafe_allow_html=True)
        inc_risk = df_a.groupby('income_cat')['loan_status'].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a2a3a')
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        ax.bar(range(len(inc_risk)), inc_risk.values, color=colors[:len(inc_risk)], alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(inc_risk)))
        ax.set_xticklabels(inc_risk.index, color='white')
        ax.set_ylabel('Default Rate (%)', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    
    with c2:
        st.markdown('<h3 class="section-header"><i class="fas fa-skull"></i> Risk Indicators</h3>', unsafe_allow_html=True)
        st.error("Very High: 35%+ default rate")
        st.warning("High: 25%+ default rate")
        st.success("Low: <15% default rate")
        st.info("Action: Risk-based pricing")

# PAGE 5: COMPARE
elif page == "Compare":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-exchange-alt"></i> Customer Comparison</h1>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<h3 class="section-header"><i class="fas fa-user"></i> Customer 1</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: c1_age = st.number_input("Age 1", 18, 80, 35)
    with c2: c1_inc = st.number_input("Income 1 (₹)", 100000, 5000000, 500000, 100000)
    with c3: c1_loan = st.number_input("Loan 1 (₹)", 10000, 1000000, 300000, 50000)
    with c4: c1_grade = st.selectbox("Grade 1", ['A', 'B', 'C', 'D', 'E', 'F'])
    with c5: c1_emp = st.number_input("Emp 1 (yrs)", 0, 50, 5)
    
    st.markdown('<h3 class="section-header"><i class="fas fa-user"></i> Customer 2</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: c2_age = st.number_input("Age 2", 18, 80, 40)
    with c2: c2_inc = st.number_input("Income 2 (₹)", 100000, 5000000, 700000, 100000)
    with c3: c2_loan = st.number_input("Loan 2 (₹)", 10000, 1000000, 400000, 50000)
    with c4: c2_grade = st.selectbox("Grade 2", ['A', 'B', 'C', 'D', 'E', 'F'])
    with c5: c2_emp = st.number_input("Emp 2 (yrs)", 0, 50, 8)
    
    if st.button("Compare Customers", use_container_width=True):
        c1_score = calculate_risk_score(c1_age, c1_inc, c1_loan, c1_grade, c1_emp)
        c2_score = calculate_risk_score(c2_age, c2_inc, c2_loan, c2_grade, c2_emp)
        
        c1_level, c1_color = get_risk_level(c1_score)
        c2_level, c2_color = get_risk_level(c2_score)
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(f"""<div class="metric-card" style="background: {c1_color};">
                <i class="fas fa-user icon-lg"></i>
                <div class="metric-label">Customer 1</div>
                <div class="metric-value">{c1_score:.0f}/100</div>
                <div class="metric-delta">{c1_level}</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class="metric-card" style="background: {c2_color};">
                <i class="fas fa-user icon-lg"></i>
                <div class="metric-label">Customer 2</div>
                <div class="metric-value">{c2_score:.0f}/100</div>
                <div class="metric-delta">{c2_level}</div>
            </div>""", unsafe_allow_html=True)
        
        st.divider()
        comparison_data = {
            'Metric': ['Age', 'Income (₹)', 'Loan (₹)', 'Grade', 'Employment', 'Risk Score'],
            'Customer 1': [c1_age, f"₹{c1_inc/100000:.1f}L", f"₹{c1_loan/100000:.1f}L", c1_grade, f"{c1_emp} yrs", f"{c1_score:.0f}/100"],
            'Customer 2': [c2_age, f"₹{c2_inc/100000:.1f}L", f"₹{c2_loan/100000:.1f}L", c2_grade, f"{c2_emp} yrs", f"{c2_score:.0f}/100"]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# PAGE 6: SCENARIOS
elif page == "Scenarios":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-road"></i> What-If Scenarios</h1>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<h3 class="section-header"><i class="fas fa-sliders-h"></i> Base Profile</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: base_age = st.number_input("Age", 18, 80, 35)
    with c2: base_inc = st.number_input("Income (₹)", 100000, 5000000, 500000, 100000)
    with c3: base_loan = st.number_input("Loan (₹)", 10000, 1000000, 300000, 50000)
    with c4: base_grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F'])
    with c5: base_emp = st.number_input("Emp (yrs)", 0, 50, 5)
    
    base_score = calculate_risk_score(base_age, base_inc, base_loan, base_grade, base_emp)
    
    st.markdown('<h3 class="section-header"><i class="fas fa-pencil-alt"></i> Modify Scenario</h3>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown("**Increase Income by:**")
        inc_change = st.slider("Income % Increase", 0, 200, 50) / 100
        new_inc = base_inc * (1 + inc_change)
    
    with c2:
        st.markdown("**Reduce Loan by:**")
        loan_change = st.slider("Loan % Reduction", 0, 100, 20) / 100
        new_loan = base_loan * (1 - loan_change)
    
    with c3:
        st.markdown("**Change Grade:**")
        new_grade = st.selectbox("New Grade", ['A', 'B', 'C', 'D', 'E', 'F'], index=0)
    
    new_score = calculate_risk_score(base_age, new_inc, new_loan, new_grade, base_emp)
    improvement = base_score - new_score
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<h3 class="section-header"><i class="fas fa-chart-bar"></i> Base Score</h3>', unsafe_allow_html=True)
        level, color = get_risk_level(base_score)
        st.markdown(f"""<div class="metric-card" style="background: {color};">
            <i class="fas fa-chart-line icon-lg"></i>
            <div class="metric-value">{base_score:.0f}/100</div>
            <div class="metric-delta">{level}</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="section-header"><i class="fas fa-arrow-right"></i> New Score</h3>', unsafe_allow_html=True)
        new_level, new_color = get_risk_level(new_score)
        st.markdown(f"""<div class="metric-card" style="background: {new_color};">
            <i class="fas fa-chart-line icon-lg"></i>
            <div class="metric-value">{new_score:.0f}/100</div>
            <div class="metric-delta">{new_level}</div>
        </div>""", unsafe_allow_html=True)
    
    st.divider()
    if improvement > 0: st.success(f"✅ Risk improved by {improvement:.0f} points!")
    elif improvement < 0: st.error(f"⚠️ Risk increased by {abs(improvement):.0f} points!")
    else: st.info("No change in risk score")

# PAGE 7: BATCH
elif page == "Batch":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-upload"></i> Batch Risk Assessment</h1>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown('<h3 class="section-header"><i class="fas fa-file-csv"></i> Upload CSV File</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV with columns: age, income, loan_amount, grade, employment_years", type=['csv'])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_df['risk_score'] = batch_df.apply(lambda x: calculate_risk_score(x.get('age', 35), x.get('income', 500000), 
                                                                                 x.get('loan_amount', 300000), x.get('grade', 'C'), 
                                                                                 x.get('employment_years', 5)), axis=1)
        batch_df['risk_level'] = batch_df['risk_score'].apply(lambda x: get_risk_level(x)[0])
        
        st.markdown('<h3 class="section-header"><i class="fas fa-table"></i> Assessment Results</h3>', unsafe_allow_html=True)
        st.dataframe(batch_df, use_container_width=True)
        
        csv_result = batch_df.to_csv(index=False)
        st.download_button("Download Results (CSV)", csv_result, "batch_assessment_results.csv", "text/csv", use_container_width=True)

# PAGE 8: ADVANCED
elif page == "Advanced":
    st.markdown('<h1 style="color:white; text-align:center;"><i class="fas fa-microscope"></i> Advanced Analytics</h1>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<h3 class="section-header"><i class="fas fa-database"></i> Data Explorer</h3>', unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3, gap="large")
    with f1: status = st.selectbox("Status:", ["All", "Non-Default", "Default"])
    with f2: grades = st.multiselect("Grade:", df['loan_grade'].unique(), default=df['loan_grade'].unique())
    with f3: intents = st.multiselect("Intent:", df['loan_intent'].unique(), default=df['loan_intent'].unique()[:3])
    
    exp_df = df.copy()
    if status == "Non-Default": exp_df = exp_df[exp_df['loan_status'] == 0]
    elif status == "Default": exp_df = exp_df[exp_df['loan_status'] == 1]
    exp_df = exp_df[exp_df['loan_grade'].isin(grades)]
    exp_df = exp_df[exp_df['loan_intent'].isin(intents)]
    
    st.info(f"**{len(exp_df)}** records (Total: {len(df)})")
    st.markdown('<h3 class="section-header"><i class="fas fa-table"></i> Data</h3>', unsafe_allow_html=True)
    st.dataframe(exp_df.head(50), use_container_width=True)
    
    csv = exp_df.to_csv(index=False)
    st.download_button("Download (CSV)", csv, f"credit_risk_{status.replace(' ', '_')}.csv", "text/csv", use_container_width=True)

st.divider()
st.markdown("""<div style='text-align: center; padding: 20px; color: rgba(255,255,255,0.7);'>
    <p><i class="fas fa-bank"></i> <strong>Credit Risk Banking Dashboard</strong></p>
</div>""", unsafe_allow_html=True)