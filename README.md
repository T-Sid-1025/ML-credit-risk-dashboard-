# 🏦 Credit Risk Banking - Innomatics Case Study

## Project Overview

This is a comprehensive **Credit Risk Assessment & Loan Default Prediction** project built following **Innomatics Research Labs** case study requirements. It covers all three sprints with complete EDA, interactive dashboard, and ML classification models.

**Domain:** Banking/Finance  
**Problem:** Predict loan default probability  
**Target:** `loan_status` (0 = Non-Default, 1 = Default)  
**Task Type:** Binary Classification  

---

## 📂 Project Structure

```
credit-risk-project/
├── 📊 01_EDA.ipynb                      # SPRINT 1: Exploratory Data Analysis
├── 🎯 app.py                            # SPRINT 2: Streamlit Dashboard
├── 🤖 ml_pipeline.py                    # SPRINT 3: ML Model Training
├── 📋 requirements.txt                  # Python dependencies
├── 📄 credit_risk_dataset.csv           # Input dataset
└── 📁 outputs/                          # Generated files
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── credit_risk_model.pkl
    └── model_evaluation_results.csv
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
Ensure `credit_risk_dataset.csv` is in the project root directory.

### 3. Run Each Sprint

#### SPRINT 1: EDA Analysis
```bash
jupyter notebook 01_EDA.ipynb
```
**Output:** 6 visualization PNGs + insights text files

#### SPRINT 2: Interactive Dashboard
```bash
streamlit run app.py
```
Opens interactive dashboard on `http://localhost:8501`

#### SPRINT 3: ML Model Training
```bash
python ml_pipeline.py
```
**Output:** Trained model + performance metrics + visualizations

---

## 📊 SPRINT 1: Exploratory Data Analysis (01_EDA.ipynb)

### What It Does:
- Loads and explores the credit risk dataset
- Analyzes data distribution and quality
- Identifies patterns between features and loan defaults
- Generates visualizations and business insights

### Key Findings:
✓ **Class Distribution:** Check default vs non-default rate  
✓ **Feature Importance:** Correlation analysis with target  
✓ **Risk Factors:** Loan grade, intent, income, employment history  
✓ **Default Patterns:** By demographics, loan characteristics  

### Generated Files:
1. `01_target_distribution.png` - Loan status pie/bar charts
2. `02_numerical_distributions.png` - Feature distributions
3. `03_categorical_distributions.png` - Category breakdowns
4. `04_correlation_analysis.png` - Heatmap & feature importance
5. `05_key_patterns.png` - Default rates by grade, intent, etc.
6. `06_advanced_analysis.png` - Income, loan amount, credit history
7. `EDA_INSIGHTS.txt` - Business recommendations
8. `EDA_SUMMARY.txt` - Summary statistics

---

## 🎯 SPRINT 2: Interactive Dashboard (app.py)

### Dashboard Sections:

#### 📈 Overview
- Key metrics (Total loans, defaults, non-defaults)
- Target variable visualization
- Quick risk assessment

#### 📊 Data Exploration
- Numerical features distribution
- Categorical features breakdown
- Feature statistics

#### 🎯 Key Patterns
- Default rate by loan grade
- Default rate by loan intent
- Default rate by home ownership
- Credit history impact

#### ⚠️ Risk Analysis
- Default risk by income level
- Default risk by loan amount
- Loan-to-income ratio analysis
- Previous default history impact

#### 🔍 Detailed Analysis
- Feature correlation heatmap
- Feature importance ranking
- Data filtering & preview
- CSV export capability

### How to Run:
```bash
streamlit run app.py
```

### Features:
✓ Interactive filters  
✓ Dynamic visualizations  
✓ Real-time data exploration  
✓ Download capabilities  
✓ Professional formatting  

---

## 🤖 SPRINT 3: ML Model Building (ml_pipeline.py)

### Models Implemented:
1. **Logistic Regression** - Linear classification baseline
2. **K-Nearest Neighbors (KNN)** - Distance-based classification
3. **Support Vector Machine (SVM)** - Kernel-based classifier
4. **Decision Tree** - Tree-based classifier
5. **Random Forest** - Ensemble method

### Pipeline Steps:

#### Step 1: Data Loading
- Load credit risk dataset
- Check missing values and shape

#### Step 2: Feature Engineering
```
Features Created:
✓ Age groups (4 categories)
✓ Income levels (4 quartiles)
✓ Loan-to-income ratio categories
✓ Risk scores (weighted combination)
```

#### Step 3: Train-Test Split
- **75% Training** data for model development
- **25% Test** data for evaluation
- **Stratified split** to preserve class distribution

#### Step 4: Data Preprocessing
- **Numerical:** StandardScaler normalization
- **Categorical:** LabelEncoding
- Fit scaler on train data only (no data leakage)

#### Step 5: Model Training
- Train 5 different classification models
- Use cross-validation (5-fold) for robustness

#### Step 6: Model Evaluation

**Metrics Used:**
- **Accuracy:** Overall correctness
- **Precision:** False positives control (important!)
- **Recall:** False negatives control (catch defaults!)
- **F1-Score:** Harmonic mean of precision & recall
- **ROC-AUC:** Overall discriminative ability

**Classification Report:**
- Per-class metrics
- Weighted averages
- Support (sample counts)

#### Step 7: Best Model Selection
- Model with highest test accuracy selected
- Retrained on full dataset for production

#### Step 8: Model Serialization
```python
model_package = {
    'model': trained_model,
    'model_name': 'Best Model Name',
    'scaler': fitted_scaler,
    'feature_names': [...],
    'use_scaling': True/False,
    'label_encoders': {...},
    'performance': {accuracy, precision, recall, f1, auc}
}
```
Saved as `credit_risk_model.pkl`

#### Step 9: Visualizations
1. **model_comparison.png** - Bar charts comparing all models
2. **confusion_matrix.png** - True positives/negatives breakdown
3. **roc_curve.png** - ROC curves for all models

---

## 📈 Key Metrics Explained

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Best for: Balanced datasets
⚠️ Misleading for imbalanced data
```

### Precision
```
Precision = TP / (TP + FP)
Meaning: Of predicted defaults, how many are actually defaults?
Important for: Avoiding false alarms (costly wrong predictions)
```

### Recall
```
Recall = TP / (TP + FN)
Meaning: Of actual defaults, how many did we catch?
Important for: Bank's risk management (missing defaults is costly)
```

### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Meaning: Harmonic mean balancing precision and recall
Best for: Imbalanced classification
```

### ROC-AUC
```
ROC-AUC: Area under the Receiver Operating Characteristic Curve
Range: 0 to 1 (1.0 = perfect, 0.5 = random)
Meaning: Model's ability to distinguish between classes
Best for: Overall model quality assessment
```

---

## 🎯 Usage Example: Making Predictions

```python
import pickle
import numpy as np

# Load trained model
with open('credit_risk_model.pkl', 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']
use_scaling = model_package['use_scaling']

# Prepare input (must match feature_names order)
input_data = np.array([[...]])  # Your customer data

# Scale if needed
if use_scaling:
    input_data = scaler.transform(input_data)

# Predict
probability = model.predict_proba(input_data)[0][1]  # Default probability
prediction = model.predict(input_data)[0]  # 0 = Safe, 1 = Default

print(f"Default Probability: {probability:.2%}")
print(f"Decision: {'⚠️ DEFAULT' if prediction == 1 else '✓ APPROVED'}")
```

---

## 💡 Business Recommendations

### 1. Risk-Adjusted Pricing
- Use model predictions for interest rate determination
- Higher risk customers → Higher interest rates

### 2. Approval Automation
- Approve low-risk customers instantly (probability < 0.3)
- Reject high-risk customers automatically (probability > 0.8)
- Route borderline cases (0.3-0.8) to manual review

### 3. Portfolio Management
- Monitor overall default rate vs predictions
- Identify systematic under-/over-estimation
- Retrain models quarterly with new data

### 4. Customer Segmentation
- Create risk tiers for customer targeting
- Offer different products based on risk
- Implement early warning system for risky customers

---

## 📊 Performance Benchmarks

**Expected Accuracy Range:** 60-85%
(Depends on dataset quality and class imbalance)

**Key Success Metrics:**
✓ High Recall (minimize missed defaults)
✓ Reasonable Precision (minimize false alarms)
✓ ROC-AUC > 0.7 (good discriminative ability)

---

## 🔍 Troubleshooting

### Issue: Module not found
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Streamlit not opening
**Solution:**
```bash
streamlit run app.py --logger.level=debug
```

### Issue: Model accuracy very low
**Solutions:**
1. Check for missing values in data
2. Verify train-test split is stratified
3. Try different feature engineering
4. Check for data leakage
5. Consider ensemble methods

---

## 📚 References

**Innomatics Case Study:** Credit Risk Banking Domain
**Problem Type:** Binary Classification
**Dataset:** credit_risk_dataset.csv (12 features, ~32K records)

---

## 👤 Project Details

**Created for:** Innomatics Research Labs  
**Course:** Data Science & ML  
**Type:** 3-Sprint Capstone Project  

**Sprints:**
- ✅ SPRINT 1: EDA & Analysis
- ✅ SPRINT 2: Interactive Dashboard
- ✅ SPRINT 3: ML Model Building

---

## 📝 License

This project is for educational purposes as part of Innomatics Research Labs curriculum.

---

## ✨ Next Steps

1. **Explore EDA notebook** → Understand data patterns
2. **Run dashboard** → Visualize insights interactively
3. **Train ML models** → Get predictions
4. **Deploy model** → Integrate into banking system
5. **Monitor performance** → Track accuracy over time

---

**Happy learning! 🚀**

Questions? Refer to the code comments and Jupyter notebook markdown cells for detailed explanations.
