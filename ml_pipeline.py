# ============================================================================
# ML PIPELINE - Credit Risk Banking - Classification
# SPRINT 3: Model Building & Evaluation
# ============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, auc)

# Models - Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

print("="*80)
print("STEP 1: DATA LOADING & EXPLORATION")
print("="*80)

# Load data
df = pd.read_csv('credit_risk_dataset.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Target variable distribution:")
print(df['loan_status'].value_counts())
print(f"\nDefault Rate: {(df['loan_status'].sum() / len(df)) * 100:.2f}%")
print(f"Missing values: {df.isnull().sum().sum()}")

# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING & DATA PREPARATION")
print("="*80)

df_processed = df.copy()

# Identify and encode categorical variables
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_cols}")

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"✓ Encoded: {col}")

# Create additional engineered features
print("\nCreating engineered features...")

# Age group
df_processed['age_group'] = pd.cut(df_processed['person_age'], 
                                    bins=[0, 25, 35, 45, 100], 
                                    labels=[0, 1, 2, 3]).astype(int)

# Income level
df_processed['income_level'] = pd.qcut(df_processed['person_income'], 
                                        q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)

# Loan to income ratio category
df_processed['loan_income_ratio_cat'] = pd.qcut(df_processed['loan_percent_income'], 
                                                 q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)

# Risk score (combination of factors)
df_processed['risk_score'] = (
    (df_processed['loan_percent_income'] / df_processed['loan_percent_income'].max()) * 0.3 +
    (df_processed['loan_int_rate'] / df_processed['loan_int_rate'].max()) * 0.3 +
    (df_processed['loan_grade'] / df_processed['loan_grade'].max()) * 0.2 +
    (df_processed['cb_person_default_on_file'].astype(int) * 0.2)
)

print(f"✓ Age group feature created")
print(f"✓ Income level feature created")
print(f"✓ Loan to income ratio category created")
print(f"✓ Risk score feature created")
print(f"Total features: {df_processed.shape[1]}")

# ============================================================================
print("\n" + "="*80)
print("STEP 3: PREPARE DATA FOR MODELING")
print("="*80)

# Select features (exclude original categorical cols that are encoded)
feature_columns = [col for col in df_processed.columns 
                   if col != 'loan_status' and col not in categorical_cols]

print(f"\nSelected features ({len(feature_columns)}):")
print(feature_columns)

X = df_processed[feature_columns]
y = df_processed['loan_status']

# Train-test split (75:25 as per Innomatics requirement)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTrain-test split (75:25):")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"  Features: {X_train.shape[1]}")

# Standardization (fit on train, transform on test)
print(f"\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Features scaled using StandardScaler")

# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN CLASSIFICATION MODELS")
print("="*80)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                           random_state=42, n_jobs=-1)
}

results = []

print("\nTraining models...\n")

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data for specific models
    if name in ['KNN', 'Support Vector Machine (SVM)', 'Logistic Regression']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test

    # Train model
    model.fit(X_train_model, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_model)
    y_test_pred = model.predict(X_test_model)
    
    # Probability predictions for ROC-AUC
    y_test_pred_proba = model.predict_proba(X_test_model)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Cross-validation
    if name in ['KNN', 'Support Vector Machine (SVM)', 'Logistic Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=5, scoring='accuracy', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=5, scoring='accuracy', n_jobs=-1)

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        'Model': name,
        'Train_Accuracy': train_accuracy,
        'Test_Accuracy': test_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'CV_Accuracy_Mean': cv_mean,
        'CV_Accuracy_Std': cv_std,
        'Overfit': train_accuracy - test_accuracy
    })

    print(f"  ✓ Accuracy: {test_accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# ============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION & COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_Accuracy', ascending=False)

print("\nMODEL PERFORMANCE RANKING:")
print("="*80)
print(results_df[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].to_string(index=False))

# Save results
results_df.to_csv('model_evaluation_results.csv', index=False)
print("\n✓ Results saved to 'model_evaluation_results.csv'")

# ============================================================================
print("\n" + "="*80)
print("STEP 6: SELECT BEST MODEL & DETAILED ANALYSIS")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_model_obj = models[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")
print(f"   Precision: {results_df.iloc[0]['Precision']:.4f}")
print(f"   Recall: {results_df.iloc[0]['Recall']:.4f}")
print(f"   F1-Score: {results_df.iloc[0]['F1_Score']:.4f}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC_AUC']:.4f}")
print(f"   CV Accuracy: {results_df.iloc[0]['CV_Accuracy_Mean']:.4f} ± {results_df.iloc[0]['CV_Accuracy_Std']:.3f}")

# Retrain best model on all data for production
best_model_name_final = best_model_name
best_model_final = models[best_model_name_final]

if best_model_name_final in ['KNN', 'Support Vector Machine (SVM)', 'Logistic Regression']:
    X_final = scaler.fit_transform(X)
    best_model_final.fit(X_final, y)
    use_scaling = True
else:
    best_model_final.fit(X, y)
    use_scaling = False

print("\n✓ Best model retrained on all data for production deployment")

# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVE MODEL PACKAGE")
print("="*80)

model_package = {
    'model': best_model_final,
    'model_name': best_model_name_final,
    'scaler': scaler if use_scaling else None,
    'feature_names': list(X.columns),
    'use_scaling': use_scaling,
    'label_encoders': label_encoders,
    'performance': {
        'test_accuracy': float(results_df.iloc[0]['Test_Accuracy']),
        'precision': float(results_df.iloc[0]['Precision']),
        'recall': float(results_df.iloc[0]['Recall']),
        'f1_score': float(results_df.iloc[0]['F1_Score']),
        'roc_auc': float(results_df.iloc[0]['ROC_AUC'])
    }
}

with open('credit_risk_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✓ Model saved as 'credit_risk_model.pkl'")
print("\nPackage contains:")
print(f"  - Model: {best_model_name_final}")
print(f"  - Features: {len(model_package['feature_names'])}")
print(f"  - Scaler: {'Yes' if use_scaling else 'No'}")
print(f"  - Encoders: {', '.join(label_encoders.keys())}")

# ============================================================================
print("\n" + "="*80)
print("STEP 8: VISUALIZE RESULTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy comparison
ax1 = axes[0, 0]
y_pos = np.arange(len(results_df))
colors = ['green' if i == 0 else 'lightblue' for i in range(len(results_df))]
ax1.barh(y_pos, results_df['Test_Accuracy'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(results_df['Model'], fontsize=9)
ax1.set_xlabel('Accuracy', fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim([0, 1])
for i, v in enumerate(results_df['Test_Accuracy']):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

# 2. Precision vs Recall comparison
ax2 = axes[0, 1]
x = np.arange(len(results_df))
width = 0.35
ax2.bar(x - width/2, results_df['Precision'], width, label='Precision', alpha=0.7)
ax2.bar(x + width/2, results_df['Recall'], width, label='Recall', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Score', fontweight='bold')
ax2.set_title('Precision vs Recall', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1])

# 3. F1 Score & ROC-AUC
ax3 = axes[1, 0]
x = np.arange(len(results_df))
width = 0.35
ax3.bar(x - width/2, results_df['F1_Score'], width, label='F1-Score', alpha=0.7)
ax3.bar(x + width/2, results_df['ROC_AUC'], width, label='ROC-AUC', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('F1-Score & ROC-AUC Comparison', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1])

# 4. Cross-Validation Scores
ax4 = axes[1, 1]
ax4.errorbar(range(len(results_df)), results_df['CV_Accuracy_Mean'], 
             yerr=results_df['CV_Accuracy_Std'], fmt='o', markersize=8, 
             capsize=5, capthick=2, alpha=0.7, color='purple', label='CV Score')
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Cross-Validation Scores (5-Fold)', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 1])
ax4.legend()

plt.suptitle('ML Model Comparison - Credit Risk Classification', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison visualization saved as 'model_comparison.png'")

# ============================================================================
print("\n" + "="*80)
print("STEP 9: CONFUSION MATRIX & DETAILED CLASSIFICATION REPORT")
print("="*80)

# Get predictions from best model
if best_model_name_final in ['KNN', 'Support Vector Machine (SVM)', 'Logistic Regression']:
    y_test_pred_best = best_model_final.predict(X_test_scaled)
else:
    y_test_pred_best = best_model_final.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred_best)

print(f"\nConfusion Matrix for {best_model_name_final}:")
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred_best, target_names=['Non-Default', 'Default']))

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
ax.set_ylabel('True Label', fontweight='bold')
ax.set_xlabel('Predicted Label', fontweight='bold')
ax.set_title(f'Confusion Matrix - {best_model_name_final}', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================================
print("\n" + "="*80)
print("STEP 10: ROC CURVE ANALYSIS")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    if name in ['KNN', 'Support Vector Machine (SVM)', 'Logistic Regression']:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    line_style = '--' if name != best_model_name_final else '-'
    line_width = 1.5 if name != best_model_name_final else 2.5
    label = f'{name} (AUC: {roc_auc:.3f})'
    if name == best_model_name_final:
        label += ' ★ BEST'
    
    ax.plot(fpr, tpr, linestyle=line_style, linewidth=line_width, label=label)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curve Comparison - All Models', fontweight='bold', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("\n✓ ROC curve comparison saved as 'roc_curve.png'")

# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETED!")
print("="*80)

print("""
Generated files:
  1. credit_risk_model.pkl - Trained best model
  2. model_evaluation_results.csv - All model metrics
  3. model_comparison.png - Accuracy comparison chart
  4. confusion_matrix.png - Best model confusion matrix
  5. roc_curve.png - ROC curves for all models

CONCLUSIONS:
""")

print(f"\n✓ BEST MODEL: {best_model_name_final}")
print(f"  - Test Accuracy: {results_df.iloc[0]['Test_Accuracy']:.2%}")
print(f"  - Precision: {results_df.iloc[0]['Precision']:.2%} (Low False Positives)")
print(f"  - Recall: {results_df.iloc[0]['Recall']:.2%} (Catches Default Cases)")
print(f"  - F1-Score: {results_df.iloc[0]['F1_Score']:.4f}")
print(f"  - ROC-AUC: {results_df.iloc[0]['ROC_AUC']:.4f}")

print(f"""
BUSINESS IMPACT:
• Model ready for production deployment
• Can be integrated into loan approval process
• Enables real-time risk assessment
• Reduces Non-Performing Assets (NPA)
• Improves customer experience with faster approvals

RECOMMENDATIONS:
• Use model with confidence threshold (not just probability)
• Monitor model performance in production
• Retrain periodically with new data
• Consider ensemble methods for better accuracy
• Implement human review for borderline cases
""")
