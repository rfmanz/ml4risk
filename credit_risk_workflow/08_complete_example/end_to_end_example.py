"""
End-to-End Credit Risk Model Development Example

This script demonstrates the complete workflow for building a production-ready
credit risk model using the ml4risk library.

The workflow includes:
1. Data preparation
2. Performance analysis and target definition
3. Reject inference
4. Feature engineering with WOE
5. Model development
6. Score alignment
7. Monitoring setup
"""

from rich import pretty
pretty.install()
from rich import print 
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')



# Add parent directories to path to import ml4risk modules
sys.path.append('../01_data_preparation')
sys.path.append('../02_performance_analysis')
sys.path.append('../03_reject_inference')
sys.path.append('../04_feature_engineering')
sys.path.append('../05_model_development')
sys.path.append('../06_score_alignment')
sys.path.append('../07_monitoring')

print("=== End-to-End Credit Risk Model Development ===\n")

#region Step 1: Data Generation and Preparation
print("Step 1: Data Generation and Preparation")
print("-" * 50)

# Generate synthetic credit data
np.random.seed(42)
n_applications = 20000
approval_rate = 0.6

# Create base features
application_data = pd.DataFrame({
    'application_id': range(n_applications),
    'application_date': pd.date_range(
        start='2022-01-01', 
        end='2022-12-31', 
        periods=n_applications
    ),
    
    # Credit bureau features
    'bureau_score': np.random.normal(650, 100, n_applications).clip(300, 850),
    'num_tradelines': np.random.poisson(5, n_applications),
    'oldest_tradeline_months': np.random.exponential(60, n_applications).clip(0, 300),
    'num_inquiries_6mo': np.random.poisson(2, n_applications),
    'num_delinquencies': np.random.poisson(0.5, n_applications),
    
    # Financial features
    'annual_income': np.random.lognormal(10.5, 0.5, n_applications),
    'debt_to_income': np.random.beta(2, 5, n_applications),
    'employment_years': np.random.exponential(5, n_applications).clip(0, 40),
    
    # Demographic features
    'age': np.random.normal(40, 12, n_applications).clip(18, 80),
    'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_applications, p=[0.3, 0.2, 0.5]),
    
    # Loan features
    'loan_amount': np.random.lognormal(9.5, 0.8, n_applications),
    'loan_purpose': np.random.choice(['DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'PERSONAL', 'OTHER'], 
                                   n_applications, p=[0.4, 0.2, 0.3, 0.1])
})

# Add some missing values
missing_features = ['debt_to_income', 'employment_years', 'num_delinquencies']
for feature in missing_features:
    missing_mask = np.random.random(n_applications) < 0.1
    application_data.loc[missing_mask, feature] = np.nan

# Create approval decision (correlated with features)
approval_score = (
    0.005 * application_data['bureau_score'] +
    -0.1 * application_data['num_delinquencies'].fillna(2) +
    0.00001 * application_data['annual_income'] +
    -2 * application_data['debt_to_income'].fillna(0.5) +
    0.05 * application_data['employment_years'].fillna(0) +
    np.random.normal(0, 0.5, n_applications)
)

application_data['approved'] = (approval_score > np.percentile(approval_score, (1-approval_rate)*100)).astype(int)

print(f"Total applications: {n_applications}")
print(f"Approval rate: {application_data['approved'].mean():.1%}")
print(f"Features with missing values: {application_data.isnull().sum().sum()}")

# Split into approved and rejected
approved_data = application_data[application_data['approved'] == 1].copy()
rejected_data = application_data[application_data['approved'] == 0].copy()

print(f"\nApproved applications: {len(approved_data)}")
print(f"Rejected applications: {len(rejected_data)}")
#endregion

#region Step 2: Performance Analysis and Target Definition
print("\n\nStep 2: Performance Analysis and Target Definition")
print("-" * 50)

# Simulate loan performance data
performance_data = []

for _, loan in approved_data.iterrows():
    loan_id = loan['application_id']
    
    # Simulate 24 months of performance
    # Risk score based on features
    risk_score = (
        -0.01 * loan['bureau_score'] +
        0.5 * loan['num_delinquencies'] +
        -0.00001 * loan['annual_income'] +
        3 * loan['debt_to_income'] +
        -0.02 * loan['employment_years'] +
        np.random.normal(0, 1)
    )
    
    # Probability of going bad
    bad_prob = 1 / (1 + np.exp(-risk_score))
    will_default = np.random.random() < bad_prob
    
    current_status = 'CURRENT'
    for month in range(1, 25):
        # Simple state transition
        if will_default and month > 6:
            if current_status == 'CURRENT' and np.random.random() < 0.3:
                current_status = '30-59DPD'
            elif current_status == '30-59DPD' and np.random.random() < 0.5:
                current_status = '60-89DPD'
            elif current_status == '60-89DPD' and np.random.random() < 0.7:
                current_status = '90+DPD'
        
        performance_data.append({
            'loan_id': loan_id,
            'month_on_book': month,
            'status': current_status
        })

performance_df = pd.DataFrame(performance_data)

# Define target based on 90+ DPD in first 12 months
target_definition = {}
for loan_id in approved_data['application_id']:
    loan_perf = performance_df[
        (performance_df['loan_id'] == loan_id) & 
        (performance_df['month_on_book'] <= 12)
    ]
    
    # Check if ever reached 90+ DPD
    went_bad = (loan_perf['status'] == '90+DPD').any()
    target_definition[loan_id] = 1 if went_bad else 0

approved_data['target'] = approved_data['application_id'].map(target_definition)

print(f"Target variable defined based on 90+ DPD in first 12 months")
print(f"Bad rate in approved population: {approved_data['target'].mean():.2%}")
#endregion

#region Step 3: Reject Inference
print("\n\nStep 3: Reject Inference")
print("-" * 50)

# Use bureau score for performance inference
from _fuzzy_augmentation import FuzzyAugmentation

# Features for reject inference
ri_features = ['bureau_score', 'num_tradelines', 'num_delinquencies', 
               'annual_income', 'debt_to_income', 'employment_years']

# Handle missing values for reject inference
for feature in ri_features:
    median_value = approved_data[feature].median()
    approved_data[feature] = approved_data[feature].fillna(median_value)
    rejected_data[feature] = rejected_data[feature].fillna(median_value)

# Apply fuzzy augmentation
fa = FuzzyAugmentation(n_iterations=3)
X_approved = approved_data[ri_features]
y_approved = approved_data['target']
X_rejected = rejected_data[ri_features]

# Fit model on approved
fa.fit(X_approved, y_approved)

# Get probabilities for rejected
reject_probs = fa.predict_proba(X_rejected)[:, 1]

# Create augmented dataset
augmented_data = approved_data.copy()
augmented_data['weight'] = 1.0

# Add rejected with inferred performance
for idx, (_, reject) in enumerate(rejected_data.iterrows()):
    # Good record
    good_record = reject.copy()
    good_record['target'] = 0
    good_record['weight'] = 1 - reject_probs[idx]
    augmented_data = pd.concat([augmented_data, pd.DataFrame([good_record])], ignore_index=True)
    
    # Bad record
    bad_record = reject.copy()
    bad_record['target'] = 1
    bad_record['weight'] = reject_probs[idx]
    augmented_data = pd.concat([augmented_data, pd.DataFrame([bad_record])], ignore_index=True)

print(f"Augmented dataset size: {len(augmented_data)}")
print(f"Weighted bad rate: {(augmented_data['target'] * augmented_data['weight']).sum() / augmented_data['weight'].sum():.2%}")
#endregion

#region Step 4: Feature Engineering with WOE
print("\n\nStep 4: Feature Engineering with WOE")
print("-" * 50)

from woe import WOE_Transform

# Define modeling features
modeling_features = [
    'bureau_score', 'num_tradelines', 'oldest_tradeline_months',
    'num_inquiries_6mo', 'num_delinquencies', 'annual_income',
    'debt_to_income', 'employment_years', 'age', 'home_ownership',
    'loan_amount', 'loan_purpose'
]

# Initialize WOE transformer
woe_transformer = WOE_Transform(
    method='tree',
    num_bin_start=20,
    min_samples_leaf=int(0.05 * augmented_data['weight'].sum()),
    min_iv=0.02
)

# Fit WOE with weights
X = augmented_data[modeling_features]
y = augmented_data['target']
weights = augmented_data['weight']

print("Fitting WOE transformation...")
woe_transformer.fit(X, y, Y_weight=weights, display=5)

# Transform features
X_woe = woe_transformer.transform(X)

# Get Information Values
iv_df = woe_transformer.get_iv().sort_values('iv', ascending=False)
print(f"\nTop features by Information Value:")
print(iv_df.head(10))

# Select features with IV > 0.02
selected_features = iv_df[iv_df['iv'] > 0.02]['attr'].tolist()
selected_features_woe = [f + '_woe' for f in selected_features]
print(f"\nSelected {len(selected_features)} features with IV > 0.02")
#endregion

#region Step 5: Model Development
print("\n\nStep 5: Model Development")
print("-" * 50)

# Split data for training and validation
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_woe[selected_features_woe], y, weights,
    test_size=0.3, random_state=42, stratify=y
)

# Build logistic regression scorecard
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'
)

# Fit model
lr_model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate model
from sklearn.metrics import roc_auc_score, classification_report

y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba, sample_weight=w_test)

print(f"Model Performance:")
print(f"- Test AUC: {auc_score:.4f}")

# Create scorecard
base_score = 600
pdo = 20  # Points to double the odds

factor = pdo / np.log(2)
offset = base_score - factor * np.log(1)

# Get feature points
feature_points = pd.DataFrame({
    'feature': selected_features_woe,
    'coefficient': lr_model.coef_[0],
    'points': -lr_model.coef_[0] * factor
})

intercept_points = -lr_model.intercept_[0] * factor + offset

print(f"\nScorecard Parameters:")
print(f"- Base score: {base_score}")
print(f"- PDO: {pdo}")
print(f"- Intercept points: {intercept_points:.2f}")
#endregion

#region Step 6: Score Calculation and Alignment
print("\n\nStep 6: Score Calculation and Alignment")
print("-" * 50)

# Calculate scores
def calculate_score(X_woe_data):
    points = intercept_points
    for idx, row in X_woe_data.iterrows():
        for feature in selected_features_woe:
            if feature in feature_points['feature'].values:
                feature_info = feature_points[feature_points['feature'] == feature].iloc[0]
                points += row[feature] * feature_info['points']
    return points

# Calculate scores for test set
test_scores = []
for idx, row in X_test.iterrows():
    score = intercept_points
    for feature in selected_features_woe:
        if feature in feature_points['feature'].values:
            feature_info = feature_points[feature_points['feature'] == feature].iloc[0]
            score += row[feature] * feature_info['points']
    test_scores.append(score)

test_scores = np.array(test_scores)

print(f"Score distribution:")
print(f"- Mean: {test_scores.mean():.1f}")
print(f"- Std: {test_scores.std():.1f}")
print(f"- Range: {test_scores.min():.1f} - {test_scores.max():.1f}")

# Create score bands and analyze
score_bands = pd.qcut(test_scores, q=10, duplicates='drop')
band_analysis = pd.DataFrame({
    'score_band': score_bands,
    'actual_bad': y_test,
    'weight': w_test
}).groupby('score_band').agg({
    'actual_bad': lambda x: (x * w_test.iloc[x.index]).sum() / w_test.iloc[x.index].sum(),
    'weight': 'sum'
}).reset_index()

band_analysis.columns = ['score_band', 'bad_rate', 'count']
print("\nBad rate by score band:")
print(band_analysis)
#endregion

#region Step 7: Model Deployment Preparation
print("\n\nStep 7: Model Deployment Preparation")
print("-" * 50)

# Save model artifacts
model_artifacts = {
    'model': lr_model,
    'woe_transformer': woe_transformer,
    'selected_features': selected_features,
    'scorecard_params': {
        'base_score': base_score,
        'pdo': pdo,
        'factor': factor,
        'offset': offset,
        'intercept_points': intercept_points
    },
    'feature_points': feature_points.to_dict(),
    'training_date': datetime.now().strftime('%Y-%m-%d'),
    'model_performance': {
        'auc': auc_score,
        'bad_rate': y_train.mean()
    }
}

# Save model
joblib.dump(model_artifacts, 'credit_risk_model.pkl')
print("Model artifacts saved to: credit_risk_model.pkl")

# Save WOE specifications for production
woe_json = woe_transformer.woe_json(selected_features)
with open('woe_specifications.json', 'w') as f:
    json.dump(json.loads(woe_json), f, indent=2)
print("WOE specifications saved to: woe_specifications.json")
#endregion

#region Step 8: Monitoring Setup
print("\n\nStep 8: Monitoring Setup")
print("-" * 50)

# Create monitoring configuration
monitoring_config = {
    'model_name': 'credit_risk_scorecard_v1',
    'deployment_date': datetime.now().strftime('%Y-%m-%d'),
    'thresholds': {
        'psi_warning': 0.1,
        'psi_critical': 0.25,
        'auc_degradation_warning': 0.03,
        'auc_degradation_critical': 0.05,
        'score_shift_warning': 20,
        'score_shift_critical': 50
    },
    'monitoring_frequency': 'daily',
    'features_to_monitor': selected_features,
    'expected_metrics': {
        'mean_score': test_scores.mean(),
        'std_score': test_scores.std(),
        'bad_rate': y_test.mean(),
        'auc': auc_score
    }
}

with open('monitoring_config.json', 'w') as f:
    json.dump(monitoring_config, f, indent=2)
print("Monitoring configuration saved to: monitoring_config.json")
#endregion

#region Final Summary and Visualization
print("\n\n" + "=" * 60)
print("MODEL DEVELOPMENT SUMMARY")
print("=" * 60)
print(f"\nData:")
print(f"- Total applications: {n_applications}")
print(f"- Approved (observed): {len(approved_data)} ({len(approved_data)/n_applications:.1%})")
print(f"- Rejected (inferred): {len(rejected_data)} ({len(rejected_data)/n_applications:.1%})")
print(f"- Augmented dataset: {len(augmented_data)} records")

print(f"\nFeatures:")
print(f"- Initial features: {len(modeling_features)}")
print(f"- Selected features (IV > 0.02): {len(selected_features)}")
print(f"- Top 3 features: {', '.join(selected_features[:3])}")

print(f"\nModel Performance:")
print(f"- AUC: {auc_score:.4f}")
print(f"- Score range: {test_scores.min():.0f} - {test_scores.max():.0f}")
print(f"- Mean score: {test_scores.mean():.1f}")

print(f"\nDeployment Artifacts:")
print(f"- Model file: credit_risk_model.pkl")
print(f"- WOE specifications: woe_specifications.json")
print(f"- Monitoring config: monitoring_config.json")

print("\n✅ Credit risk model development completed successfully!")

# Create visualization summary
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Score distribution
axes[0, 0].hist(test_scores, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(test_scores.mean(), color='red', linestyle='--', label=f'Mean: {test_scores.mean():.1f}')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Score Distribution')
axes[0, 0].legend()

# 2. ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, sample_weight=w_test)
axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()

# 3. Bad rate by score band
axes[1, 0].bar(range(len(band_analysis)), band_analysis['bad_rate'])
axes[1, 0].set_xlabel('Score Band (Low to High)')
axes[1, 0].set_ylabel('Bad Rate')
axes[1, 0].set_title('Bad Rate by Score Band')
axes[1, 0].set_xticks(range(len(band_analysis)))
axes[1, 0].set_xticklabels([f'B{i+1}' for i in range(len(band_analysis))])

# 4. Feature importance (top 10)
top_features = feature_points.nlargest(10, 'points')[['feature', 'points']]
axes[1, 1].barh(range(len(top_features)), top_features['points'])
axes[1, 1].set_yticks(range(len(top_features)))
axes[1, 1].set_yticklabels([f.replace('_woe', '') for f in top_features['feature']])
axes[1, 1].set_xlabel('Points Contribution')
axes[1, 1].set_title('Top 10 Features by Points')

plt.suptitle('Credit Risk Model Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel summary visualization saved to: model_summary.png")
#endregion