"""
Example: Reject Inference for Credit Risk Modeling

This script demonstrates various reject inference techniques to handle
selection bias from rejected loan applications.

Inputs:
- Approved loans with known performance (Good/Bad)
- Rejected applications without performance
- External data (e.g., credit scores)

Outputs:
- Augmented dataset with inferred reject performance
- Population weights for unbiased modeling
"""

import pandas as pd
import numpy as np
from _fuzzy_augmentation import FuzzyAugmentation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Example 1: Simple Hard Cutoff Method
def hard_cutoff_inference(approved_df, rejected_df, score_col='credit_score', cutoff_percentile=20):
    """
    Simple reject inference: assign worst X% of rejects as bad
    
    Args:
        approved_df: DataFrame with approved loans and performance
        rejected_df: DataFrame with rejected applications
        score_col: External score column (e.g., credit bureau score)
        cutoff_percentile: Percentile of rejects to assign as bad
        
    Returns:
        combined_df: Combined dataset with inferred performance
    """
    # Calculate cutoff based on approved bads
    bad_scores = approved_df[approved_df['target'] == 1][score_col]
    score_cutoff = np.percentile(bad_scores, cutoff_percentile)
    
    # Assign performance to rejects
    rejected_df = rejected_df.copy()
    rejected_df['target'] = (rejected_df[score_col] <= score_cutoff).astype(int)
    rejected_df['weight'] = 1.0
    rejected_df['inference_method'] = 'hard_cutoff'
    
    # Combine datasets
    approved_df = approved_df.copy()
    approved_df['weight'] = 1.0
    approved_df['inference_method'] = 'observed'
    
    combined_df = pd.concat([approved_df, rejected_df], ignore_index=True)
    
    print(f"Hard Cutoff Inference Results:")
    print(f"- Score cutoff: {score_cutoff:.0f}")
    print(f"- Approved bad rate: {approved_df['target'].mean():.2%}")
    print(f"- Inferred reject bad rate: {rejected_df['target'].mean():.2%}")
    print(f"- Overall bad rate: {combined_df['target'].mean():.2%}")
    
    return combined_df


# Example 2: Fuzzy Augmentation Method
def fuzzy_augmentation_inference(approved_df, rejected_df, features, n_iterations=5):
    """
    Fuzzy augmentation: create weighted good/bad records for rejects
    
    Args:
        approved_df: DataFrame with approved loans and performance
        rejected_df: DataFrame with rejected applications
        features: List of feature columns to use
        n_iterations: Number of iterations for convergence
        
    Returns:
        augmented_df: Dataset with fuzzy augmented records
    """
    # Initialize fuzzy augmentation
    fa = FuzzyAugmentation(
        base_estimator=LogisticRegression(max_iter=1000),
        n_iterations=n_iterations,
        convergence_threshold=0.001
    )
    
    # Prepare data
    X_approved = approved_df[features]
    y_approved = approved_df['target']
    X_rejected = rejected_df[features]
    
    # Fit and transform
    fa.fit(X_approved, y_approved)
    
    # Get probabilities for rejected
    reject_probs = fa.predict_proba(X_rejected)[:, 1]
    
    # Create augmented records
    augmented_records = []
    
    for idx, prob in enumerate(reject_probs):
        # Create good record
        good_record = rejected_df.iloc[idx].copy()
        good_record['target'] = 0
        good_record['weight'] = 1 - prob
        good_record['inference_method'] = 'fuzzy_good'
        augmented_records.append(good_record)
        
        # Create bad record
        bad_record = rejected_df.iloc[idx].copy()
        bad_record['target'] = 1
        bad_record['weight'] = prob
        bad_record['inference_method'] = 'fuzzy_bad'
        augmented_records.append(bad_record)
    
    # Combine with approved
    approved_df = approved_df.copy()
    approved_df['weight'] = 1.0
    approved_df['inference_method'] = 'observed'
    
    augmented_df = pd.concat([approved_df, pd.DataFrame(augmented_records)], ignore_index=True)
    
    print(f"\nFuzzy Augmentation Results:")
    print(f"- Approved records: {len(approved_df)}")
    print(f"- Rejected records: {len(rejected_df)}")
    print(f"- Augmented records: {len(augmented_df)}")
    print(f"- Weighted bad rate: {(augmented_df['target'] * augmented_df['weight']).sum() / augmented_df['weight'].sum():.2%}")
    
    return augmented_df


# Example 3: Performance Scoring Method
def performance_scoring_inference(approved_df, rejected_df, external_score='bureau_score'):
    """
    Use external credit bureau scores to infer performance
    
    Args:
        approved_df: DataFrame with approved loans and performance
        rejected_df: DataFrame with rejected applications
        external_score: Column name for external score
        
    Returns:
        combined_df: Combined dataset with inferred performance
    """
    # Build relationship between external score and performance
    # Group approved by score bands
    approved_df['score_band'] = pd.qcut(approved_df[external_score], q=10, duplicates='drop')
    
    # Calculate bad rate by score band
    band_performance = approved_df.groupby('score_band').agg({
        'target': ['mean', 'count']
    }).reset_index()
    band_performance.columns = ['score_band', 'bad_rate', 'count']
    
    # Apply to rejected
    rejected_df = rejected_df.copy()
    rejected_df['score_band'] = pd.cut(
        rejected_df[external_score],
        bins=approved_df.groupby('score_band')[external_score].min().values.tolist() + [float('inf')],
        labels=band_performance['score_band'].unique()
    )
    
    # Merge bad rates
    rejected_df = rejected_df.merge(
        band_performance[['score_band', 'bad_rate']],
        on='score_band',
        how='left'
    )
    
    # Create weighted records based on bad rate
    augmented_records = []
    
    for idx, row in rejected_df.iterrows():
        bad_rate = row['bad_rate']
        
        # Create good record
        good_record = row.copy()
        good_record['target'] = 0
        good_record['weight'] = 1 - bad_rate
        good_record['inference_method'] = 'score_based_good'
        augmented_records.append(good_record)
        
        # Create bad record
        bad_record = row.copy()
        bad_record['target'] = 1
        bad_record['weight'] = bad_rate
        bad_record['inference_method'] = 'score_based_bad'
        augmented_records.append(bad_record)
    
    # Combine with approved
    approved_df = approved_df.copy()
    approved_df['weight'] = 1.0
    approved_df['inference_method'] = 'observed'
    
    combined_df = pd.concat([approved_df, pd.DataFrame(augmented_records)], ignore_index=True)
    
    print(f"\nPerformance Scoring Results:")
    print(f"- Score bands used: {len(band_performance)}")
    print(f"- Weighted reject bad rate: {(rejected_df['bad_rate']).mean():.2%}")
    
    return combined_df


# Example 4: Augmentation Method (Reweighting)
def augmentation_reweighting(approved_df, rejected_df, features):
    """
    Reweight approved population to represent full through-the-door population
    
    Args:
        approved_df: DataFrame with approved loans and performance
        rejected_df: DataFrame with rejected applications
        features: List of feature columns to use
        
    Returns:
        reweighted_df: Approved data with weights to represent full population
    """
    # Combine for full population
    approved_df = approved_df.copy()
    rejected_df = rejected_df.copy()
    
    approved_df['is_approved'] = 1
    rejected_df['is_approved'] = 0
    
    combined = pd.concat([approved_df[features + ['is_approved']], 
                         rejected_df[features + ['is_approved']]], 
                         ignore_index=True)
    
    # Build approval model
    approval_model = LogisticRegression(max_iter=1000)
    X = combined[features]
    y = combined['is_approved']
    
    approval_model.fit(X, y)
    
    # Calculate approval probability for approved loans
    approval_probs = approval_model.predict_proba(approved_df[features])[:, 1]
    
    # Calculate weights (inverse of approval probability)
    approved_df['weight'] = 1 / approval_probs
    
    # Normalize weights
    approved_df['weight'] = approved_df['weight'] / approved_df['weight'].mean()
    
    print(f"\nAugmentation Reweighting Results:")
    print(f"- Min weight: {approved_df['weight'].min():.2f}")
    print(f"- Max weight: {approved_df['weight'].max():.2f}")
    print(f"- Weighted bad rate: {(approved_df['target'] * approved_df['weight']).sum() / approved_df['weight'].sum():.2%}")
    
    return approved_df


# Example 5: Compare inference methods
def compare_inference_methods(approved_df, rejected_df, features, external_score='bureau_score'):
    """
    Compare different reject inference methods
    
    Args:
        approved_df: DataFrame with approved loans
        rejected_df: DataFrame with rejected applications
        features: List of features for modeling
        external_score: External score column
        
    Returns:
        comparison_df: DataFrame comparing methods
    """
    results = []
    
    # No inference (approved only)
    results.append({
        'method': 'No Inference',
        'sample_size': len(approved_df),
        'bad_rate': approved_df['target'].mean(),
        'approval_rate': len(approved_df) / (len(approved_df) + len(rejected_df))
    })
    
    # Hard cutoff
    hard_cutoff_df = hard_cutoff_inference(approved_df, rejected_df, external_score)
    results.append({
        'method': 'Hard Cutoff',
        'sample_size': len(hard_cutoff_df),
        'bad_rate': hard_cutoff_df['target'].mean(),
        'approval_rate': 1.0  # Using full population
    })
    
    # Fuzzy augmentation
    fuzzy_df = fuzzy_augmentation_inference(approved_df, rejected_df, features)
    weighted_bad_rate = (fuzzy_df['target'] * fuzzy_df['weight']).sum() / fuzzy_df['weight'].sum()
    results.append({
        'method': 'Fuzzy Augmentation',
        'sample_size': len(fuzzy_df),
        'bad_rate': weighted_bad_rate,
        'approval_rate': 1.0
    })
    
    # Performance scoring
    perf_score_df = performance_scoring_inference(approved_df, rejected_df, external_score)
    weighted_bad_rate = (perf_score_df['target'] * perf_score_df['weight']).sum() / perf_score_df['weight'].sum()
    results.append({
        'method': 'Performance Scoring',
        'sample_size': len(perf_score_df),
        'bad_rate': weighted_bad_rate,
        'approval_rate': 1.0
    })
    
    # Augmentation reweighting
    reweight_df = augmentation_reweighting(approved_df, rejected_df, features)
    weighted_bad_rate = (reweight_df['target'] * reweight_df['weight']).sum() / reweight_df['weight'].sum()
    results.append({
        'method': 'Augmentation Reweighting',
        'sample_size': len(reweight_df),
        'bad_rate': weighted_bad_rate,
        'approval_rate': len(approved_df) / (len(approved_df) + len(rejected_df))
    })
    
    comparison_df = pd.DataFrame(results)
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bad rates
    comparison_df.plot(x='method', y='bad_rate', kind='bar', ax=ax1, legend=False)
    ax1.set_title('Bad Rate by Inference Method')
    ax1.set_ylabel('Bad Rate')
    ax1.set_xticklabels(comparison_df['method'], rotation=45, ha='right')
    
    # Sample sizes
    comparison_df.plot(x='method', y='sample_size', kind='bar', ax=ax2, legend=False, color='orange')
    ax2.set_title('Sample Size by Inference Method')
    ax2.set_ylabel('Sample Size')
    ax2.set_xticklabels(comparison_df['method'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_applications = 10000
    approval_rate = 0.6
    
    # Create features
    data = pd.DataFrame({
        'bureau_score': np.random.normal(650, 100, n_applications).clip(300, 850),
        'income': np.random.lognormal(10.5, 0.5, n_applications),
        'debt_to_income': np.random.beta(2, 5, n_applications),
        'employment_years': np.random.exponential(5, n_applications).clip(0, 40),
        'num_tradelines': np.random.poisson(5, n_applications)
    })
    
    # Create approval decision (correlated with features)
    approval_score = (
        0.005 * data['bureau_score'] +
        0.00001 * data['income'] -
        2 * data['debt_to_income'] +
        0.05 * data['employment_years'] +
        np.random.normal(0, 0.5, n_applications)
    )
    
    data['approved'] = (approval_score > np.percentile(approval_score, (1-approval_rate)*100)).astype(int)
    
    # Create performance for approved (correlated with features)
    performance_score = (
        -0.01 * data['bureau_score'] +
        -0.00001 * data['income'] +
        3 * data['debt_to_income'] -
        0.02 * data['employment_years'] +
        np.random.normal(0, 1, n_applications)
    )
    
    data.loc[data['approved'] == 1, 'target'] = (
        performance_score[data['approved'] == 1] > 
        np.percentile(performance_score[data['approved'] == 1], 80)
    ).astype(int)
    
    # Split into approved and rejected
    approved_df = data[data['approved'] == 1].drop('approved', axis=1).reset_index(drop=True)
    rejected_df = data[data['approved'] == 0].drop(['approved', 'target'], axis=1).reset_index(drop=True)
    
    print(f"Data Summary:")
    print(f"- Total applications: {n_applications}")
    print(f"- Approved: {len(approved_df)} ({len(approved_df)/n_applications:.1%})")
    print(f"- Rejected: {len(rejected_df)} ({len(rejected_df)/n_applications:.1%})")
    print(f"- Observed bad rate: {approved_df['target'].mean():.2%}")
    
    # Define features for modeling
    features = ['bureau_score', 'income', 'debt_to_income', 'employment_years', 'num_tradelines']
    
    # Compare inference methods
    print("\n=== Comparing Reject Inference Methods ===")
    comparison_df = compare_inference_methods(approved_df, rejected_df, features, 'bureau_score')
    print("\n", comparison_df)
    
    print("\nAnalysis complete!")