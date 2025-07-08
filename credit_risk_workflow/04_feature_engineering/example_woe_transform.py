"""
Example: Feature Engineering with Weight of Evidence (WOE)

This script demonstrates how to transform features using WOE for
credit risk modeling, including binning, monotonicity enforcement,
and missing value imputation.

Inputs:
- Feature data (numerical and categorical)
- Target variable (Good=0, Bad=1)
- Sample weights (from reject inference if applicable)

Outputs:
- WOE-transformed features
- Information Value (IV) for feature selection
- Binning specifications for production scoring
"""

import pandas as pd
import numpy as np
from woe import WOE_Transform
from imputer import WOEImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Example 1: Basic WOE Transformation
def basic_woe_transformation(df, features, target, display_plots=True):
    """
    Apply basic WOE transformation to features
    
    Args:
        df: DataFrame with features and target
        features: List of feature columns
        target: Target column name
        display_plots: Whether to show WOE plots
        
    Returns:
        woe_df: WOE-transformed features
        woe_transformer: Fitted WOE transformer
        iv_df: Information Value for each feature
    """
    # Separate features and target
    X = df[features]
    y = df[target]
    
    # Initialize WOE transformer
    woe_transformer = WOE_Transform(
        method='tree',  # Use decision tree for binning
        num_bin_start=20,  # Initial number of bins
        min_samples_leaf=int(0.05 * len(df)),  # Min 5% of data per bin
        min_iv=0.01  # Minimum IV to keep feature
    )
    
    # Fit and transform
    print("Fitting WOE transformation...")
    woe_transformer.fit(X, y, display=5)  # Display top 5 features
    
    # Transform features
    woe_df = woe_transformer.transform(X)
    
    # Get Information Values
    iv_df = woe_transformer.get_iv().sort_values('iv', ascending=False)
    
    print(f"\nInformation Value Summary:")
    print(f"Features with IV > 0.3 (Strong): {len(iv_df[iv_df['iv'] > 0.3])}")
    print(f"Features with IV 0.1-0.3 (Medium): {len(iv_df[(iv_df['iv'] >= 0.1) & (iv_df['iv'] <= 0.3)])}")
    print(f"Features with IV 0.02-0.1 (Weak): {len(iv_df[(iv_df['iv'] >= 0.02) & (iv_df['iv'] < 0.1)])}")
    print(f"Features with IV < 0.02 (Not useful): {len(iv_df[iv_df['iv'] < 0.02])}")
    
    if display_plots:
        # Plot top features by IV
        plt.figure(figsize=(10, 6))
        top_features = iv_df.head(20)
        plt.barh(range(len(top_features)), top_features['iv'])
        plt.yticks(range(len(top_features)), top_features['attr'])
        plt.xlabel('Information Value')
        plt.title('Top 20 Features by Information Value')
        plt.tight_layout()
        plt.show()
        
        # Plot WOE patterns for top features
        plot_woe_patterns(woe_transformer, iv_df.head(6)['attr'].tolist())
    
    return woe_df, woe_transformer, iv_df


# Example 2: WOE Transformation with Sample Weights
def weighted_woe_transformation(df, features, target, weights):
    """
    Apply WOE transformation with sample weights (e.g., from reject inference)
    
    Args:
        df: DataFrame with features and target
        features: List of feature columns
        target: Target column name
        weights: Sample weights column name
        
    Returns:
        woe_df: WOE-transformed features
        woe_transformer: Fitted WOE transformer
    """
    X = df[features]
    y = df[target]
    w = df[weights]
    
    # Initialize WOE transformer
    woe_transformer = WOE_Transform(
        method='tree',
        num_bin_start=20,
        min_samples_leaf=int(0.05 * w.sum()),  # Min 5% of weighted sample
        min_iv=0.01
    )
    
    # Fit with weights
    print("Fitting weighted WOE transformation...")
    woe_transformer.fit(X, y, Y_weight=w, display=3)
    
    # Transform
    woe_df = woe_transformer.transform(X)
    
    # Compare weighted vs unweighted bad rates
    print(f"\nWeighted vs Unweighted Statistics:")
    print(f"Unweighted bad rate: {y.mean():.2%}")
    print(f"Weighted bad rate: {(y * w).sum() / w.sum():.2%}")
    
    return woe_df, woe_transformer


# Example 3: Handle Special Values
def woe_with_special_values(df, features, target, special_values=[-999, -998, -997]):
    """
    Handle special values separately in WOE transformation
    
    Args:
        df: DataFrame with features and target
        features: List of feature columns
        target: Target column name
        special_values: List of special values to handle separately
        
    Returns:
        woe_df: WOE-transformed features
        woe_transformer: Fitted WOE transformer
    """
    X = df[features]
    y = df[target]
    
    # Initialize WOE transformer
    woe_transformer = WOE_Transform(
        method='tree',
        num_bin_start=20,
        min_samples_leaf=int(0.05 * len(df)),
        min_iv=0.01
    )
    
    # Fit with special values
    print(f"Fitting WOE with special values: {special_values}")
    woe_transformer.fit(X, y, special_value=special_values, display=3)
    
    # Transform
    woe_df = woe_transformer.transform(X)
    
    # Show how special values were handled
    for feature in features[:3]:  # Show first 3 features
        if feature in woe_transformer.woe:
            woe_detail = woe_transformer.iv_detail.get(feature)
            if woe_detail is not None:
                special_rows = woe_detail[woe_detail.index == 'special']
                if not special_rows.empty:
                    print(f"\nSpecial values handling for {feature}:")
                    print(special_rows[['#accts', '%accts', 'target_rate', 'woe']])
    
    return woe_df, woe_transformer


# Example 4: WOE-based Imputation
def woe_based_imputation(df, features, target, woe_transformer):
    """
    Impute missing values using WOE relationships
    
    Args:
        df: DataFrame with features and target
        features: List of feature columns
        target: Target column name
        woe_transformer: Fitted WOE transformer
        
    Returns:
        imputed_df: DataFrame with imputed values
        imputer: Fitted WOE imputer
    """
    X = df[features].copy()
    y = df[target]
    
    # Check missing values
    missing_summary = X.isnull().sum()
    print("Missing values before imputation:")
    print(missing_summary[missing_summary > 0])
    
    # Initialize WOE imputer
    imputer = WOEImputer(
        woe_transformer=woe_transformer,
        imputation_method='closest_bin',  # or 'midpoint'
        target_column=target
    )
    
    # Fit and transform
    imputer.fit(X, y)
    X_imputed = imputer.transform(X)
    
    # Verify no missing values
    print("\nMissing values after imputation:")
    print(X_imputed.isnull().sum().sum())
    
    # Compare distributions before and after imputation
    for feature in features[:3]:  # Show first 3 features
        if missing_summary[feature] > 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            X[feature].hist(bins=30, alpha=0.7, label='Original')
            X_imputed[feature].hist(bins=30, alpha=0.7, label='After Imputation')
            plt.xlabel(feature)
            plt.legend()
            plt.title(f'{feature}: Distribution Comparison')
            
            plt.subplot(1, 2, 2)
            # Compare bad rates
            original_bad_rate = y[X[feature].notna()].mean()
            imputed_bad_rate = y[X[feature].isna()].mean() if X[feature].isna().any() else 0
            overall_bad_rate = y.mean()
            
            plt.bar(['Non-missing', 'Missing', 'Overall'], 
                   [original_bad_rate, imputed_bad_rate, overall_bad_rate])
            plt.ylabel('Bad Rate')
            plt.title(f'{feature}: Bad Rate Comparison')
            
            plt.tight_layout()
            plt.show()
    
    return X_imputed, imputer


# Example 5: Feature Selection based on IV
def select_features_by_iv(iv_df, min_iv=0.02, max_features=30, exclude_correlated=True):
    """
    Select features based on Information Value
    
    Args:
        iv_df: DataFrame with Information Values
        min_iv: Minimum IV threshold
        max_features: Maximum number of features to select
        exclude_correlated: Whether to exclude highly correlated features
        
    Returns:
        selected_features: List of selected feature names
    """
    # Filter by minimum IV
    qualified_features = iv_df[iv_df['iv'] >= min_iv].copy()
    
    print(f"Features with IV >= {min_iv}: {len(qualified_features)}")
    
    # Sort by IV and limit to max_features
    qualified_features = qualified_features.head(max_features)
    
    selected_features = qualified_features['attr'].tolist()
    
    # Create selection report
    print(f"\nFeature Selection Summary:")
    print(f"- Total features: {len(iv_df)}")
    print(f"- Features above IV threshold: {len(iv_df[iv_df['iv'] >= min_iv])}")
    print(f"- Features selected: {len(selected_features)}")
    print(f"- Top 5 features by IV:")
    for idx, row in qualified_features.head(5).iterrows():
        print(f"  {row['attr']}: IV = {row['iv']:.4f}")
    
    return selected_features


# Example 6: Save WOE specifications for production
def save_woe_for_production(woe_transformer, selected_features, output_path='woe_spec.json'):
    """
    Save WOE binning specifications for production scoring
    
    Args:
        woe_transformer: Fitted WOE transformer
        selected_features: List of features to save
        output_path: Path to save JSON file
    """
    # Generate WOE JSON
    woe_json = woe_transformer.woe_json(selected_features)
    
    # Parse and enhance with metadata
    woe_spec = json.loads(woe_json)
    
    # Add metadata
    metadata = {
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_features': len(selected_features),
        'min_samples_leaf': woe_transformer.min_samples_leaf,
        'binning_method': woe_transformer.method,
        'features': selected_features
    }
    
    final_spec = {
        'metadata': metadata,
        'woe_specifications': woe_spec
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(final_spec, f, indent=2)
    
    print(f"WOE specifications saved to: {output_path}")
    print(f"Features included: {len(selected_features)}")
    
    return final_spec


# Helper function to plot WOE patterns
def plot_woe_patterns(woe_transformer, features):
    """Plot WOE patterns for specified features"""
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        if feature in woe_transformer.woe:
            woe_detail = woe_transformer.woe[feature].reset_index()
            
            if 'min' in woe_detail.columns:  # Numerical feature
                # Plot WOE by bin
                x = range(len(woe_detail))
                y = woe_detail['woe']
                
                ax.plot(x, y, marker='o', linewidth=2, markersize=8)
                ax.set_xticks(x)
                ax.set_xticklabels([f"{row['min']:.1f}-{row['max']:.1f}" 
                                   if pd.notna(row['min']) else row['index'] 
                                   for _, row in woe_detail.iterrows()], 
                                  rotation=45, ha='right')
            else:  # Categorical feature
                # Bar plot for categories
                woe_detail = woe_detail.head(10)  # Limit to top 10 categories
                ax.bar(range(len(woe_detail)), woe_detail['woe'])
                ax.set_xticks(range(len(woe_detail)))
                ax.set_xticklabels(woe_detail[feature], rotation=45, ha='right')
            
            ax.set_xlabel('Bin/Category')
            ax.set_ylabel('WOE')
            ax.set_title(f'{feature} (IV={woe_transformer.iv.get(feature, 0):.3f})')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Create features with different relationships to target
    data = pd.DataFrame({
        # Strong predictors
        'bureau_score': np.random.normal(650, 100, n_samples).clip(300, 850),
        'payment_history': np.random.choice(['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'VERY_POOR'], 
                                          n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        
        # Medium predictors  
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        
        # Weak predictors
        'num_inquiries': np.random.poisson(2, n_samples),
        'age': np.random.normal(40, 12, n_samples).clip(18, 80),
        
        # Features with missing values
        'revolving_balance': np.where(np.random.random(n_samples) < 0.1, np.nan, 
                                     np.random.lognormal(8, 1.5, n_samples)),
        'months_since_delinquency': np.where(np.random.random(n_samples) < 0.7, np.nan,
                                           np.random.exponential(20, n_samples))
    })
    
    # Create target correlated with features
    risk_score = (
        -0.01 * data['bureau_score'] +
        data['payment_history'].map({'EXCELLENT': -2, 'GOOD': -1, 'FAIR': 0, 'POOR': 1, 'VERY_POOR': 2}) +
        -0.00001 * data['income'] +
        3 * data['debt_to_income'] +
        0.1 * data['num_inquiries'] +
        np.random.normal(0, 1, n_samples)
    )
    
    data['target'] = (risk_score > np.percentile(risk_score, 80)).astype(int)
    
    # Add some special values
    data.loc[np.random.choice(n_samples, 100), 'income'] = -999
    
    print("=== WOE Feature Engineering Example ===\n")
    print(f"Dataset shape: {data.shape}")
    print(f"Bad rate: {data['target'].mean():.2%}")
    print(f"Features with missing values: {data.isnull().sum().sum()}")
    
    # Define features
    features = [col for col in data.columns if col != 'target']
    
    # Example 1: Basic WOE transformation
    print("\n1. Basic WOE Transformation")
    woe_df, woe_transformer, iv_df = basic_woe_transformation(data, features, 'target')
    
    # Example 2: Feature selection
    print("\n2. Feature Selection by IV")
    selected_features = select_features_by_iv(iv_df, min_iv=0.02, max_features=20)
    
    # Example 3: WOE with special values
    print("\n3. WOE with Special Values")
    woe_special_df, woe_special = woe_with_special_values(data, features, 'target', [-999])
    
    # Example 4: Save for production
    print("\n4. Save WOE Specifications")
    woe_spec = save_woe_for_production(woe_transformer, selected_features[:10])
    
    print("\nFeature engineering complete!")