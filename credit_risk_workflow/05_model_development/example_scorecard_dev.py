"""
Example: Credit Scorecard Development

This script demonstrates how to develop credit scorecards using
WOE-transformed features, including traditional logistic regression
and modern machine learning approaches.

Inputs:
- WOE-transformed features
- Target variable (Good=0, Bad=1)
- Sample weights (if using reject inference)

Outputs:
- Trained scorecard model
- Model coefficients and scores
- Performance metrics and validation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Example 1: Traditional Logistic Regression Scorecard
def build_logistic_scorecard(X_woe, y, sample_weights=None, base_score=600, pdo=20):
    """
    Build traditional logistic regression scorecard
    
    Args:
        X_woe: WOE-transformed features
        y: Target variable
        sample_weights: Optional sample weights
        base_score: Base score at odds of 1:1
        pdo: Points to double the odds
        
    Returns:
        scorecard: Dictionary with scoring rules
        model: Fitted logistic regression model
        performance: Model performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_woe, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if sample_weights is not None:
        weights_train = sample_weights.iloc[X_train.index]
        weights_test = sample_weights.iloc[X_test.index]
    else:
        weights_train = None
        weights_test = None
    
    # Build logistic regression model
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'
    )
    
    # Fit model
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    auc_score = roc_auc_score(y_test, y_pred_proba, sample_weight=weights_test)
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_woe, y, cv=5, scoring='roc_auc', 
        fit_params={'sample_weight': sample_weights} if sample_weights is not None else None
    )
    
    print(f"Model Performance:")
    print(f"- Test AUC: {auc_score:.4f}")
    print(f"- CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create scorecard
    # Score = Offset + Factor * log(odds)
    # Factor = pdo / log(2)
    # Offset = base_score - Factor * log(odds_at_base_score)
    
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(1)  # log(1) = 0 for 1:1 odds
    
    # Get coefficients
    coefficients = pd.DataFrame({
        'feature': X_woe.columns,
        'coefficient': model.coef_[0],
        'points': -model.coef_[0] * factor
    }).sort_values('coefficient', ascending=False)
    
    # Add intercept
    intercept_points = -model.intercept_[0] * factor
    
    scorecard = {
        'base_score': base_score,
        'pdo': pdo,
        'intercept_points': intercept_points,
        'offset': offset,
        'factor': factor,
        'features': coefficients,
        'model': model
    }
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = coefficients.nlargest(10, 'coefficient')
    bottom_features = coefficients.nsmallest(10, 'coefficient')
    plot_features = pd.concat([top_features, bottom_features])
    
    plt.barh(range(len(plot_features)), plot_features['coefficient'])
    plt.yticks(range(len(plot_features)), plot_features['feature'])
    plt.xlabel('Coefficient')
    plt.title('Top Positive and Negative Scorecard Features')
    plt.tight_layout()
    plt.show()
    
    performance = {
        'auc_test': auc_score,
        'auc_cv_mean': cv_scores.mean(),
        'auc_cv_std': cv_scores.std()
    }
    
    return scorecard, model, performance


# Example 2: Create Score Mapping Table
def create_score_mapping(scorecard, woe_bins, features):
    """
    Create detailed score mapping for each feature value
    
    Args:
        scorecard: Scorecard dictionary from build_logistic_scorecard
        woe_bins: WOE binning information
        features: List of features to include
        
    Returns:
        score_mapping: DataFrame with score for each bin
    """
    score_mappings = []
    
    for feature in features:
        if feature not in woe_bins:
            continue
            
        # Get feature coefficient
        feature_info = scorecard['features'][scorecard['features']['feature'] == feature]
        if len(feature_info) == 0:
            continue
            
        points_per_woe = feature_info['points'].values[0]
        
        # Get WOE bins
        woe_info = woe_bins[feature]
        
        # Calculate points for each bin
        for _, row in woe_info.iterrows():
            score_mappings.append({
                'feature': feature,
                'bin': f"{row.get('min', 'N/A')} - {row.get('max', 'N/A')}",
                'woe': row['woe'],
                'points': points_per_woe * row['woe'],
                'pct_population': row.get('%accts', 'N/A')
            })
    
    score_mapping_df = pd.DataFrame(score_mappings)
    
    # Add base points
    base_points_row = pd.DataFrame([{
        'feature': 'BASE_SCORE',
        'bin': 'Intercept',
        'woe': 0,
        'points': scorecard['intercept_points'] + scorecard['offset'],
        'pct_population': '100%'
    }])
    
    score_mapping_df = pd.concat([base_points_row, score_mapping_df], ignore_index=True)
    
    return score_mapping_df


# Example 3: Segmented Scorecard Development
def build_segmented_scorecard(X_woe, y, segment_column, segment_df):
    """
    Build separate scorecards for different segments
    
    Args:
        X_woe: WOE-transformed features
        y: Target variable
        segment_column: Column defining segments
        segment_df: DataFrame with segment information
        
    Returns:
        segmented_scorecards: Dictionary of scorecards by segment
    """
    segments = segment_df[segment_column].unique()
    segmented_scorecards = {}
    
    print(f"Building scorecards for {len(segments)} segments...")
    
    for segment in segments:
        print(f"\nSegment: {segment}")
        
        # Get segment data
        segment_mask = segment_df[segment_column] == segment
        X_segment = X_woe[segment_mask]
        y_segment = y[segment_mask]
        
        print(f"- Sample size: {len(X_segment)}")
        print(f"- Bad rate: {y_segment.mean():.2%}")
        
        # Build scorecard for segment
        scorecard, model, performance = build_logistic_scorecard(
            X_segment, y_segment, base_score=600, pdo=20
        )
        
        segmented_scorecards[segment] = {
            'scorecard': scorecard,
            'model': model,
            'performance': performance,
            'sample_size': len(X_segment),
            'bad_rate': y_segment.mean()
        }
    
    # Compare segment performance
    comparison_df = pd.DataFrame([
        {
            'segment': seg,
            'sample_size': info['sample_size'],
            'bad_rate': info['bad_rate'],
            'auc': info['performance']['auc_test']
        }
        for seg, info in segmented_scorecards.items()
    ])
    
    print("\n=== Segment Comparison ===")
    print(comparison_df)
    
    return segmented_scorecards


# Example 4: Advanced ML Scorecard (LightGBM)
def build_ml_scorecard(X_woe, y, sample_weights=None):
    """
    Build machine learning based scorecard using LightGBM
    
    Args:
        X_woe: WOE-transformed features
        y: Target variable
        sample_weights: Optional sample weights
        
    Returns:
        model: Trained LightGBM model
        performance: Model performance metrics
        feature_importance: Feature importance DataFrame
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_woe, y, test_size=0.3, random_state=42, stratify=y
    )
    
    if sample_weights is not None:
        weights_train = sample_weights.iloc[X_train.index]
        weights_test = sample_weights.iloc[X_test.index]
    else:
        weights_train = None
        weights_test = None
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, label=y_train, weight=weights_train,
        categorical_feature='auto'
    )
    valid_data = lgb.Dataset(
        X_test, label=y_test, weight=weights_test,
        reference=train_data
    )
    
    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'min_data_in_leaf': 100,
        'max_depth': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Get predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate performance
    auc_score = roc_auc_score(y_test, y_pred_proba, sample_weight=weights_test)
    
    print(f"\nLightGBM Model Performance:")
    print(f"- Test AUC: {auc_score:.4f}")
    print(f"- Best iteration: {model.best_iteration}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_woe.columns,
        'importance': model.feature_importance(importance_type='gain'),
        'split': model.feature_importance(importance_type='split')
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Features - LightGBM')
    plt.tight_layout()
    plt.show()
    
    performance = {
        'auc_test': auc_score,
        'best_iteration': model.best_iteration
    }
    
    return model, performance, importance_df


# Example 5: Model Validation and Stability
def validate_scorecard(model, X_woe, y, n_folds=5):
    """
    Comprehensive validation of scorecard model
    
    Args:
        model: Trained model
        X_woe: WOE-transformed features
        y: Target variable
        n_folds: Number of folds for cross-validation
        
    Returns:
        validation_results: Dictionary with validation metrics
    """
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    validation_results = {
        'fold_auc': [],
        'fold_ks': [],
        'score_distributions': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_woe, y)):
        X_train_fold = X_woe.iloc[train_idx]
        X_val_fold = X_woe.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Refit model on fold
        if hasattr(model, 'fit'):
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train_fold, y_train_fold)
        else:
            # For LightGBM
            continue
        
        # Get predictions
        y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        validation_results['fold_auc'].append(auc)
        
        # Calculate KS statistic
        fpr, tpr, _ = roc_curve(y_val_fold, y_pred_proba)
        ks = max(tpr - fpr)
        validation_results['fold_ks'].append(ks)
        
        # Store score distribution
        validation_results['score_distributions'].append({
            'fold': fold,
            'scores': y_pred_proba,
            'labels': y_val_fold
        })
    
    # Summary statistics
    print("\n=== Cross-Validation Results ===")
    print(f"AUC: {np.mean(validation_results['fold_auc']):.4f} "
          f"(+/- {np.std(validation_results['fold_auc']) * 2:.4f})")
    print(f"KS: {np.mean(validation_results['fold_ks']):.4f} "
          f"(+/- {np.std(validation_results['fold_ks']) * 2:.4f})")
    
    # Plot score distributions by fold
    plt.figure(figsize=(12, 8))
    
    for i, dist_info in enumerate(validation_results['score_distributions'][:3]):  # First 3 folds
        plt.subplot(2, 3, i+1)
        
        scores_good = dist_info['scores'][dist_info['labels'] == 0]
        scores_bad = dist_info['scores'][dist_info['labels'] == 1]
        
        plt.hist(scores_good, bins=30, alpha=0.5, label='Good', density=True)
        plt.hist(scores_bad, bins=30, alpha=0.5, label='Bad', density=True)
        plt.xlabel('Probability Score')
        plt.ylabel('Density')
        plt.title(f'Fold {dist_info["fold"] + 1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return validation_results


# Example 6: Create final scorecard report
def create_scorecard_report(scorecard, validation_results, output_path='scorecard_report.txt'):
    """
    Create comprehensive scorecard documentation
    
    Args:
        scorecard: Scorecard dictionary
        validation_results: Validation results
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("CREDIT SCORECARD MODEL REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Model parameters
        f.write("MODEL PARAMETERS:\n")
        f.write(f"- Base Score: {scorecard['base_score']}\n")
        f.write(f"- PDO (Points to Double Odds): {scorecard['pdo']}\n")
        f.write(f"- Offset: {scorecard['offset']:.2f}\n")
        f.write(f"- Factor: {scorecard['factor']:.2f}\n")
        f.write(f"- Intercept Points: {scorecard['intercept_points']:.2f}\n\n")
        
        # Performance metrics
        f.write("MODEL PERFORMANCE:\n")
        f.write(f"- Mean CV AUC: {np.mean(validation_results['fold_auc']):.4f}\n")
        f.write(f"- Std CV AUC: {np.std(validation_results['fold_auc']):.4f}\n")
        f.write(f"- Mean CV KS: {np.mean(validation_results['fold_ks']):.4f}\n\n")
        
        # Feature coefficients
        f.write("FEATURE COEFFICIENTS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Feature':<30} {'Coefficient':>12} {'Points':>12}\n")
        f.write("-" * 50 + "\n")
        
        for _, row in scorecard['features'].iterrows():
            f.write(f"{row['feature']:<30} {row['coefficient']:>12.4f} {row['points']:>12.2f}\n")
        
        f.write("\n")
        f.write("=" * 50 + "\n")
    
    print(f"Scorecard report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample WOE-transformed data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    # Create correlated features
    X_woe = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}_woe' for i in range(n_features)]
    )
    
    # Add some correlation structure
    X_woe['feature_1_woe'] = X_woe['feature_0_woe'] * 0.7 + np.random.randn(n_samples) * 0.3
    X_woe['feature_2_woe'] = -X_woe['feature_0_woe'] * 0.5 + np.random.randn(n_samples) * 0.5
    
    # Create target with logistic relationship
    logit = (
        0.5 * X_woe['feature_0_woe'] +
        0.3 * X_woe['feature_1_woe'] +
        -0.4 * X_woe['feature_2_woe'] +
        0.2 * X_woe['feature_5_woe'] +
        -0.3 * X_woe['feature_10_woe'] +
        np.random.randn(n_samples) * 0.5
    )
    
    y = (1 / (1 + np.exp(-logit)) > 0.8).astype(int)
    
    print("=== Credit Scorecard Development Example ===\n")
    print(f"Dataset shape: {X_woe.shape}")
    print(f"Bad rate: {y.mean():.2%}")
    
    # Example 1: Build traditional scorecard
    print("\n1. Building Traditional Logistic Regression Scorecard")
    scorecard, lr_model, lr_performance = build_logistic_scorecard(X_woe, y)
    
    # Example 2: Build ML scorecard
    print("\n2. Building Machine Learning Scorecard (LightGBM)")
    lgb_model, lgb_performance, feature_importance = build_ml_scorecard(X_woe, y)
    
    # Example 3: Validate scorecard
    print("\n3. Validating Scorecard Model")
    validation_results = validate_scorecard(lr_model, X_woe, y)
    
    # Example 4: Create report
    print("\n4. Creating Scorecard Report")
    create_scorecard_report(scorecard, validation_results)
    
    # Example 5: Save models
    print("\n5. Saving Models")
    joblib.dump(lr_model, 'logistic_scorecard.pkl')
    lgb_model.save_model('lightgbm_scorecard.txt')
    print("Models saved successfully!")
    
    print("\nScorecard development complete!")