"""
Example: Score Alignment for Model Migration

This script demonstrates how to align scores between different models
to ensure business continuity during model updates.

Inputs:
- Old model scores and performance
- New model scores and performance
- Target variable

Outputs:
- Score mapping table
- Aligned scores maintaining risk ordering
- Comparison metrics
"""

import pandas as pd
import numpy as np
from score_alignment import get_cum_bad_rate, get_score_alignment_table
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Example 1: Basic Score Alignment
def basic_score_alignment(old_scores, new_scores, target, n_bins=20):
    """
    Align new model scores to match old model score distribution
    
    Args:
        old_scores: Scores from current production model
        new_scores: Scores from new model
        target: Target variable (0=Good, 1=Bad)
        n_bins: Number of bins for alignment
        
    Returns:
        aligned_scores: New scores aligned to old scale
        alignment_table: Mapping table
    """
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'old_score': old_scores,
        'new_score': new_scores,
        'target': target
    })
    
    # Calculate cumulative bad rates
    old_cum_bad = get_cum_bad_rate(df, 'old_score', 'target')
    new_cum_bad = get_cum_bad_rate(df, 'new_score', 'target')
    
    # Get alignment table
    alignment_table = get_score_alignment_table(
        old_cum_bad, 
        new_cum_bad,
        old_score_col='old_score',
        new_score_col='new_score'
    )
    
    # Apply alignment
    aligned_scores = np.interp(new_scores, alignment_table['new_score'], alignment_table['old_score'])
    
    # Visualize alignment
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Score mapping
    axes[0, 0].scatter(alignment_table['new_score'], alignment_table['old_score'], alpha=0.6)
    axes[0, 0].plot(alignment_table['new_score'], alignment_table['old_score'], 'r-', linewidth=2)
    axes[0, 0].set_xlabel('New Score')
    axes[0, 0].set_ylabel('Aligned Score')
    axes[0, 0].set_title('Score Alignment Mapping')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution comparison
    axes[0, 1].hist(old_scores, bins=30, alpha=0.5, label='Old Model', density=True)
    axes[0, 1].hist(new_scores, bins=30, alpha=0.5, label='New Model', density=True)
    axes[0, 1].hist(aligned_scores, bins=30, alpha=0.5, label='Aligned New', density=True)
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Score Distributions')
    axes[0, 1].legend()
    
    # 3. Bad rate by score band
    df['aligned_score'] = aligned_scores
    
    for score_col, label in [('old_score', 'Old'), ('new_score', 'New'), ('aligned_score', 'Aligned')]:
        df[f'{score_col}_band'] = pd.qcut(df[score_col], q=10, duplicates='drop')
        bad_rates = df.groupby(f'{score_col}_band')['target'].mean()
        axes[1, 0].plot(range(len(bad_rates)), bad_rates, marker='o', label=label)
    
    axes[1, 0].set_xlabel('Score Band (Decile)')
    axes[1, 0].set_ylabel('Bad Rate')
    axes[1, 0].set_title('Bad Rate by Score Band')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rank correlation
    rank_old_new = stats.spearmanr(old_scores, new_scores)[0]
    rank_old_aligned = stats.spearmanr(old_scores, aligned_scores)[0]
    
    axes[1, 1].bar(['Old vs New', 'Old vs Aligned'], [rank_old_new, rank_old_aligned])
    axes[1, 1].set_ylabel('Spearman Correlation')
    axes[1, 1].set_title('Rank Order Preservation')
    axes[1, 1].set_ylim(0, 1)
    
    for i, v in enumerate([rank_old_new, rank_old_aligned]):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Alignment Summary:")
    print(f"- Rank correlation (old vs new): {rank_old_new:.3f}")
    print(f"- Rank correlation (old vs aligned): {rank_old_aligned:.3f}")
    print(f"- Score range preserved: {aligned_scores.min():.0f} - {aligned_scores.max():.0f}")
    
    return aligned_scores, alignment_table


# Example 2: Isotonic Regression Alignment
def isotonic_score_alignment(old_scores, new_scores, target):
    """
    Use isotonic regression for monotonic score alignment
    
    Args:
        old_scores: Current production scores
        new_scores: New model scores
        target: Target variable
        
    Returns:
        aligned_scores: Monotonically aligned scores
        iso_model: Fitted isotonic regression model
    """
    # Sort by new scores
    sort_idx = np.argsort(new_scores)
    new_sorted = new_scores[sort_idx]
    old_sorted = old_scores[sort_idx]
    
    # Fit isotonic regression
    iso_model = IsotonicRegression(increasing=True)
    iso_model.fit(new_sorted, old_sorted)
    
    # Apply alignment
    aligned_scores = iso_model.predict(new_scores)
    
    # Ensure monotonicity is preserved
    print("Monotonicity Check:")
    print(f"- New scores monotonic: {np.all(np.diff(np.sort(new_scores)) >= 0)}")
    print(f"- Aligned scores monotonic: {np.all(np.diff(np.sort(aligned_scores)) >= 0)}")
    
    # Plot isotonic regression
    plt.figure(figsize=(10, 6))
    plt.scatter(new_scores, old_scores, alpha=0.3, label='Original pairs')
    
    # Plot isotonic fit
    x_plot = np.linspace(new_scores.min(), new_scores.max(), 100)
    y_plot = iso_model.predict(x_plot)
    plt.plot(x_plot, y_plot, 'r-', linewidth=3, label='Isotonic fit')
    
    plt.xlabel('New Score')
    plt.ylabel('Old Score')
    plt.title('Isotonic Regression Score Alignment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return aligned_scores, iso_model


# Example 3: Business Rule Preservation
def align_with_business_rules(old_scores, new_scores, target, 
                            approval_threshold=650, tier_boundaries=[600, 650, 700]):
    """
    Align scores while preserving specific business rules
    
    Args:
        old_scores: Current production scores
        new_scores: New model scores
        target: Target variable
        approval_threshold: Score threshold for approval
        tier_boundaries: Score boundaries for risk tiers
        
    Returns:
        aligned_scores: Scores aligned to preserve business rules
        rule_preservation: Dictionary of rule preservation metrics
    """
    # Calculate current approval rate
    old_approval_rate = (old_scores >= approval_threshold).mean()
    
    # Find new threshold to match approval rate
    new_threshold = np.percentile(new_scores, (1 - old_approval_rate) * 100)
    
    # Basic linear alignment with constraints
    # Map [new_min, new_threshold, new_max] to [old_min, approval_threshold, old_max]
    
    def piecewise_linear_map(scores, new_points, old_points):
        """Piecewise linear mapping"""
        aligned = np.zeros_like(scores)
        
        for i in range(len(new_points) - 1):
            mask = (scores >= new_points[i]) & (scores < new_points[i + 1])
            if i == len(new_points) - 2:  # Last segment
                mask = mask | (scores == new_points[i + 1])
            
            # Linear interpolation in this segment
            slope = (old_points[i + 1] - old_points[i]) / (new_points[i + 1] - new_points[i])
            aligned[mask] = old_points[i] + slope * (scores[mask] - new_points[i])
        
        return aligned
    
    # Define mapping points
    new_percentiles = [0, 25, 50, 75, 100]
    new_points = [np.percentile(new_scores, p) for p in new_percentiles]
    old_points = [np.percentile(old_scores, p) for p in new_percentiles]
    
    # Ensure approval threshold is preserved
    # Find where new_threshold falls in new_points
    for i in range(len(new_points) - 1):
        if new_threshold >= new_points[i] and new_threshold <= new_points[i + 1]:
            # Insert the threshold
            ratio = (new_threshold - new_points[i]) / (new_points[i + 1] - new_points[i])
            old_threshold_mapped = old_points[i] + ratio * (old_points[i + 1] - old_points[i])
            
            # Adjust to exact threshold
            adjustment = approval_threshold - old_threshold_mapped
            new_points.insert(i + 1, new_threshold)
            old_points.insert(i + 1, approval_threshold)
            break
    
    # Apply alignment
    aligned_scores = piecewise_linear_map(new_scores, new_points, old_points)
    
    # Check rule preservation
    rule_preservation = {
        'old_approval_rate': old_approval_rate,
        'new_approval_rate': (new_scores >= new_threshold).mean(),
        'aligned_approval_rate': (aligned_scores >= approval_threshold).mean(),
        'approval_rate_diff': abs((aligned_scores >= approval_threshold).mean() - old_approval_rate)
    }
    
    # Check tier preservation
    for i, boundary in enumerate(tier_boundaries):
        old_pct = (old_scores >= boundary).mean()
        aligned_pct = (aligned_scores >= boundary).mean()
        rule_preservation[f'tier_{boundary}_preservation'] = abs(aligned_pct - old_pct)
    
    # Visualize rule preservation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Approval rates
    categories = ['Old Model', 'New Model\n(Original)', 'New Model\n(Aligned)']
    approval_rates = [
        old_approval_rate,
        (new_scores >= new_threshold).mean(),
        (aligned_scores >= approval_threshold).mean()
    ]
    
    axes[0].bar(categories, approval_rates)
    axes[0].axhline(y=old_approval_rate, color='r', linestyle='--', label='Target Rate')
    axes[0].set_ylabel('Approval Rate')
    axes[0].set_title('Approval Rate Preservation')
    axes[0].legend()
    
    for i, v in enumerate(approval_rates):
        axes[0].text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    # Score distribution with tiers
    axes[1].hist(old_scores, bins=30, alpha=0.5, label='Old Model', density=True)
    axes[1].hist(aligned_scores, bins=30, alpha=0.5, label='Aligned New Model', density=True)
    
    for boundary in tier_boundaries:
        axes[1].axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Score Distribution with Risk Tiers')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nBusiness Rule Preservation:")
    for key, value in rule_preservation.items():
        if 'rate' in key:
            print(f"- {key}: {value:.1%}")
        else:
            print(f"- {key}: {value:.3f}")
    
    return aligned_scores, rule_preservation


# Example 4: Population Stability Index (PSI) After Alignment
def calculate_psi_impact(old_scores, new_scores, aligned_scores, n_bins=10):
    """
    Calculate PSI to measure population stability after alignment
    
    Args:
        old_scores: Original production scores
        new_scores: New model scores (unaligned)
        aligned_scores: Aligned new model scores
        n_bins: Number of bins for PSI calculation
        
    Returns:
        psi_results: Dictionary with PSI metrics
    """
    def calculate_psi(expected, actual, bins):
        """Calculate PSI between two distributions"""
        expected_percents = pd.cut(expected, bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
        actual_percents = pd.cut(actual, bins=bins, include_lowest=True).value_counts(normalize=True).sort_index()
        
        # Avoid log(0)
        expected_percents = expected_percents.replace(0, 0.0001)
        actual_percents = actual_percents.replace(0, 0.0001)
        
        psi = sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi, expected_percents, actual_percents
    
    # Define bins based on old scores
    bins = np.percentile(old_scores, np.linspace(0, 100, n_bins + 1))
    bins[0] = bins[0] - 0.001  # Ensure lowest value is included
    bins[-1] = bins[-1] + 0.001  # Ensure highest value is included
    
    # Calculate PSI
    psi_new, _, _ = calculate_psi(old_scores, new_scores, bins)
    psi_aligned, expected_dist, aligned_dist = calculate_psi(old_scores, aligned_scores, bins)
    
    print(f"Population Stability Index (PSI):")
    print(f"- PSI (Old vs New): {psi_new:.4f}")
    print(f"- PSI (Old vs Aligned): {psi_aligned:.4f}")
    print(f"- PSI Improvement: {psi_new - psi_aligned:.4f}")
    
    # PSI interpretation
    def interpret_psi(psi):
        if psi < 0.1:
            return "Insignificant change"
        elif psi < 0.25:
            return "Some change"
        else:
            return "Significant change"
    
    print(f"\nInterpretation:")
    print(f"- New model: {interpret_psi(psi_new)}")
    print(f"- Aligned model: {interpret_psi(psi_aligned)}")
    
    # Visualize distribution changes
    plt.figure(figsize=(10, 6))
    
    x = range(len(expected_dist))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], expected_dist, width, label='Old Model', alpha=0.7)
    plt.bar([i + width/2 for i in x], aligned_dist, width, label='Aligned New Model', alpha=0.7)
    
    plt.xlabel('Score Bin')
    plt.ylabel('Proportion')
    plt.title('Score Distribution Comparison')
    plt.xticks(x, [f'Bin {i+1}' for i in x], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add PSI contribution for each bin
    psi_contributions = (aligned_dist - expected_dist) * np.log(aligned_dist / expected_dist)
    for i, contrib in enumerate(psi_contributions):
        plt.text(i, max(expected_dist[i], aligned_dist[i]) + 0.01, 
                f'{contrib:.3f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'psi_new': psi_new,
        'psi_aligned': psi_aligned,
        'psi_improvement': psi_new - psi_aligned,
        'expected_distribution': expected_dist,
        'aligned_distribution': aligned_dist
    }


# Example 5: Create comprehensive alignment report
def create_alignment_report(old_scores, new_scores, aligned_scores, target, 
                          output_path='alignment_report.txt'):
    """
    Create detailed score alignment report
    
    Args:
        old_scores: Original production scores
        new_scores: New model scores
        aligned_scores: Aligned scores
        target: Target variable
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SCORE ALIGNMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Score statistics
        f.write("SCORE STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<20} {'Old Model':>12} {'New Model':>12} {'Aligned':>12}\n")
        f.write("-" * 40 + "\n")
        
        for metric_name, metric_func in [
            ('Mean', np.mean),
            ('Std Dev', np.std),
            ('Min', np.min),
            ('25th Percentile', lambda x: np.percentile(x, 25)),
            ('Median', np.median),
            ('75th Percentile', lambda x: np.percentile(x, 75)),
            ('Max', np.max)
        ]:
            f.write(f"{metric_name:<20} {metric_func(old_scores):>12.1f} "
                   f"{metric_func(new_scores):>12.1f} {metric_func(aligned_scores):>12.1f}\n")
        
        # Correlation analysis
        f.write("\n\nCORRELATION ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        pearson_old_new = stats.pearsonr(old_scores, new_scores)[0]
        pearson_old_aligned = stats.pearsonr(old_scores, aligned_scores)[0]
        spearman_old_new = stats.spearmanr(old_scores, new_scores)[0]
        spearman_old_aligned = stats.spearmanr(old_scores, aligned_scores)[0]
        
        f.write(f"Pearson Correlation:\n")
        f.write(f"  Old vs New: {pearson_old_new:.4f}\n")
        f.write(f"  Old vs Aligned: {pearson_old_aligned:.4f}\n")
        f.write(f"\nSpearman Correlation:\n")
        f.write(f"  Old vs New: {spearman_old_new:.4f}\n")
        f.write(f"  Old vs Aligned: {spearman_old_aligned:.4f}\n")
        
        # Bad rate preservation
        f.write("\n\nBAD RATE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        # Create score bands
        n_bands = 10
        old_bands = pd.qcut(old_scores, q=n_bands, duplicates='drop')
        new_bands = pd.qcut(new_scores, q=n_bands, duplicates='drop')
        aligned_bands = pd.qcut(aligned_scores, q=n_bands, duplicates='drop')
        
        f.write(f"{'Band':<10} {'Old BR':>10} {'New BR':>10} {'Aligned BR':>10}\n")
        f.write("-" * 40 + "\n")
        
        for i in range(n_bands):
            old_br = target[old_bands == old_bands.cat.categories[i]].mean()
            new_br = target[new_bands == new_bands.cat.categories[i]].mean()
            aligned_br = target[aligned_bands == aligned_bands.cat.categories[i]].mean()
            
            f.write(f"{i+1:<10} {old_br:>10.1%} {new_br:>10.1%} {aligned_br:>10.1%}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Alignment report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Simulate old model scores (300-850 range)
    old_scores = np.random.beta(5, 2, n_samples) * 550 + 300
    
    # Simulate new model scores (different distribution, 0-1000 range)
    new_scores = np.random.gamma(5, 100, n_samples)
    new_scores = np.clip(new_scores, 0, 1000)
    
    # Create correlated target
    old_probs = 1 / (1 + np.exp((old_scores - 600) / 50))
    target = (np.random.random(n_samples) < old_probs).astype(int)
    
    print("=== Score Alignment Example ===\n")
    print(f"Sample size: {n_samples}")
    print(f"Old score range: {old_scores.min():.0f} - {old_scores.max():.0f}")
    print(f"New score range: {new_scores.min():.0f} - {new_scores.max():.0f}")
    print(f"Bad rate: {target.mean():.2%}")
    
    # Example 1: Basic alignment
    print("\n1. Basic Score Alignment")
    aligned_scores, alignment_table = basic_score_alignment(old_scores, new_scores, target)
    
    # Example 2: Isotonic alignment
    print("\n2. Isotonic Regression Alignment")
    iso_aligned_scores, iso_model = isotonic_score_alignment(old_scores, new_scores, target)
    
    # Example 3: Business rule preservation
    print("\n3. Business Rule Preserving Alignment")
    rule_aligned_scores, rule_metrics = align_with_business_rules(
        old_scores, new_scores, target,
        approval_threshold=650,
        tier_boundaries=[600, 650, 700]
    )
    
    # Example 4: PSI analysis
    print("\n4. Population Stability Analysis")
    psi_results = calculate_psi_impact(old_scores, new_scores, aligned_scores)
    
    # Example 5: Create report
    print("\n5. Creating Alignment Report")
    create_alignment_report(old_scores, new_scores, aligned_scores, target)
    
    print("\nScore alignment complete!")