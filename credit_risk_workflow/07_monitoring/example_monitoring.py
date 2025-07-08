"""
Example: Production Model Monitoring

This script demonstrates how to monitor credit risk models in production,
including performance tracking, drift detection, and automated alerts.

Inputs:
- Production scoring data
- Actual performance outcomes
- Original development data

Outputs:
- Performance monitoring reports
- Drift detection alerts
- Retraining recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from monitor import MonitorBase
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Example 1: Basic Performance Monitoring
class CreditModelMonitor(MonitorBase):
    """
    Credit risk model monitoring implementation
    """
    
    def __init__(self, model, woe_transformer, feature_list, thresholds):
        super().__init__()
        self.model = model
        self.woe_transformer = woe_transformer
        self.feature_list = feature_list
        self.thresholds = thresholds
        
    def get_dev_data(self):
        """Get development/training data for comparison"""
        # In production, this would load from a database or file
        # For example, loading the data used to train the model
        return self.context.get('dev_data')
    
    def get_prod_data(self):
        """Get recent production data"""
        # In production, this would query recent scoring data
        return self.context.get('prod_data')
    
    def get_pred(self):
        """Generate predictions on production data"""
        prod_data = self.context.get('prod_data')
        
        # Apply WOE transformation
        X_woe = self.woe_transformer.transform(prod_data[self.feature_list])
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X_woe)[:, 1]
        else:
            predictions = self.model.predict(X_woe)
        
        self.context['predictions'] = predictions
        return predictions
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now(),
            'psi': self.calculate_psi(),
            'score_distribution': self.analyze_score_distribution(),
            'feature_drift': self.check_feature_drift(),
            'performance_metrics': self.calculate_performance_metrics()
        }
        
        self.context['report'] = report
        return report
    
    def refit(self):
        """Retrain model if needed"""
        report = self.context.get('report', {})
        
        # Check if retraining is needed
        if report.get('psi', 0) > self.thresholds['psi_threshold']:
            print("High PSI detected. Retraining recommended.")
            # Implement retraining logic here
            return True
        
        return False
    
    def calculate_psi(self):
        """Calculate Population Stability Index"""
        dev_data = self.context.get('dev_data')
        prod_data = self.context.get('prod_data')
        
        psi_values = {}
        
        for feature in self.feature_list:
            # Create bins based on development data
            _, bins = pd.qcut(dev_data[feature].dropna(), q=10, retbins=True, duplicates='drop')
            
            # Calculate distributions
            dev_dist = pd.cut(dev_data[feature], bins=bins, include_lowest=True).value_counts(normalize=True)
            prod_dist = pd.cut(prod_data[feature], bins=bins, include_lowest=True).value_counts(normalize=True)
            
            # Ensure same index
            all_bins = dev_dist.index.union(prod_dist.index)
            dev_dist = dev_dist.reindex(all_bins, fill_value=0.0001)
            prod_dist = prod_dist.reindex(all_bins, fill_value=0.0001)
            
            # Calculate PSI
            psi = np.sum((prod_dist - dev_dist) * np.log(prod_dist / dev_dist))
            psi_values[feature] = psi
        
        return psi_values
    
    def analyze_score_distribution(self):
        """Analyze score distribution changes"""
        predictions = self.context.get('predictions', [])
        
        return {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'percentiles': {
                '25': np.percentile(predictions, 25),
                '50': np.percentile(predictions, 50),
                '75': np.percentile(predictions, 75)
            }
        }
    
    def check_feature_drift(self):
        """Check for feature drift using statistical tests"""
        dev_data = self.context.get('dev_data')
        prod_data = self.context.get('prod_data')
        
        drift_results = {}
        
        for feature in self.feature_list:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(
                dev_data[feature].dropna(),
                prod_data[feature].dropna()
            )
            
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'drift_detected': ks_pvalue < 0.05
            }
        
        return drift_results
    
    def calculate_performance_metrics(self):
        """Calculate model performance metrics if outcomes are available"""
        prod_data = self.context.get('prod_data')
        predictions = self.context.get('predictions')
        
        # Check if we have actual outcomes
        if 'actual_target' not in prod_data.columns:
            return {'status': 'Outcomes not yet available'}
        
        from sklearn.metrics import roc_auc_score, confusion_matrix
        
        actual = prod_data['actual_target']
        
        return {
            'auc': roc_auc_score(actual, predictions),
            'bad_rate': actual.mean(),
            'predicted_bad_rate': predictions.mean()
        }


# Example 2: Monitoring Dashboard
def create_monitoring_dashboard(monitor_results, output_path='monitoring_dashboard.png'):
    """
    Create visual monitoring dashboard
    
    Args:
        monitor_results: Results from monitoring
        output_path: Path to save dashboard image
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. PSI by Feature
    ax1 = plt.subplot(2, 3, 1)
    psi_data = monitor_results['psi']
    features = list(psi_data.keys())[:10]  # Top 10 features
    psi_values = [psi_data[f] for f in features]
    
    bars = ax1.barh(features, psi_values)
    
    # Color code by PSI level
    for bar, psi in zip(bars, psi_values):
        if psi < 0.1:
            bar.set_color('green')
        elif psi < 0.25:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    ax1.axvline(x=0.1, color='yellow', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('PSI')
    ax1.set_title('Population Stability Index by Feature')
    
    # 2. Score Distribution Over Time
    ax2 = plt.subplot(2, 3, 2)
    # Simulate historical data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    scores = [monitor_results['score_distribution']['mean'] + np.random.normal(0, 5) for _ in dates]
    
    ax2.plot(dates, scores, marker='o')
    ax2.axhline(y=monitor_results['score_distribution']['mean'], color='r', linestyle='--', 
                label='Current Mean')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Mean Score Trend')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Feature Drift Detection
    ax3 = plt.subplot(2, 3, 3)
    drift_data = monitor_results['feature_drift']
    drift_features = list(drift_data.keys())[:10]
    drift_detected = [drift_data[f]['drift_detected'] for f in drift_features]
    
    colors = ['red' if d else 'green' for d in drift_detected]
    ax3.barh(drift_features, [1] * len(drift_features), color=colors)
    ax3.set_xlim(0, 1.5)
    ax3.set_xlabel('Drift Status')
    ax3.set_title('Feature Drift Detection')
    ax3.set_xticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='No Drift'),
                      Patch(facecolor='red', label='Drift Detected')]
    ax3.legend(handles=legend_elements, loc='center right')
    
    # 4. Performance Metrics (if available)
    ax4 = plt.subplot(2, 3, 4)
    perf_metrics = monitor_results.get('performance_metrics', {})
    
    if 'auc' in perf_metrics:
        metrics = ['AUC', 'Bad Rate', 'Predicted BR']
        values = [
            perf_metrics.get('auc', 0),
            perf_metrics.get('bad_rate', 0),
            perf_metrics.get('predicted_bad_rate', 0)
        ]
        
        ax4.bar(metrics, values)
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics')
        
        for i, v in enumerate(values):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')
    else:
        ax4.text(0.5, 0.5, 'Performance outcomes\nnot yet available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Performance Metrics')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # 5. Alert Summary
    ax5 = plt.subplot(2, 3, 5)
    alerts = []
    
    # Check PSI alerts
    high_psi_features = [f for f, v in psi_data.items() if v > 0.25]
    if high_psi_features:
        alerts.append(f"High PSI: {len(high_psi_features)} features")
    
    # Check drift alerts
    drift_features = [f for f, d in drift_data.items() if d['drift_detected']]
    if drift_features:
        alerts.append(f"Drift detected: {len(drift_features)} features")
    
    # Check score shift
    score_dist = monitor_results['score_distribution']
    if abs(score_dist['mean'] - 0.2) > 0.05:  # Assuming expected mean of 0.2
        alerts.append("Significant score shift detected")
    
    if not alerts:
        alerts = ["No alerts - Model performing as expected"]
    
    ax5.text(0.1, 0.9, "ALERTS", fontsize=14, fontweight='bold', transform=ax5.transAxes)
    for i, alert in enumerate(alerts):
        color = 'red' if i < len(alerts) - 1 else 'green'
        ax5.text(0.1, 0.7 - i*0.15, f"• {alert}", transform=ax5.transAxes, 
                color=color, fontsize=12)
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title('Alert Summary')
    
    # 6. Monitoring Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.1, 0.9, "MONITORING SUMMARY", fontsize=14, fontweight='bold', 
            transform=ax6.transAxes)
    
    summary_text = f"""
    Timestamp: {monitor_results['timestamp'].strftime('%Y-%m-%d %H:%M')}
    
    Overall Health: {'⚠️ Warning' if alerts[0] != "No alerts - Model performing as expected" else '✅ Good'}
    
    Key Metrics:
    • Max PSI: {max(psi_data.values()):.3f}
    • Features with drift: {len([d for d in drift_data.values() if d['drift_detected']])}
    • Score mean: {score_dist['mean']:.3f}
    • Score std: {score_dist['std']:.3f}
    
    Recommendation: {'Monitor closely' if alerts[0] != "No alerts - Model performing as expected" else 'Continue monitoring'}
    """
    
    ax6.text(0.1, 0.1, summary_text, transform=ax6.transAxes, fontsize=10, 
            verticalalignment='bottom')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_title('Summary')
    
    plt.suptitle('Credit Risk Model Monitoring Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dashboard saved to: {output_path}")


# Example 3: Automated Monitoring Pipeline
def run_automated_monitoring(model, woe_transformer, feature_list, 
                           dev_data, prod_data, config):
    """
    Run automated monitoring pipeline
    
    Args:
        model: Trained model
        woe_transformer: WOE transformer
        feature_list: List of features
        dev_data: Development data
        prod_data: Production data
        config: Monitoring configuration
        
    Returns:
        monitoring_results: Complete monitoring results
        actions: Recommended actions
    """
    # Initialize monitor
    monitor = CreditModelMonitor(
        model=model,
        woe_transformer=woe_transformer,
        feature_list=feature_list,
        thresholds=config['thresholds']
    )
    
    # Set context
    monitor.context = {
        'dev_data': dev_data,
        'prod_data': prod_data
    }
    
    # Run monitoring pipeline
    monitor.run()
    
    # Get results
    monitoring_results = monitor.context['report']
    
    # Determine actions
    actions = []
    
    # Check PSI
    max_psi = max(monitoring_results['psi'].values())
    if max_psi > config['thresholds']['psi_threshold']:
        actions.append({
            'type': 'alert',
            'severity': 'high',
            'message': f'High PSI detected: {max_psi:.3f}',
            'recommendation': 'Consider model retraining'
        })
    
    # Check drift
    drift_count = sum(1 for d in monitoring_results['feature_drift'].values() 
                     if d['drift_detected'])
    if drift_count > config['thresholds']['max_drift_features']:
        actions.append({
            'type': 'alert',
            'severity': 'medium',
            'message': f'{drift_count} features show significant drift',
            'recommendation': 'Investigate feature changes'
        })
    
    # Check score distribution
    score_dist = monitoring_results['score_distribution']
    if abs(score_dist['mean'] - config['expected_score_mean']) > config['thresholds']['score_shift_threshold']:
        actions.append({
            'type': 'alert',
            'severity': 'medium',
            'message': 'Significant score distribution shift',
            'recommendation': 'Review recent population changes'
        })
    
    if not actions:
        actions.append({
            'type': 'info',
            'severity': 'low',
            'message': 'Model performing within expected parameters',
            'recommendation': 'Continue regular monitoring'
        })
    
    return monitoring_results, actions


# Example 4: Historical Monitoring Trends
def analyze_historical_trends(historical_results):
    """
    Analyze trends from historical monitoring results
    
    Args:
        historical_results: List of monitoring results over time
        
    Returns:
        trend_analysis: Dictionary with trend insights
    """
    # Extract time series data
    dates = [r['timestamp'] for r in historical_results]
    psi_trends = {}
    score_means = []
    
    # Collect PSI trends for each feature
    all_features = set()
    for result in historical_results:
        all_features.update(result['psi'].keys())
    
    for feature in all_features:
        psi_trends[feature] = [r['psi'].get(feature, 0) for r in historical_results]
    
    # Collect score distribution trends
    for result in historical_results:
        score_means.append(result['score_distribution']['mean'])
    
    # Plot trends
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # PSI trends
    ax1 = axes[0]
    for feature, psi_values in list(psi_trends.items())[:5]:  # Top 5 features
        ax1.plot(dates, psi_values, marker='o', label=feature)
    
    ax1.axhline(y=0.1, color='yellow', linestyle='--', alpha=0.5, label='Warning threshold')
    ax1.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PSI')
    ax1.set_title('PSI Trends by Feature')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Score distribution trends
    ax2 = axes[1]
    ax2.plot(dates, score_means, marker='o', color='blue', linewidth=2)
    
    # Add trend line
    x_numeric = np.arange(len(dates))
    z = np.polyfit(x_numeric, score_means, 1)
    p = np.poly1d(z)
    ax2.plot(dates, p(x_numeric), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Mean Score')
    ax2.set_title('Score Distribution Trend')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate trend statistics
    trend_analysis = {
        'psi_increasing_features': [],
        'psi_stable_features': [],
        'score_trend': 'increasing' if z[0] > 0.001 else 'decreasing' if z[0] < -0.001 else 'stable',
        'score_trend_magnitude': abs(z[0])
    }
    
    # Analyze PSI trends
    for feature, psi_values in psi_trends.items():
        if len(psi_values) > 1:
            trend = np.polyfit(range(len(psi_values)), psi_values, 1)[0]
            if trend > 0.001:
                trend_analysis['psi_increasing_features'].append(feature)
            else:
                trend_analysis['psi_stable_features'].append(feature)
    
    return trend_analysis


# Example 5: Monitoring Configuration
def create_monitoring_config():
    """
    Create monitoring configuration with thresholds and parameters
    
    Returns:
        config: Dictionary with monitoring configuration
    """
    config = {
        'thresholds': {
            'psi_threshold': 0.25,  # Maximum acceptable PSI
            'max_drift_features': 3,  # Maximum features with drift
            'score_shift_threshold': 0.05,  # Maximum score mean shift
            'auc_degradation_threshold': 0.05,  # Maximum AUC drop
            'bad_rate_deviation_threshold': 0.2  # Maximum relative bad rate change
        },
        'monitoring_frequency': 'daily',
        'retraining_frequency': 'quarterly',
        'expected_score_mean': 0.2,
        'expected_bad_rate': 0.08,
        'alert_channels': ['email', 'slack'],
        'dashboard_update_frequency': 'hourly'
    }
    
    return config


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples_dev = 10000
    n_samples_prod = 2000
    n_features = 10
    
    # Development data (training data)
    dev_data = pd.DataFrame(
        np.random.randn(n_samples_dev, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Production data (with some drift)
    prod_data = pd.DataFrame(
        np.random.randn(n_samples_prod, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Introduce drift in some features
    prod_data['feature_0'] += 0.5  # Mean shift
    prod_data['feature_1'] *= 1.5  # Variance change
    prod_data['feature_2'] = np.random.exponential(1, n_samples_prod)  # Distribution change
    
    # Create dummy model and transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    # For demonstration, create a simple model
    y_dev = (dev_data['feature_0'] + dev_data['feature_1'] > 0).astype(int)
    model = LogisticRegression()
    model.fit(dev_data, y_dev)
    
    # Create dummy WOE transformer
    class DummyWOETransformer:
        def transform(self, X):
            return X
    
    woe_transformer = DummyWOETransformer()
    
    print("=== Credit Risk Model Monitoring Example ===\n")
    
    # Example 1: Create monitoring configuration
    print("1. Creating Monitoring Configuration")
    config = create_monitoring_config()
    print(f"Monitoring thresholds set:")
    for key, value in config['thresholds'].items():
        print(f"  - {key}: {value}")
    
    # Example 2: Run monitoring
    print("\n2. Running Automated Monitoring")
    feature_list = [f'feature_{i}' for i in range(n_features)]
    monitoring_results, actions = run_automated_monitoring(
        model, woe_transformer, feature_list,
        dev_data, prod_data, config
    )
    
    print(f"\nMonitoring completed at: {monitoring_results['timestamp']}")
    print(f"Actions required: {len(actions)}")
    for action in actions:
        print(f"  - [{action['severity'].upper()}] {action['message']}")
    
    # Example 3: Create dashboard
    print("\n3. Creating Monitoring Dashboard")
    create_monitoring_dashboard(monitoring_results)
    
    # Example 4: Analyze historical trends
    print("\n4. Analyzing Historical Trends")
    # Simulate historical results
    historical_results = []
    for i in range(30):
        # Create prod data with increasing drift
        hist_prod_data = pd.DataFrame(
            np.random.randn(n_samples_prod, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        hist_prod_data['feature_0'] += 0.5 + i * 0.01  # Increasing drift
        
        # Run monitoring
        monitor = CreditModelMonitor(model, woe_transformer, feature_list, config['thresholds'])
        monitor.context = {'dev_data': dev_data, 'prod_data': hist_prod_data}
        monitor.run()
        
        result = monitor.context['report']
        result['timestamp'] = datetime.now() - timedelta(days=30-i)
        historical_results.append(result)
    
    trend_analysis = analyze_historical_trends(historical_results)
    print(f"\nTrend Analysis:")
    print(f"- Score trend: {trend_analysis['score_trend']}")
    print(f"- Features with increasing PSI: {len(trend_analysis['psi_increasing_features'])}")
    
    print("\nMonitoring setup complete!")