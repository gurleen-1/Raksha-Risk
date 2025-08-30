"""Unified entry point for the RakshaRisk project.

This script orchestrates both the fraud detection and credit default
prediction pipelines.  It parses command‑line arguments, loads data (or
generates synthetic samples), trains models, evaluates them and writes
results to disk.  See the README for dataset descriptions and usage
examples.
"""

from __future__ import annotations

import argparse
import json
import os

from src.fraud_detection import run_fraud_pipeline
from src.default_prediction import run_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud detection and credit default prediction pipelines")
    # Fraud detection arguments
    parser.add_argument('--fraud-data', type=str, default='data/creditcard.csv', help='Path to credit card fraud dataset')
    parser.add_argument('--use-synthetic-fraud', action='store_true', help='Use synthetic fraud dataset instead of real data')
    parser.add_argument('--contamination', type=float, default=0.001, help='Expected fraud rate for anomaly detectors')
    # Default prediction arguments
    parser.add_argument('--default-data', type=str, default='data/default_of_credit_card_clients.xls', help='Path to credit default dataset')
    parser.add_argument('--use-synthetic-default', action='store_true', help='Use synthetic default dataset instead of real data')
    # General
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save all outputs')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Fraud detection
    iso_metrics, lof_metrics = run_fraud_pipeline(
        data_path=args.fraud_data,
        use_synthetic=args.use_synthetic_fraud,
        contamination=args.contamination,
        output_dir=args.output_dir
    )
    # Default prediction
    rf_metrics, lr_metrics = run_default_pipeline(
        data_path=args.default_data,
        use_synthetic=args.use_synthetic_default,
        output_dir=args.output_dir
    )
    # Combine metrics and save as JSON
    results = {
        'fraud_detection': {
            'isolation_forest': iso_metrics,
            'local_outlier_factor': lof_metrics,
        },
        'default_prediction': {
            'random_forest': rf_metrics,
            'logistic_regression': lr_metrics,
        }
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()