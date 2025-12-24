#!/bin/bash
# Complete experiment loop for detection and evaluation
# This script runs all combinations of detection settings and evaluates them

set -e  # Exit on error

# Configuration
FEATURE_DIR="feature"
DATABASE_PATH="database.pkl"
RESULTS_DIR="experiment_results"
EVAL_DIR="evaluation_results"
LOG_DIR="logs"

# Create directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${EVAL_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Starting Complete Experiment Loop"
echo "=========================================="
echo ""

# Step 1: Build database
echo "Step 1: Building database..."
echo "----------------------------------------"
# python functions/database_construct.py \
#     --feature_dir "${FEATURE_DIR}" \
#     --output "${DATABASE_PATH}" \
#     2>&1 | tee "${LOG_DIR}/database_construction.log"

if [ ! -f "${DATABASE_PATH}" ]; then
    echo "ERROR: Database construction failed!"
    exit 1
fi

echo "Database built successfully: ${DATABASE_PATH}"
echo ""


echo '''
# Step 2: Run detection experiments with different configurations
echo "Step 2: Running detection experiments..."
echo "----------------------------------------"

# Define experiment configurations
declare -a DETECTION_MODES=("metric" "feature")
declare -a METRIC_MODES=("numerical_only" "with_classification" "numerical_only_normalized" "with_classification_normalized")
declare -a SIMILARITY_METRICS=("cosine" "euclidean")
declare -a OOD_THRESHOLDS=("0.3" "0.5" "0.7")

experiment_count=0

# Experiment 1: Metric-based detection with different metric modes and similarity metrics
for metric_mode in "${METRIC_MODES[@]}"; do
    for sim_metric in "${SIMILARITY_METRICS[@]}"; do
        for threshold in "${OOD_THRESHOLDS[@]}"; do
            experiment_count=$((experiment_count + 1))
            exp_name="metric_${metric_mode}_${sim_metric}_th${threshold}"
            results_file="${RESULTS_DIR}/detection_${exp_name}.json"
            log_file="${LOG_DIR}/detection_${exp_name}.log"
            
            echo "[${experiment_count}] Running: ${exp_name}"
            
            python functions/detecting.py \
                --database "${DATABASE_PATH}" \
                --feature_dir "${FEATURE_DIR}" \
                --output "${results_file}" \
                --detection_mode "metric" \
                --metric_mode "${metric_mode}" \
                --similarity_metric "${sim_metric}" \
                --ood_threshold "${threshold}" \
                --top_k_metrics 10 \
                2>&1 | tee "${log_file}"
            
            if [ ! -f "${results_file}" ]; then
                echo "WARNING: Detection failed for ${exp_name}"
                continue
            fi
            
            echo "  -> Results saved to ${results_file}"
        done
    done
done

# Experiment 2: Feature-based detection with different similarity metrics
for sim_metric in "${SIMILARITY_METRICS[@]}"; do
    for threshold in "${OOD_THRESHOLDS[@]}"; do
        experiment_count=$((experiment_count + 1))
        exp_name="feature_${sim_metric}_th${threshold}"
        results_file="${RESULTS_DIR}/detection_${exp_name}.json"
        log_file="${LOG_DIR}/detection_${exp_name}.log"
        
        echo "[${experiment_count}] Running: ${exp_name}"
        
        python functions/detecting.py \
            --database "${DATABASE_PATH}" \
            --feature_dir "${FEATURE_DIR}" \
            --output "${results_file}" \
            --detection_mode "feature" \
            --similarity_metric "${sim_metric}" \
            --ood_threshold "${threshold}" \
            --top_k_metrics 10 \
            2>&1 | tee "${log_file}"
        
        if [ ! -f "${results_file}" ]; then
            echo "WARNING: Detection failed for ${exp_name}"
            continue
        fi
        
        echo "  -> Results saved to ${results_file}"
    done
done

# Experiment 3: Raw features detection with different aggregation methods
# declare -a RAW_FEAT_AGGREGATIONS=("mean" "max" "top_k_mean")
declare -a RAW_FEAT_AGGREGATIONS=("mean")
for sim_metric in "${SIMILARITY_METRICS[@]}"; do
    for threshold in "${OOD_THRESHOLDS[@]}"; do
        for agg_method in "${RAW_FEAT_AGGREGATIONS[@]}"; do
            experiment_count=$((experiment_count + 1))
            exp_name="raw_features_${agg_method}_${sim_metric}_th${threshold}"
            results_file="${RESULTS_DIR}/detection_${exp_name}.json"
            log_file="${LOG_DIR}/detection_${exp_name}.log"
            
            echo "[${experiment_count}] Running: ${exp_name}"
            
            python functions/detecting.py \
                --database "${DATABASE_PATH}" \
                --feature_dir "${FEATURE_DIR}" \
                --output "${results_file}" \
                --detection_mode "raw_features" \
                --similarity_metric "${sim_metric}" \
                --ood_threshold "${threshold}" \
                --raw_features_aggregation "${agg_method}" \
                --top_k_metrics 10 \
                2>&1 | tee "${log_file}"
            
            if [ ! -f "${results_file}" ]; then
                echo "WARNING: Detection failed for ${exp_name}"
                continue
            fi
            
            echo "  -> Results saved to ${results_file}"
        done
    done
done

# Experiment 4: Few-shot learning with different k_shot values and aggregation methods
declare -a K_SHOT_VALUES=("3" "5" "10" "20")
# declare -a FEWSHOT_AGGREGATIONS=("mean" "max" "median")
declare -a FEWSHOT_AGGREGATIONS=("mean")
for k_shot in "${K_SHOT_VALUES[@]}"; do
    for agg_method in "${FEWSHOT_AGGREGATIONS[@]}"; do
        for sim_metric in "${SIMILARITY_METRICS[@]}"; do
            for threshold in "${OOD_THRESHOLDS[@]}"; do
                experiment_count=$((experiment_count + 1))
                exp_name="fewshot_k${k_shot}_${agg_method}_${sim_metric}_th${threshold}"
                results_file="${RESULTS_DIR}/detection_${exp_name}.json"
                log_file="${LOG_DIR}/detection_${exp_name}.log"
                
                echo "[${experiment_count}] Running: ${exp_name}"
                
                python functions/detecting.py \
                    --database "${DATABASE_PATH}" \
                    --feature_dir "${FEATURE_DIR}" \
                    --output "${results_file}" \
                    --detection_mode "fewshot" \
                    --similarity_metric "${sim_metric}" \
                    --ood_threshold "${threshold}" \
                    --k_shot "${k_shot}" \
                    --fewshot_aggregation "${agg_method}" \
                    --raw_features_aggregation "mean" \
                    --top_k_metrics 10 \
                    2>&1 | tee "${log_file}"
                
                if [ ! -f "${results_file}" ]; then
                    echo "WARNING: Detection failed for ${exp_name}"
                    continue
                fi
                
                echo "  -> Results saved to ${results_file}"
            done
        done
    done
done

echo ""
echo "Total experiments run: ${experiment_count}"
echo ""
'''

# Step 3: Evaluate all results
echo "Step 3: Evaluating all detection results..."
echo "----------------------------------------"

eval_count=0

# Find all detection result files
for results_file in "${RESULTS_DIR}"/detection_*.json; do
    if [ ! -f "${results_file}" ]; then
        continue
    fi
    
    eval_count=$((eval_count + 1))
    basename=$(basename "${results_file}" .json)
    eval_file="${EVAL_DIR}/eval_${basename}.json"
    confusion_file="${EVAL_DIR}/confusion_${basename}.png"
    log_file="${LOG_DIR}/eval_${basename}.log"
    
    echo "[${eval_count}] Evaluating: ${basename}"
    
    python functions/evaluate.py \
        --results "${results_file}" \
        --database "${DATABASE_PATH}" \
        --output "${eval_file}" \
        --confusion_matrix "${confusion_file}" \
        --top_k_values 1 3 5 10 \
        2>&1 | tee "${log_file}"
    
    if [ ! -f "${eval_file}" ]; then
        echo "WARNING: Evaluation failed for ${basename}"
        continue
    fi
    
    echo "  -> Evaluation saved to ${eval_file}"
    echo "  -> Confusion matrix saved to ${confusion_file}"
done

echo ""
echo "Total evaluations: ${eval_count}"
echo ""

# Step 4: Generate summary report
echo "Step 4: Generating summary report..."
echo "----------------------------------------"

summary_file="${EVAL_DIR}/experiment_summary.txt"
{
    echo "=========================================="
    echo "EXPERIMENT SUMMARY REPORT"
    echo "=========================================="
    echo "Generated: $(date)"
    echo ""
    echo "Total Detection Experiments: ${experiment_count}"
    echo "Total Evaluations: ${eval_count}"
    echo ""
    echo "----------------------------------------"
    echo "BEST PERFORMING CONFIGURATIONS"
    echo "----------------------------------------"
    echo ""
    
    # Extract key metrics from all evaluation files
    for eval_file in "${EVAL_DIR}"/eval_*.json; do
        if [ ! -f "${eval_file}" ]; then
            continue
        fi
        
        basename=$(basename "${eval_file}" .json | sed 's/^eval_detection_//')
        
        # Extract key metrics using Python
        python3 << EOF
import json
import sys

try:
    with open('${eval_file}', 'r') as f:
        data = json.load(f)
    
    exp_name = '${basename}'
    
    # Extract metrics
    consistency = data.get('classification_consistency', {}).get('overall_consistency', {})
    separation = data.get('dataset_separation', {}).get('separation_score', {})
    top1 = data.get('top_k_accuracy', {}).get('top_1_agreement', {})
    confidence = data.get('confidence_distribution', {}).get('overall', {})
    
    print(f"Experiment: {exp_name}")
    if consistency:
        print(f"  Consistency: {consistency.get('mean_consistency', 'N/A'):.3f}")
    if separation:
        print(f"  Separation Score: {separation.get('score', 'N/A'):.3f}")
    if top1:
        print(f"  Top-1 Agreement: {top1.get('overall_mean', 'N/A'):.3f}")
    if confidence:
        print(f"  Mean Confidence: {confidence.get('mean', 'N/A'):.3f}")
    print("")
except Exception as e:
    pass
EOF
    done
    
    echo "----------------------------------------"
    echo "Detailed results are in: ${EVAL_DIR}/"
    echo "Detection results are in: ${RESULTS_DIR}/"
    echo "Logs are in: ${LOG_DIR}/"
    echo "=========================================="
} > "${summary_file}"

echo "Summary report saved to: ${summary_file}"
echo ""

# Step 5: Create comparison plots (optional - requires matplotlib/seaborn)
echo "Step 5: Creating comparison visualizations..."
echo "----------------------------------------"

python3 << 'PYTHON_EOF'
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

eval_dir = Path("evaluation_results")
if not eval_dir.exists():
    print("Evaluation directory not found, skipping visualization")
    exit(0)

# Collect all metrics
experiments = []
consistencies = []
separation_scores = []
top1_agreements = []
mean_confidences = []

for eval_file in eval_dir.glob("eval_*.json"):
    try:
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        exp_name = eval_file.stem.replace("eval_detection_", "")
        experiments.append(exp_name)
        
        consistency = data.get('classification_consistency', {}).get('overall_consistency', {})
        separation = data.get('dataset_separation', {}).get('separation_score', {})
        top1 = data.get('top_k_accuracy', {}).get('top_1_agreement', {})
        confidence = data.get('confidence_distribution', {}).get('overall', {})
        
        consistencies.append(consistency.get('mean_consistency', 0) if consistency else 0)
        separation_scores.append(separation.get('score', 0) if separation else 0)
        top1_agreements.append(top1.get('overall_mean', 0) if top1 else 0)
        mean_confidences.append(confidence.get('mean', 0) if confidence else 0)
    except:
        continue

if len(experiments) == 0:
    print("No evaluation files found, skipping visualization")
    exit(0)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Experiment Comparison', fontsize=16)

# Truncate experiment names for readability
exp_names_short = [name[:40] + '...' if len(name) > 40 else name for name in experiments]

# Plot 1: Consistency
axes[0, 0].barh(range(len(consistencies)), consistencies)
axes[0, 0].set_yticks(range(len(exp_names_short)))
axes[0, 0].set_yticklabels(exp_names_short, fontsize=8)
axes[0, 0].set_xlabel('Mean Consistency')
axes[0, 0].set_title('Classification Consistency')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Separation Score
axes[0, 1].barh(range(len(separation_scores)), separation_scores)
axes[0, 1].set_yticks(range(len(exp_names_short)))
axes[0, 1].set_yticklabels(exp_names_short, fontsize=8)
axes[0, 1].set_xlabel('Separation Score')
axes[0, 1].set_title('Dataset Separation')
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Top-1 Agreement
axes[1, 0].barh(range(len(top1_agreements)), top1_agreements)
axes[1, 0].set_yticks(range(len(exp_names_short)))
axes[1, 0].set_yticklabels(exp_names_short, fontsize=8)
axes[1, 0].set_xlabel('Top-1 Agreement')
axes[1, 0].set_title('Top-1 Agreement Score')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Mean Confidence
axes[1, 1].barh(range(len(mean_confidences)), mean_confidences)
axes[1, 1].set_yticks(range(len(exp_names_short)))
axes[1, 1].set_yticklabels(exp_names_short, fontsize=8)
axes[1, 1].set_xlabel('Mean Confidence (Similarity)')
axes[1, 1].set_title('Confidence Distribution')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
output_path = eval_dir / "experiment_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison plot saved to: {output_path}")
PYTHON_EOF

echo ""
echo "=========================================="
echo "Experiment Loop Completed!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo "  - Detection results: ${RESULTS_DIR}/"
echo "  - Evaluation results: ${EVAL_DIR}/"
echo "  - Logs: ${LOG_DIR}/"
echo "  - Summary report: ${EVAL_DIR}/experiment_summary.txt"
echo ""
echo "To view the summary:"
echo "  cat ${EVAL_DIR}/experiment_summary.txt"
echo ""

