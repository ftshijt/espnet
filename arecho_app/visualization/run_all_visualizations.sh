#!/bin/bash
# Script to iterate through all visualization options
#
# Usage:
#   ./run_all_visualizations.sh [feature_dir]
#
# Examples:
#   ./run_all_visualizations.sh feature
#   ./run_all_visualizations.sh feature_10k

# Don't exit on error - continue with remaining visualizations
set +e

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [feature_dir]"
    echo ""
    echo "Iterates through all visualization options and generates plots."
    echo ""
    echo "Arguments:"
    echo "  feature_dir    Path to feature directory (default: 'feature')"
    echo ""
    echo "Output:"
    echo "  All plots are saved to 'visualization_outputs/' directory"
    echo ""
    exit 0
fi

# Configuration
FEATURE_DIR="${1:-feature}"  # Default to "feature" if not provided
OUTPUT_DIR="visualization_outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISUALIZE_SCRIPT="${SCRIPT_DIR}/visualize.py"

# Check if feature directory exists
if [[ ! -d "${FEATURE_DIR}" ]]; then
    echo "Error: Feature directory '${FEATURE_DIR}' does not exist!"
    echo "Usage: $0 [feature_dir]"
    exit 1
fi

# Check if visualize script exists
if [[ ! -f "${VISUALIZE_SCRIPT}" ]]; then
    echo "Error: Visualization script '${VISUALIZE_SCRIPT}' not found!"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Running all visualization combinations"
echo "Feature directory: ${FEATURE_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Counter for tracking progress
total=0
completed=0

# Function to run visualization and track progress
run_viz() {
    local cmd="$1"
    local description="$2"
    total=$((total + 1))
    
    echo "[$total] $description"
    echo "Command: $cmd"
    
    if eval "$cmd"; then
        completed=$((completed + 1))
        echo "✓ Success"
    else
        echo "✗ Failed"
    fi
    echo ""
}

# ============================================
# FEATURE-BASED VISUALIZATIONS
# ============================================
echo "=========================================="
echo "FEATURE-BASED VISUALIZATIONS"
echo "=========================================="
echo ""

# Different aggregation methods without statistics
for agg in mean raw; do
    output_file="${OUTPUT_DIR}/features_${agg}.png"
    run_viz \
        "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode features --aggregation ${agg} --output ${output_file}" \
        "Features: aggregation=${agg}"
done

# Different aggregation methods with statistics
for agg in mean raw; do
    output_file="${OUTPUT_DIR}/features_${agg}_stats.png"
    run_viz \
        "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode features --aggregation ${agg} --use_statistics --output ${output_file}" \
        "Features: aggregation=${agg}, with dataset statistics"
done

# ============================================
# METRIC-BASED VISUALIZATIONS
# ============================================
echo "=========================================="
echo "METRIC-BASED VISUALIZATIONS"
echo "=========================================="
echo ""

# Basic metrics (numerical only)
output_file="${OUTPUT_DIR}/metrics_numerical.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --output ${output_file}" \
    "Metrics: numerical only"

# Metrics with classification
output_file="${OUTPUT_DIR}/metrics_with_classification.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --use_classification --output ${output_file}" \
    "Metrics: with classification (categorical encoded)"

# Metrics normalized
output_file="${OUTPUT_DIR}/metrics_normalized.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --normalize --output ${output_file}" \
    "Metrics: normalized"

# Metrics with classification and normalized
output_file="${OUTPUT_DIR}/metrics_classification_normalized.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --use_classification --normalize --output ${output_file}" \
    "Metrics: with classification and normalized"

# Metrics with dataset statistics
output_file="${OUTPUT_DIR}/metrics_stats.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --metric_statistics --output ${output_file}" \
    "Metrics: dataset statistics"

# Metrics with classification and dataset statistics
output_file="${OUTPUT_DIR}/metrics_classification_stats.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --use_classification --metric_statistics --output ${output_file}" \
    "Metrics: with classification and dataset statistics"

# Metrics normalized with dataset statistics
output_file="${OUTPUT_DIR}/metrics_normalized_stats.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --normalize --metric_statistics --output ${output_file}" \
    "Metrics: normalized with dataset statistics"

# Metrics with all options
output_file="${OUTPUT_DIR}/metrics_full.png"
run_viz \
    "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --use_classification --normalize --metric_statistics --output ${output_file}" \
    "Metrics: full options (classification + normalized + statistics)"

# ============================================
# OPTIONAL: Different t-SNE parameters
# ============================================
echo "=========================================="
echo "OPTIONAL: Different t-SNE parameters"
echo "=========================================="
echo ""

# Different perplexity values for features (mean aggregation)
for perp in 15 30 50; do
    output_file="${OUTPUT_DIR}/features_mean_perplexity${perp}.png"
    run_viz \
        "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode features --aggregation mean --perplexity ${perp} --output ${output_file}" \
        "Features: mean aggregation, perplexity=${perp}"
done

# Different perplexity values for metrics (with classification and normalized)
for perp in 15 30 50; do
    output_file="${OUTPUT_DIR}/metrics_full_perplexity${perp}.png"
    run_viz \
        "python3 ${VISUALIZE_SCRIPT} --feature_dir ${FEATURE_DIR} --mode metrics --use_classification --normalize --perplexity ${perp} --output ${output_file}" \
        "Metrics: full options, perplexity=${perp}"
done

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total visualizations attempted: $total"
echo "Successfully completed: $completed"
echo "Failed: $((total - completed))"
echo ""
echo "All outputs saved to: ${OUTPUT_DIR}/"
echo ""

# Count generated files
png_count=$(ls -1 "${OUTPUT_DIR}"/*.png 2>/dev/null | wc -l)
if [[ $png_count -gt 0 ]]; then
    echo "Generated files: $png_count PNG files"
    echo ""
    echo "File list:"
    ls -lh "${OUTPUT_DIR}"/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
else
    echo "No PNG files generated."
fi
echo ""

