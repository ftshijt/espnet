# Audio Analysis and Detection Framework

A comprehensive framework for analyzing audio files, detecting similarities against training datasets, and evaluating detection performance. This system provides both an interactive interface for analyzing individual audio files and a complete experiment pipeline for systematic evaluation.

## Overview

This framework consists of two main components:

1. **Interactive Audio Analyzer**: A user-friendly tool for analyzing audio files interactively, extracting features, calculating similarities, detecting out-of-domain samples, and generating LLM-powered summaries.

2. **Experiment Pipeline**: A systematic framework for building databases from training data, running detection experiments with various configurations, and evaluating performance with comprehensive metrics.

Both components share the same underlying detection concepts and can be used together or independently.

## Features

### Interactive Audio Analyzer
- **Feature Extraction**: Extract features from audio files using ESPnet's Universa model
- **Similarity Detection**: Compare audio files against training datasets using multiple detection modes:
  - Feature-based detection
  - Metric-based detection
  - Raw features detection
- **Out-of-Domain Detection**: Identify audio files that are significantly different from training data
- **LLM Summarization**: Generate AI-powered summaries of analysis results (Qwen, OpenAI, or Anthropic)
- **Interactive Mode**: User-friendly interactive interface
- **Batch Processing**: Analyze multiple audio files at once
- **Visualization**: Create plots and detailed analysis reports

### Experiment Pipeline
- **Database Construction**: Build statistics database from training datasets
- **Systematic Detection**: Run detection with multiple configuration combinations
- **Comprehensive Evaluation**: Evaluate detection performance with detailed metrics
- **Automated Experiments**: Run all experiment combinations automatically
- **Result Analysis**: Generate visualizations and comparison reports

## Installation

### Prerequisites

1. Python 3.8+
2. CUDA-capable GPU (recommended) or CPU
3. Database file (`database.pkl`) - should be generated from training data
4. ESPnet environment with Universa model

### Required Packages

```bash
pip install numpy torch librosa soundfile tqdm scikit-learn

# For LLM summarization (optional - choose one or more)
pip install openai  # For OpenAI API
pip install anthropic  # For Anthropic API
pip install transformers accelerate  # For Qwen (open-source, local inference)

# For visualization (optional)
pip install matplotlib seaborn
```

## Quick Start

### Interactive Audio Analyzer

#### Basic Usage

Analyze a single audio file or multiple files:

```bash
# Analyze specific files
python interactive_audio_analyzer.py --audio_files file1.wav file2.wav file3.wav

# Analyze all audio files in a directory
python interactive_audio_analyzer.py --audio_dir /path/to/audio/files

# Interactive mode
python interactive_audio_analyzer.py --interactive
```

#### With Custom Settings

```bash
python interactive_audio_analyzer.py \
    --audio_files file1.wav file2.wav \
    --detection_mode metric \
    --metric_mode with_classification \
    --similarity_metric cosine \
    --ood_threshold 0.5 \
    --output results.json
```

#### With LLM Summarization

**Option 1: Using Qwen (Open-Source, Local, Recommended for Easy Start)**

Qwen works offline and doesn't require API keys. It's enabled by default:

```bash
# Uses Qwen2.5-0.5B-Instruct by default (lightweight, easy to use)
python interactive_audio_analyzer.py \
    --audio_files file1.wav file2.wav \
    --generate_summary \
    --output results.json

# Use a different Qwen model
python interactive_audio_analyzer.py \
    --audio_files file1.wav file2.wav \
    --generate_summary \
    --llm_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --output results.json
```

**Option 2: Using OpenAI API**

```bash
export OPENAI_API_KEY="your-api-key-here"
python interactive_audio_analyzer.py \
    --audio_files file1.wav file2.wav \
    --generate_summary \
    --output results.json
```

**Option 3: Using Anthropic API**

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
python interactive_audio_analyzer.py \
    --audio_files file1.wav file2.wav \
    --generate_summary \
    --output results.json
```

**Note**: If API keys are set, they take priority. Otherwise, Qwen is used automatically.

### Experiment Pipeline

#### Run All Experiments

Run all experiments with different configurations:

```bash
./run_experiments.sh
```

This will:
- Build the database from training datasets
- Run detection with all configuration combinations
- Evaluate all results
- Generate summary reports and visualizations

#### Manual Usage

**Step 1: Build Database**

```bash
python functions/database_construct.py \
    --feature_dir feature \
    --output database.pkl
```

**Step 2: Run Detection**

```bash
# Metric-based detection
python functions/detecting.py \
    --database database.pkl \
    --feature_dir feature \
    --output detection_results.json \
    --detection_mode metric \
    --metric_mode numerical_only_normalized \
    --similarity_metric cosine \
    --ood_threshold 0.5

# Feature-based detection
python functions/detecting.py \
    --database database.pkl \
    --feature_dir feature \
    --output detection_results.json \
    --detection_mode feature \
    --similarity_metric cosine \
    --ood_threshold 0.5
```

**Step 3: Evaluate Results**

```bash
python functions/evaluate.py \
    --results detection_results.json \
    --database database.pkl \
    --output evaluation_results.json \
    --confusion_matrix confusion_matrix.png \
    --top_k_values 1 3 5 10
```

## Interactive Audio Analyzer - Detailed Guide

### Command-Line Options

#### Main Options

- `--audio_files`: List of audio file paths to analyze
- `--audio_dir`: Directory containing audio files to analyze
- `--interactive`: Run in interactive mode
- `--database`: Path to database pickle file (default: `database.pkl`)
- `--model`: Model name (default: `espnet/arecho_base_v0`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--output`: Output JSON file path (default: `analysis_results.json`)

#### Detection Options

- `--detection_mode`: Detection mode - `feature`, `metric`, or `raw_features` (default: `feature`)
- `--metric_mode`: Metric mode - `numerical_only`, `with_classification`, etc. (only if `detection_mode=metric`)
- `--similarity_metric`: Similarity metric - `cosine` or `euclidean` (default: `cosine`)
- `--ood_threshold`: Out-of-domain threshold 0.0-1.0 (default: `0.5`)
- `--generate_summary`: Generate LLM summary (uses Qwen by default, or API if keys are set)
- `--llm_model`: Qwen model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct"). Default: Qwen2.5-0.5B-Instruct
- `--use_qwen`: Use Qwen as default LLM if no API keys found (default: True)
- `--no_qwen`: Disable Qwen LLM (use only API-based LLMs)
- `--max_files`: Maximum number of files to process (for testing)

### Detection Modes

#### Feature-Based Detection (`--detection_mode feature`)

Uses aggregated feature statistics to compare against training datasets. Good for general similarity detection.

#### Metric-Based Detection (`--detection_mode metric`)

Uses extracted metrics (numerical and/or categorical) for comparison. Provides detailed metric differences.

**Metric Modes:**
- `numerical_only`: Only numerical metrics
- `with_classification`: Includes categorical metrics
- `numerical_only_normalized`: Normalized numerical metrics
- `with_classification_normalized`: Normalized with classification

#### Raw Features Detection (`--detection_mode raw_features`)

Direct comparison with individual training examples. More computationally intensive but potentially more accurate.

### Usage Examples

#### Example 1: Basic Analysis

```python
from interactive_audio_analyzer import AudioAnalyzer, print_analysis_summary

# Initialize analyzer
with AudioAnalyzer(
    database_path="database.pkl",
    model_name="espnet/arecho_base_v0",
    device="cuda"
) as analyzer:
    
    # Analyze files
    results = analyzer.analyze_audio_files(
        ["audio1.wav", "audio2.wav", "audio3.wav"],
        detection_mode="feature",
        similarity_metric="cosine",
        ood_threshold=0.5
    )
    
    # Print summary
    print_analysis_summary(results)
```

#### Example 2: Metric-Based Detection

```python
with AudioAnalyzer(database_path="database.pkl") as analyzer:
    results = analyzer.analyze_audio_files(
        ["audio1.wav", "audio2.wav"],
        detection_mode="metric",
        metric_mode="with_classification",
        similarity_metric="cosine",
        ood_threshold=0.5
    )
    
    # Check metric differences
    for result in results:
        if 'metric_differences' in result:
            print(f"\n{result['audio_name']}:")
            for diff in result['metric_differences'][:5]:
                print(f"  {diff['metric']}: z-score={diff['z_score']:.2f}")
```

#### Example 3: With LLM Summary (Qwen - Open Source)

```python
# Qwen is used by default (no API key needed)
with AudioAnalyzer(
    database_path="database.pkl",
    llm_model="Qwen/Qwen2.5-0.5B-Instruct"  # Optional: specify model
) as analyzer:
    results = analyzer.analyze_audio_files(["audio1.wav", "audio2.wav"])
    
    # Generate summary using Qwen
    if analyzer.llm_client:
        summary = analyzer.generate_summary(results)
        print(summary)
```

#### Example 4: With LLM Summary (OpenAI/Anthropic API)

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

with AudioAnalyzer(database_path="database.pkl", use_qwen=False) as analyzer:
    results = analyzer.analyze_audio_files(["audio1.wav", "audio2.wav"])
    
    # Generate summary
    if analyzer.llm_client:
        summary = analyzer.generate_summary(results)
        print(summary)
```

### Analyzing Results

Use the `analyze_results.py` script to create visualizations and detailed reports:

```bash
# Generate analysis report and plots
python analyze_results.py results.json

# Save detailed report
python analyze_results.py results.json --report detailed_report.json

# Compare multiple result files
python analyze_results.py results.json --compare results1.json results2.json results3.json
```

### Output Format

The analysis results are saved as JSON with the following structure:

```json
{
  "results": [
    {
      "audio_path": "path/to/audio.wav",
      "audio_name": "audio",
      "detection_mode": "feature",
      "similarity_metric": "cosine",
      "is_out_of_domain": false,
      "best_match": {
        "dataset": "LibriSpeech",
        "similarity": 0.85
      },
      "top_similar_datasets": [
        {"dataset": "LibriSpeech", "similarity": 0.85},
        {"dataset": "CommonVoice", "similarity": 0.72},
        ...
      ],
      "all_similarities": {
        "LibriSpeech": 0.85,
        "CommonVoice": 0.72,
        ...
      }
    }
  ],
  "summary": "LLM-generated summary (if available)",
  "metadata": {
    "total_files": 10,
    "successful": 10,
    "ood_count": 2
  }
}
```

### Understanding Results

#### Similarity Scores

- **Range**: 0.0 to 1.0 (higher is more similar)
- **Interpretation**:
  - > 0.7: Very similar to training data
  - 0.5-0.7: Moderately similar
  - < 0.5: Out-of-domain (different from training data)

#### Out-of-Domain Detection

A file is marked as out-of-domain if its best similarity score is below the threshold (default: 0.5). This indicates the audio is significantly different from the training datasets.

#### Best Match

The training dataset with the highest similarity score. This indicates which training dataset the audio is most similar to.

### LLM Options Comparison

| Option | Pros | Cons | Setup |
|--------|------|------|-------|
| **Qwen (Default)** | Free, offline, no API keys | Requires local storage (~1-3GB), slower on CPU | `pip install transformers accelerate` |
| **OpenAI API** | Fast, high quality | Requires API key, costs money, needs internet | Set `OPENAI_API_KEY` |
| **Anthropic API** | High quality | Requires API key, costs money, needs internet | Set `ANTHROPIC_API_KEY` |

**Recommendation**: Start with Qwen for easy setup. Use API-based options for better quality or faster inference.

## Experiment Pipeline - Detailed Guide

### Experiment Configurations

The `run_experiments.sh` script runs all combinations of:

#### Detection Modes
- **metric**: Uses metric distributions for similarity
- **feature**: Uses feature distributions for similarity

#### Metric Modes (for metric detection)
- **numerical_only**: Only numerical metrics
- **with_classification**: Numerical + categorical metrics (encoded)
- **numerical_only_normalized**: Normalized numerical metrics
- **with_classification_normalized**: Normalized with classification

#### Similarity Metrics
- **cosine**: Cosine similarity
- **euclidean**: Euclidean distance (converted to similarity)

#### OOD Thresholds
- **0.3**: Strict (more samples marked as OOD)
- **0.5**: Moderate (default)
- **0.7**: Lenient (fewer samples marked as OOD)

**Total experiments**: 4 metric modes × 2 similarity metrics × 3 thresholds + 2 similarity metrics × 3 thresholds = 30 experiments

### Evaluation Metrics

The evaluation script computes the following metrics:

#### 1. Classification Consistency

**Question**: Are samples from the same test dataset classified similarly?

**Metrics**:
- **Per-dataset consistency**: For each test dataset, what percentage of samples have the same best match?
- **Overall consistency**: Mean consistency across all test datasets
- **Match distribution**: Which training datasets are matched to each test dataset

**Interpretation**:
- High consistency (>0.7): Samples from same dataset cluster together well
- Low consistency (<0.5): High variability in classification

#### 2. Dataset Separation

**Question**: Can different test datasets be distinguished from each other?

**Metrics**:
- **Intra-dataset similarity**: Mean similarity within each test dataset
- **Inter-dataset similarity**: Mean similarity difference between different test datasets
- **Separation score**: Ratio of inter-dataset difference to intra-dataset variance
  - Higher score = better separation
  - Score > 1.0 indicates good separation

**Interpretation**:
- High separation score: Different datasets are well-distinguished
- Low separation score: Datasets overlap significantly

#### 3. Top-K Agreement

**Question**: Do samples from the same test dataset agree on their top-k matches?

**Metrics**:
- **Top-k agreement**: Jaccard similarity of top-k matches between samples from same dataset
- Computed for k = 1, 3, 5, 10

**Interpretation**:
- High agreement (>0.6): Consistent matching patterns
- Low agreement (<0.3): High variability in matches

#### 4. Confidence Distribution

**Question**: What is the distribution of similarity/confidence scores?

**Metrics**:
- **Overall statistics**: Mean, std, min, max, median of all similarity scores
- **In-domain vs Out-of-domain**: Separate statistics for OOD and in-domain samples

**Interpretation**:
- Large gap between in-domain and OOD means: Good OOD detection
- Overlapping distributions: Difficult to distinguish OOD samples

#### 5. Confusion Matrix

**Question**: Which test datasets are confused with which training datasets?

**Visualization**: Heatmap showing test dataset (rows) vs predicted best match (columns)

**Interpretation**:
- Diagonal patterns: Good matching (test datasets match to similar training datasets)
- Off-diagonal patterns: Confusion between datasets

#### 6. Metric Importance

**Question**: Which metrics are most frequently identified as different?

**Metrics**:
- **Frequency**: How often each metric appears in top-k differences
- **Mean z-score**: Average z-score (standard deviations from training mean)
- **Max z-score**: Maximum z-score observed

**Interpretation**:
- High frequency + high z-score: Important discriminative metric
- Useful for understanding what makes datasets different

### Output Files

#### Detection Results (`detection_*.json`)

Each result contains:
- `dataset`: Test dataset name
- `utterance_id`: Sample ID
- `is_out_of_domain`: OOD flag
- `best_match`: Best matching training dataset and similarity score
- `top_similar_datasets`: Top 5 similar datasets
- `metric_differences`: Top-k most different metrics (for metric mode)

#### Evaluation Results (`eval_*.json`)

Contains all evaluation metrics:
- `summary`: Overall statistics
- `classification_consistency`: Consistency metrics
- `dataset_separation`: Separation metrics
- `top_k_accuracy`: Top-k agreement scores
- `confidence_distribution`: Confidence statistics
- `confusion_matrix`: Confusion matrix data
- `metric_importance`: Most important metrics

#### Visualizations
- `confusion_*.png`: Confusion matrix heatmaps
- `experiment_comparison.png`: Comparison across all experiments

### Interpreting Results

#### Best Configuration Selection

Look for configurations with:
1. **High classification consistency** (>0.7): Samples from same dataset cluster together
2. **High separation score** (>1.0): Different datasets are distinguishable
3. **High top-1 agreement** (>0.6): Consistent matching
4. **Clear OOD separation**: Large gap between in-domain and OOD confidence means

#### Common Patterns

**Good Performance**:
- High consistency + high separation + clear OOD gap
- Confusion matrix shows clear patterns (not random)

**Poor Performance**:
- Low consistency + low separation
- All samples marked as OOD (threshold too strict)
- No clear patterns in confusion matrix

**Configuration Recommendations**:
- **For OOD detection**: Use normalized metrics with moderate threshold (0.5)
- **For dataset matching**: Use feature-based or classification metrics
- **For interpretability**: Use metric mode to see which metrics differ

### Advanced Usage

#### Custom Train/Test Split

```bash
python functions/database_construct.py \
    --feature_dir feature \
    --output database.pkl \
    --train_datasets LibriSpeech AISHELL-1 commonvoice
```

#### Custom Test Datasets

```bash
python functions/detecting.py \
    --database database.pkl \
    --feature_dir feature \
    --output detection_results.json \
    --test_datasets maritime_11 maritime_12
```

#### Focused Evaluation

```bash
# Only evaluate specific metrics
python functions/evaluate.py \
    --results detection_results.json \
    --database database.pkl \
    --output evaluation_results.json \
    --top_k_values 1 5
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from interactive_audio_analyzer import AudioAnalyzer

analyzer = AudioAnalyzer(database_path="database.pkl")

# Analyze single file with custom settings
result = analyzer.analyze_audio_file(
    "audio.wav",
    detection_mode="metric",
    metric_mode="with_classification",
    similarity_metric="euclidean",
    ood_threshold=0.6
)

# Access detailed information
print(f"Best match: {result['best_match']['dataset']}")
print(f"Similarity: {result['best_match']['similarity']}")
print(f"Top 5 matches: {result['top_similar_datasets'][:5]}")

analyzer.cleanup()
```

### Batch Processing with Different Settings

```python
detection_modes = ["feature", "metric", "raw_features"]
all_results = {}

for mode in detection_modes:
    with AudioAnalyzer(database_path="database.pkl") as analyzer:
        results = analyzer.analyze_audio_files(
            audio_files,
            detection_mode=mode,
            similarity_metric="cosine"
        )
        all_results[mode] = results
```

## File Structure

```
.
├── functions/
│   ├── database_construct.py  # Database construction
│   ├── detecting.py            # Detection script
│   └── evaluate.py             # Evaluation script
├── feature/                    # Feature directory
│   ├── dataset1/
│   │   ├── sample1.json
│   │   ├── sample1_encoded_feat.npz
│   │   └── ...
│   └── dataset2/
│       └── ...
├── database.pkl                # Built database
├── interactive_audio_analyzer.py  # Interactive analyzer
├── analyze_results.py          # Analysis utilities
├── example_usage.py            # Usage examples
├── experiment_results/         # Detection results
├── evaluation_results/         # Evaluation results
├── logs/                       # Experiment logs
└── run_experiments.sh          # Main experiment script
```

## Troubleshooting

### Database Not Found

Ensure `database.pkl` exists. If not, build it using:

```bash
python functions/database_construct.py --feature_dir feature --output database.pkl
```

### CUDA Out of Memory

- Use `--device cpu` for CPU inference (slower)
- Process fewer files at once using `--max_files`
- Reduce batch size in the code

### LLM Not Working

**For Qwen (Open-Source):**
- Install transformers: `pip install transformers accelerate`
- First run will download the model (may take a few minutes)
- Check available disk space (models can be 1-3GB)
- Try a smaller model: `--llm_model "Qwen/Qwen2.5-0.5B-Instruct"`

**For API-based LLMs:**
- Check that API key is set: `echo $OPENAI_API_KEY` or `echo $ANTHROPIC_API_KEY`
- Verify internet connection
- Check API key permissions and quota

### Audio File Errors

- Ensure audio files are in supported formats (WAV, FLAC, MP3, etc.)
- Check file paths are correct
- Verify audio files are not corrupted

### Database Construction Fails

- Check that feature directory exists and contains dataset subdirectories
- Verify that npz and json files are present

### Detection Fails

- Ensure database file exists and is valid
- Check that test datasets are in the feature directory
- Verify file permissions

### Evaluation Fails

- Ensure detection results file exists and is valid JSON
- Check that database file matches the one used for detection

## Performance Tips

1. **GPU Usage**: Always use `--device cuda` if available (much faster)
2. **Batch Processing**: Process multiple files in one run rather than individually
3. **Detection Mode**: `feature` mode is fastest, `raw_features` is slowest but most accurate
4. **Database Size**: Larger databases take longer but provide better comparisons

## Limitations

- Requires pre-built database from training data
- LLM summarization: Qwen works offline but requires local model download (~1-3GB). API-based LLMs require internet and API keys.
- GPU recommended for reasonable processing speed (both audio analysis and LLM inference)
- Audio files must be in supported formats

## Examples and Analysis

See `example_usage.py` for more detailed examples:

```bash
python example_usage.py
```

The examples demonstrate:
1. Basic usage
2. Metric-based detection
3. LLM summarization
4. Batch analysis with different settings
5. Detailed per-file analysis

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add examples to `example_usage.py`
3. Update this README
4. Test with various audio files and settings

## License

Same as the parent ESPnet project.

## Citation

If you use this evaluation framework, please cite appropriately.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example usage scripts
3. Examine the code comments for detailed documentation

