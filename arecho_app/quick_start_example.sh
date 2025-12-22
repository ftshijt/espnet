#!/bin/bash
# Quick start example script for Interactive Audio Analyzer

echo "=========================================="
echo "Interactive Audio Analyzer - Quick Start"
echo "=========================================="
echo ""

# Check if database exists
if [ ! -f "database.pkl" ]; then
    echo "ERROR: database.pkl not found!"
    echo "Please build the database first:"
    echo "  python functions/database_construct.py --feature_dir feature --output database.pkl"
    exit 1
fi

echo "Database found: database.pkl"
echo ""

# Example 1: Basic usage (if audio files exist)
echo "Example 1: Basic Analysis"
echo "-------------------------"
echo "To analyze audio files, run:"
echo "  python interactive_audio_analyzer.py --audio_files file1.wav file2.wav"
echo ""

# Example 2: Interactive mode
echo "Example 2: Interactive Mode"
echo "---------------------------"
echo "For interactive mode, run:"
echo "  python interactive_audio_analyzer.py --interactive"
echo ""

# Example 3: With LLM summary
echo "Example 3: With LLM Summary"
echo "--------------------------"
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Note: Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable LLM summarization"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo "  python interactive_audio_analyzer.py --audio_files *.wav --generate_summary"
else
    echo "API key found! You can use:"
    echo "  python interactive_audio_analyzer.py --audio_files *.wav --generate_summary"
fi
echo ""

# Example 4: Analyze results
echo "Example 4: Analyze Results"
echo "-------------------------"
echo "After running analysis, create visualizations:"
echo "  python analyze_results.py analysis_results.json"
echo ""

# Example 5: Different detection modes
echo "Example 5: Try Different Detection Modes"
echo "----------------------------------------"
echo "Feature-based (fastest):"
echo "  python interactive_audio_analyzer.py --audio_files *.wav --detection_mode feature"
echo ""
echo "Metric-based (more detailed):"
echo "  python interactive_audio_analyzer.py --audio_files *.wav --detection_mode metric --metric_mode with_classification"
echo ""
echo "Raw features (most accurate, slowest):"
echo "  python interactive_audio_analyzer.py --audio_files *.wav --detection_mode raw_features"
echo ""

echo "=========================================="
echo "For more examples, see:"
echo "  - example_usage.py"
echo "  - INTERACTIVE_ANALYZER_README.md"
echo "=========================================="

