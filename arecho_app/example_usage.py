#!/usr/bin/env python3
"""
Example usage of the Interactive Audio Analyzer

This script demonstrates how to use the audio analyzer with example audio files.
"""

import json
from pathlib import Path
from interactive_audio_analyzer import AudioAnalyzer, print_analysis_summary, save_results

def example_basic_usage():
    """Basic example: analyze a few audio files."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Example audio files (replace with actual paths)
    audio_files = [
        # Add your audio file paths here
        # "/path/to/audio1.wav",
        # "/path/to/audio2.wav",
    ]
    
    # If no files provided, show how to use it
    if not audio_files or not all(Path(f).exists() for f in audio_files):
        print("\nTo use this example, provide actual audio file paths:")
        print("  audio_files = ['/path/to/audio1.wav', '/path/to/audio2.wav']")
        return
    
    # Initialize analyzer
    with AudioAnalyzer(
        database_path="database.pkl",
        model_name="espnet/arecho_base_v0",
        device="cuda"
    ) as analyzer:
        
        # Analyze files
        results = analyzer.analyze_audio_files(
            audio_files,
            detection_mode="feature",
            similarity_metric="cosine",
            ood_threshold=0.5
        )
        
        # Print summary
        print_analysis_summary(results)
        
        # Save results
        save_results(results, "example_basic_results.json")


def example_metric_mode():
    """Example using metric-based detection."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Metric-Based Detection")
    print("="*80)
    
    audio_files = [
        # Add your audio file paths here
    ]
    
    if not audio_files or not all(Path(f).exists() for f in audio_files):
        print("\nTo use this example, provide actual audio file paths")
        return
    
    with AudioAnalyzer(
        database_path="database.pkl",
        model_name="espnet/arecho_base_v0",
        device="cuda"
    ) as analyzer:
        
        # Use metric mode with classification
        results = analyzer.analyze_audio_files(
            audio_files,
            detection_mode="metric",
            metric_mode="with_classification",
            similarity_metric="cosine",
            ood_threshold=0.5
        )
        
        # Print summary
        print_analysis_summary(results)
        
        # Show metric differences for first result
        if results and 'metric_differences' in results[0]:
            print("\nTop metric differences for first file:")
            for diff in results[0]['metric_differences'][:5]:
                print(f"  {diff['metric']}: z-score={diff['z_score']:.2f}, "
                      f"sample={diff['sample_value']:.2f}, train_mean={diff['train_mean']:.2f}")


def example_with_llm_summary():
    """Example with LLM summarization."""
    print("\n" + "="*80)
    print("EXAMPLE 3: With LLM Summarization")
    print("="*80)
    
    audio_files = [
        # Add your audio file paths here
    ]
    
    if not audio_files or not all(Path(f).exists() for f in audio_files):
        print("\nTo use this example, provide actual audio file paths")
        print("Also ensure OPENAI_API_KEY or ANTHROPIC_API_KEY is set")
        return
    
    with AudioAnalyzer(
        database_path="database.pkl",
        model_name="espnet/arecho_base_v0",
        device="cuda"
    ) as analyzer:
        
        # Analyze files
        results = analyzer.analyze_audio_files(
            audio_files,
            detection_mode="feature",
            similarity_metric="cosine",
            ood_threshold=0.5
        )
        
        # Generate LLM summary
        if analyzer.llm_client:
            print("\nGenerating LLM summary...")
            summary = analyzer.generate_summary(results)
            if summary:
                print("\n" + "="*80)
                print("LLM SUMMARY")
                print("="*80)
                print(summary)
                
                # Save with summary
                save_results(results, "example_llm_results.json", summary=summary)
        else:
            print("\nLLM not available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


def example_batch_analysis():
    """Example: analyze multiple files with different settings."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Analysis with Different Settings")
    print("="*80)
    
    audio_files = [
        # Add your audio file paths here
    ]
    
    if not audio_files or not all(Path(f).exists() for f in audio_files):
        print("\nTo use this example, provide actual audio file paths")
        return
    
    # Try different detection modes
    detection_modes = ["feature", "metric", "raw_features"]
    
    all_results = {}
    
    with AudioAnalyzer(
        database_path="database.pkl",
        model_name="espnet/arecho_base_v0",
        device="cuda"
    ) as analyzer:
        
        for mode in detection_modes:
            print(f"\nAnalyzing with detection_mode='{mode}'...")
            
            metric_mode = "numerical_only" if mode == "metric" else None
            
            results = analyzer.analyze_audio_files(
                audio_files,
                detection_mode=mode,
                metric_mode=metric_mode or "numerical_only",
                similarity_metric="cosine",
                ood_threshold=0.5
            )
            
            all_results[mode] = results
            
            # Print summary for this mode
            print(f"\nResults for {mode}:")
            print_analysis_summary(results)
        
        # Compare results
        print("\n" + "="*80)
        print("COMPARISON ACROSS MODES")
        print("="*80)
        
        for mode, results in all_results.items():
            ood_count = sum(1 for r in results if r.get('is_out_of_domain', False))
            print(f"{mode}: {ood_count}/{len(results)} OOD ({ood_count/len(results)*100:.1f}%)")


def example_detailed_analysis():
    """Example: detailed per-file analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Detailed Per-File Analysis")
    print("="*80)
    
    audio_files = [
        # Add your audio file paths here
    ]
    
    if not audio_files or not all(Path(f).exists() for f in audio_files):
        print("\nTo use this example, provide actual audio file paths")
        return
    
    with AudioAnalyzer(
        database_path="database.pkl",
        model_name="espnet/arecho_base_v0",
        device="cuda"
    ) as analyzer:
        
        # Analyze each file individually for detailed output
        for audio_file in audio_files:
            print(f"\n{'='*80}")
            print(f"Analyzing: {Path(audio_file).name}")
            print('='*80)
            
            result = analyzer.analyze_audio_file(
                audio_file,
                detection_mode="feature",
                similarity_metric="cosine",
                ood_threshold=0.5
            )
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            print(f"\nStatus: {'OUT-OF-DOMAIN' if result.get('is_out_of_domain') else 'IN-DOMAIN'}")
            
            best_match = result.get('best_match', {})
            if best_match and best_match.get('dataset'):
                print(f"Best match: {best_match['dataset']} (similarity: {best_match['similarity']:.3f})")
            
            print("\nTop 10 similar datasets:")
            for i, match in enumerate(result.get('top_similar_datasets', [])[:10], 1):
                print(f"  {i:2d}. {match['dataset']:30s} {match['similarity']:.4f}")


if __name__ == "__main__":
    print("Interactive Audio Analyzer - Example Usage")
    print("="*80)
    print("\nNote: These examples require actual audio files.")
    print("Replace the audio_files lists with real file paths to run.")
    print("\nAvailable examples:")
    print("  1. Basic usage")
    print("  2. Metric-based detection")
    print("  3. With LLM summarization")
    print("  4. Batch analysis with different settings")
    print("  5. Detailed per-file analysis")
    
    # Uncomment the example you want to run:
    # example_basic_usage()
    # example_metric_mode()
    # example_with_llm_summary()
    # example_batch_analysis()
    # example_detailed_analysis()
    
    print("\n" + "="*80)
    print("To run examples, uncomment the function calls in the script")
    print("="*80)


