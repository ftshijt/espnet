#!/usr/bin/env python3
"""
Interactive Audio Analyzer

This script provides an interactive interface to:
1. Extract features from a list of audio files
2. Calculate similarity and out-of-domain detection information
3. Generate LLM-powered summaries and analysis
4. Visualize and explore results

Usage:
    python interactive_audio_analyzer.py --audio_files file1.wav file2.wav ...
    python interactive_audio_analyzer.py --audio_dir /path/to/audio/dir
    python interactive_audio_analyzer.py --interactive  # Interactive mode
"""

import numpy as np
import torch
import librosa
import soundfile
import json
import pickle
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import tempfile
import shutil

# Import from existing modules
from espnet2.bin.universa_inference import UniversaInference
from functions.detecting import (
    load_database, load_sample_features, load_sample_metrics,
    compute_similarity_by_features, compute_similarity_by_metrics,
    compute_similarity_by_raw_features_direct,
    detect_out_of_domain, detect_metric_differences, detect_metric_similarities
)

# Try to import LLM libraries (optional)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI library not found. LLM summarization will be disabled.")
    print("Install with: pip install openai")

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Try to import Qwen (open-source LLM)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("Warning: transformers library not found. Qwen LLM summarization will be disabled.")
    print("Install with: pip install transformers accelerate")


class AudioAnalyzer:
    """Main class for interactive audio analysis."""
    
    def __init__(self, 
                 database_path: str = "database.pkl",
                 model_name: str = "espnet/arecho_base_v0",
                 device: str = "cuda",
                 temp_dir: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 use_qwen: bool = True):
        """
        Initialize the audio analyzer.
        
        Args:
            database_path: Path to the database pickle file
            model_name: Name of the Universa model to use
            device: Device to use for inference ('cuda' or 'cpu')
            temp_dir: Temporary directory for storing extracted features
            llm_model: LLM model name (for Qwen, e.g., "Qwen/Qwen2.5-0.5B-Instruct")
            use_qwen: Whether to use Qwen as default LLM if no API keys found
        """
        self.database_path = database_path
        self.model_name = model_name
        self.device = device
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="audio_analyzer_")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Load database
        print(f"Loading database from {database_path}...")
        if not Path(database_path).exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        self.database = load_database(database_path)
        print(f"Database loaded: {len(self.database['feature_stats'])} training datasets")
        
        # Initialize model
        print(f"Loading model: {model_name}...")
        self.model = UniversaInference.from_pretrained(model_name, device=device)
        print("Model loaded successfully")
        
        # LLM configuration
        self.llm_provider = None
        self.llm_client = None
        self.llm_model_name = llm_model
        self.use_qwen = use_qwen
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM client if API keys are available or use Qwen as fallback."""
        # Try OpenAI first
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = openai.OpenAI(api_key=api_key)
                self.llm_provider = "openai"
                print("OpenAI API configured")
                return
        
        # Try Anthropic
        if HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = Anthropic(api_key=api_key)
                self.llm_provider = "anthropic"
                print("Anthropic API configured")
                return
        
        # Try Qwen (open-source, local inference)
        if self.use_qwen and HAS_QWEN:
            try:
                # Default to a lightweight Qwen model if not specified
                qwen_model = self.llm_model_name or "Qwen/Qwen2.5-0.5B-Instruct"
                
                print(f"Loading Qwen model: {qwen_model}...")
                print("Note: First run will download the model (may take a few minutes)")
                
                # Determine device for LLM (use CPU if CUDA is busy or not available)
                llm_device = self.device
                if llm_device == "cuda" and not torch.cuda.is_available():
                    llm_device = "cpu"
                    print("CUDA not available for LLM, using CPU")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(qwen_model, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    qwen_model,
                    torch_dtype=torch.float16 if llm_device == "cuda" else torch.float32,
                    device_map=llm_device if llm_device == "cuda" else None,
                    trust_remote_code=True
                )
                
                if llm_device == "cpu":
                    model = model.to("cpu")
                
                self.llm_client = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': llm_device
                }
                self.llm_provider = "qwen"
                self.llm_model_name = qwen_model
                print(f"Qwen model loaded successfully on {llm_device}")
                return
            except Exception as e:
                print(f"Warning: Failed to load Qwen model: {e}")
                print("You can install with: pip install transformers accelerate")
                if self.llm_model_name:
                    print(f"Or try a different model name")
        
        if not self.use_qwen:
            print("Qwen disabled. No LLM API keys found. Summarization will be disabled.")
            print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY, or enable Qwen with --use_qwen")
        else:
            print("No LLM available. Summarization will be disabled.")
            print("Options:")
            print("  1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY for API-based LLMs")
            print("  2. Install transformers: pip install transformers accelerate")
            print("  3. Use --llm_model to specify a Qwen model")
    
    def read_audio_file(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Read audio file and return audio array and sample rate."""
        try:
            audio, sr = soundfile.read(audio_path)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Could not read audio file {audio_path}: {e}")
    
    def preprocess_audio(self, audio_array: np.ndarray, sr: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess audio for UniversaInference."""
        if sr is None:
            sr = 16000
        elif sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
        audio = audio_array.astype(np.float32)
        audio = torch.from_numpy(audio).unsqueeze(0)
        audio_lengths = torch.tensor([len(audio[0])])
        return audio, audio_lengths
    
    def extract_features(self, audio_path: str) -> Optional[Dict]:
        """Extract features from an audio file."""
        try:
            audio_array, sr = self.read_audio_file(audio_path)
            audio, audio_lengths = self.preprocess_audio(audio_array, sr)
        except Exception as e:
            print(f"Error reading audio from {audio_path}: {e}")
            return None
        
        # Move to device
        audio = audio.to(self.device)
        audio_lengths = audio_lengths.to(self.device)
        
        # Placeholder reference audio
        ref_audio = torch.zeros(1, 8000, dtype=torch.float32).to(self.device)
        ref_audio_lengths = torch.tensor([8000]).to(self.device)
        
        # Inference
        try:
            with torch.no_grad():
                result = self.model(
                    audio.float(),
                    audio_lengths,
                    ref_audio=ref_audio.float(),
                    ref_audio_lengths=ref_audio_lengths
                )
            return result
        except Exception as e:
            print(f"Error during inference for {audio_path}: {e}")
            return None
    
    def save_features_temp(self, result: Dict, utterance_id: str) -> Tuple[Path, Path]:
        """Save extracted features to temporary directory."""
        output_dir = Path(self.temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (excluding encoded_feat)
        json_path = output_dir / f"{utterance_id}.json"
        json_data = {}
        for key, value in result.items():
            if key == 'encoded_feat':
                continue
            if isinstance(value, torch.Tensor):
                json_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save encoded_feat as npz
        npz_path = output_dir / f"{utterance_id}_encoded_feat.npz"
        encoded_feat = result.get('encoded_feat', None)
        if encoded_feat is not None:
            if isinstance(encoded_feat, torch.Tensor):
                encoded_feat = encoded_feat.cpu().numpy()
            np.savez_compressed(npz_path, encoded_feat=encoded_feat.squeeze(0))
        
        return json_path, npz_path
    
    def analyze_audio_file(self, 
                          audio_path: str,
                          detection_mode: str = "feature",
                          metric_mode: str = "numerical_only",
                          similarity_metric: str = "cosine",
                          ood_threshold: float = 0.5) -> Dict:
        """
        Analyze a single audio file: extract features and compute information.
        
        Args:
            audio_path: Path to audio file
            detection_mode: 'metric', 'feature', or 'raw_features'
            metric_mode: Metric mode (if detection_mode='metric')
            similarity_metric: 'cosine' or 'euclidean'
            ood_threshold: Threshold for out-of-domain detection
        
        Returns:
            Dictionary with analysis results
        """
        audio_name = Path(audio_path).stem
        
        # Step 1: Extract features
        print(f"\nExtracting features from {audio_path}...")
        result = self.extract_features(audio_path)
        if result is None:
            return {
                'audio_path': audio_path,
                'audio_name': audio_name,
                'error': 'Failed to extract features'
            }
        
        # Save features temporarily
        json_path, npz_path = self.save_features_temp(result, audio_name)
        
        # Step 2: Compute similarities and detect OOD
        print(f"Computing similarities and detecting out-of-domain...")
        
        analysis_result = {
            'audio_path': audio_path,
            'audio_name': audio_name,
            'detection_mode': detection_mode,
            'similarity_metric': similarity_metric,
            'ood_threshold': ood_threshold
        }
        
        if detection_mode == 'feature':
            # Load features
            sample_feat = load_sample_features(self.temp_dir, "", audio_name)
            if sample_feat is None:
                analysis_result['error'] = 'Failed to load features'
                return analysis_result
            
            # Compute similarities
            similarities = compute_similarity_by_features(
                sample_feat, self.database, similarity_metric=similarity_metric
            )
            
            # Detect out-of-domain
            is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
            best_match_dataset = best_match[0] if best_match else None
        
        elif detection_mode == 'raw_features':
            sample_feat = load_sample_features(self.temp_dir, "", audio_name)
            if sample_feat is None:
                analysis_result['error'] = 'Failed to load features'
                return analysis_result
            
            similarities = compute_similarity_by_raw_features_direct(
                sample_feat, self.database, similarity_metric=similarity_metric
            )
            
            # Detect out-of-domain
            is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
            best_match_dataset = best_match[0] if best_match else None
        
        else:  # metric mode
            # Get metric names and types
            metric_names = self.database['metadata']['metric_names'].get(metric_mode, None)
            metric_types = self.database['metadata']['metric_types'].get(metric_mode, None)
            label_encoders_dict = None
            if 'with_classification' in metric_mode:
                label_encoders_dict = self.database['metadata'].get('label_encoders', None)
            
            sample_metrics = load_sample_metrics(
                self.temp_dir, "", audio_name,
                use_classification='classification' in metric_mode,
                metric_names=metric_names,
                metric_types=metric_types,
                label_encoders_dict=label_encoders_dict
            )
            if sample_metrics is None:
                analysis_result['error'] = 'Failed to load metrics'
                return analysis_result
            
            # Normalize if needed
            if 'normalized' in metric_mode:
                if 'numerical_only' in metric_mode:
                    scaler_info = self.database['metadata'].get('numerical_only_scaler', None)
                else:
                    scaler_info = self.database['metadata'].get('with_classification_scaler', None)
                
                if scaler_info:
                    sample_metrics = (sample_metrics - scaler_info['mean']) / (scaler_info['scale'] + 1e-8)
            
            # Compute similarities
            similarities = compute_similarity_by_metrics(
                sample_metrics, self.database, metric_mode=metric_mode,
                similarity_metric=similarity_metric
            )
            
            # Detect out-of-domain first to know if we should look for similarities or differences
            is_ood, best_match = detect_out_of_domain(similarities, threshold=ood_threshold)
            best_match_dataset = best_match[0] if best_match else None
            
            # For in-domain matches: find most similar metrics (leading to the match)
            # For OOD: find most different metrics (leading to OOD classification)
            if not is_ood and best_match_dataset:
                # In-domain: identify metrics most similar to best match
                metric_similarities = detect_metric_similarities(
                    sample_metrics, self.database, metric_mode=metric_mode,
                    top_k=10, metric_names=metric_names,
                    best_match_dataset=best_match_dataset
                )
                analysis_result['metric_similarities'] = [
                    {
                        'metric': name,
                        'similarity_score': float(score),
                        'match_mean': float(match_mean),
                        'sample_value': float(sample_val)
                    }
                    for name, score, match_mean, sample_val in metric_similarities
                ]
                
                # Also include differences for context (compared to best match)
                metric_differences = detect_metric_differences(
                    sample_metrics, self.database, metric_mode=metric_mode,
                    top_k=10, metric_names=metric_names,
                    best_match_dataset=best_match_dataset
                )
                analysis_result['metric_differences'] = [
                    {
                        'metric': name,
                        'z_score': float(score),
                        'match_mean': float(train_mean),
                        'sample_value': float(sample_val)
                    }
                    for name, score, train_mean, sample_val in metric_differences
                ]
            else:
                # OOD: identify metrics most different from training data
                metric_differences = detect_metric_differences(
                    sample_metrics, self.database, metric_mode=metric_mode,
                    top_k=10, metric_names=metric_names,
                    best_match_dataset=None  # Compare to average training data
                )
                analysis_result['metric_differences'] = [
                    {
                        'metric': name,
                        'z_score': float(score),
                        'train_mean': float(train_mean),
                        'sample_value': float(sample_val)
                    }
                    for name, score, train_mean, sample_val in metric_differences
                ]
            
            analysis_result['metric_mode'] = metric_mode
        
        # Get top similar datasets
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        analysis_result['is_out_of_domain'] = bool(is_ood)
        analysis_result['best_match'] = {
            'dataset': best_match_dataset,
            'similarity': float(best_match[1]) if best_match else None
        }
        analysis_result['top_similar_datasets'] = [
            {'dataset': name, 'similarity': float(score)}
            for name, score in sorted_similarities[:10]
        ]
        analysis_result['all_similarities'] = {
            name: float(score) for name, score in similarities.items()
        }
        
        return analysis_result
    
    def analyze_audio_files(self,
                           audio_files: List[str],
                           detection_mode: str = "feature",
                           metric_mode: str = "numerical_only",
                           similarity_metric: str = "cosine",
                           ood_threshold: float = 0.5) -> List[Dict]:
        """
        Analyze multiple audio files.
        
        Args:
            audio_files: List of paths to audio files
            detection_mode: Detection mode
            metric_mode: Metric mode
            similarity_metric: Similarity metric
            ood_threshold: OOD threshold
        
        Returns:
            List of analysis results
        """
        results = []
        
        for audio_file in tqdm(audio_files, desc="Analyzing audio files"):
            result = self.analyze_audio_file(
                audio_file,
                detection_mode=detection_mode,
                metric_mode=metric_mode,
                similarity_metric=similarity_metric,
                ood_threshold=ood_threshold
            )
            results.append(result)
        
        return results
    
    def generate_summary(self, results: List[Dict]) -> Optional[str]:
        """
        Generate LLM-powered summary of analysis results.
        
        Args:
            results: List of analysis results
        
        Returns:
            Summary text or None if LLM is not available
        """
        if self.llm_client is None:
            return None
        
        # Prepare summary data
        summary_data = {
            'total_files': len(results),
            'ood_count': sum(1 for r in results if r.get('is_out_of_domain', False)),
            'in_domain_count': sum(1 for r in results if not r.get('is_out_of_domain', False)),
            'best_matches': defaultdict(int),
            'top_datasets': defaultdict(int)
        }
        
        for result in results:
            if 'error' in result:
                continue
            best_match = result.get('best_match', {})
            if best_match and best_match.get('dataset'):
                summary_data['best_matches'][best_match['dataset']] += 1
            
            top_similar = result.get('top_similar_datasets', [])
            for match in top_similar[:3]:  # Top 3
                summary_data['top_datasets'][match['dataset']] += match['similarity']
        
        # Format prompt
        prompt = f"""Analyze the following audio file analysis results and provide a comprehensive summary:

Total audio files analyzed: {summary_data['total_files']}
Out-of-domain files: {summary_data['ood_count']}
In-domain files: {summary_data['in_domain_count']}

Most common best matches:
{chr(10).join(f"  - {dataset}: {count} files" for dataset, count in sorted(summary_data['best_matches'].items(), key=lambda x: x[1], reverse=True)[:10])}

Top similar training datasets (aggregated similarity):
{chr(10).join(f"  - {dataset}: {score:.3f}" for dataset, score in sorted(summary_data['top_datasets'].items(), key=lambda x: x[1], reverse=True)[:10])}

Please provide:
1. A high-level summary of the audio characteristics
2. Insights about which training datasets these audios are most similar to
3. Observations about out-of-domain detection
4. Any notable patterns or anomalies
5. Recommendations for further analysis

Format the response in a clear, structured way."""

        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # or "gpt-4" for better quality
                    messages=[
                        {"role": "system", "content": "You are an expert audio analysis assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",  # or "claude-3-opus-20240229" for better quality
                    max_tokens=1000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
            elif self.llm_provider == "qwen":
                # Qwen local inference
                model = self.llm_client['model']
                tokenizer = self.llm_client['tokenizer']
                device = self.llm_client['device']
                
                # Format prompt for Qwen
                messages = [
                    {"role": "system", "content": "You are an expert audio analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
                
                # Tokenize
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=1000,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()
        
        except Exception as e:
            print(f"Error generating LLM summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Clean up temporary files."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Find all audio files in a directory."""
    if extensions is None:
        extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
    
    directory = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.glob(f"**/*{ext}"))
        audio_files.extend(directory.glob(f"**/*{ext.upper()}"))
    
    return [str(f) for f in audio_files]


def print_analysis_summary(results: List[Dict]):
    """Print a formatted summary of analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    total = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    ood_count = sum(1 for r in results if r.get('is_out_of_domain', False))
    in_domain_count = successful - ood_count
    
    print(f"\nTotal files: {total}")
    print(f"Successfully analyzed: {successful}")
    print(f"Out-of-domain: {ood_count} ({ood_count/successful*100:.1f}%)" if successful > 0 else "N/A")
    print(f"In-domain: {in_domain_count} ({in_domain_count/successful*100:.1f}%)" if successful > 0 else "N/A")
    
    # Most common best matches
    best_matches = defaultdict(int)
    for r in results:
        if 'error' not in r and r.get('best_match') and r['best_match'].get('dataset'):
            best_matches[r['best_match']['dataset']] += 1
    
    if best_matches:
        print(f"\nMost common best matches:")
        for dataset, count in sorted(best_matches.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dataset}: {count} files")
    
    # Per-file details
    print(f"\n" + "-"*80)
    print("PER-FILE RESULTS")
    print("-"*80)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.get('audio_name', 'Unknown')}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            ood_status = "OUT-OF-DOMAIN" if result.get('is_out_of_domain') else "IN-DOMAIN"
            print(f"  Status: {ood_status}")
            best_match = result.get('best_match', {})
            if best_match and best_match.get('dataset'):
                print(f"  Best match: {best_match['dataset']} (similarity: {best_match['similarity']:.3f})")
            
            top_similar = result.get('top_similar_datasets', [])[:5]
            if top_similar:
                print(f"  Top 5 similar datasets:")
                for match in top_similar:
                    print(f"    - {match['dataset']}: {match['similarity']:.3f}")


def save_results(results: List[Dict], output_path: str, summary: Optional[str] = None):
    """Save analysis results to JSON file."""
    output = {
        'results': results,
        'summary': summary,
        'metadata': {
            'total_files': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'ood_count': sum(1 for r in results if r.get('is_out_of_domain', False))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def interactive_mode(analyzer: AudioAnalyzer):
    """Run in interactive mode."""
    print("\n" + "="*80)
    print("INTERACTIVE AUDIO ANALYZER")
    print("="*80)
    print("\nEnter audio file paths (one per line, empty line to finish):")
    
    audio_files = []
    while True:
        line = input().strip()
        if not line:
            break
        if Path(line).exists():
            audio_files.append(line)
            print(f"  Added: {line}")
        else:
            print(f"  Warning: File not found: {line}")
    
    if not audio_files:
        print("No audio files provided. Exiting.")
        return
    
    print(f"\nAnalyzing {len(audio_files)} audio files...")
    
    # Configuration
    detection_mode = input("\nDetection mode [feature/metric/raw_features] (default: feature): ").strip() or "feature"
    if detection_mode not in ['feature', 'metric', 'raw_features']:
        detection_mode = "feature"
    
    metric_mode = "numerical_only"
    if detection_mode == 'metric':
        metric_mode = input("Metric mode [numerical_only/with_classification] (default: numerical_only): ").strip() or "numerical_only"
    
    similarity_metric = input("Similarity metric [cosine/euclidean] (default: cosine): ").strip() or "cosine"
    if similarity_metric not in ['cosine', 'euclidean']:
        similarity_metric = "cosine"
    
    try:
        ood_threshold = float(input("OOD threshold [0.0-1.0] (default: 0.5): ").strip() or "0.5")
    except ValueError:
        ood_threshold = 0.5
    
    # Analyze
    results = analyzer.analyze_audio_files(
        audio_files,
        detection_mode=detection_mode,
        metric_mode=metric_mode,
        similarity_metric=similarity_metric,
        ood_threshold=ood_threshold
    )
    
    # Print summary
    print_analysis_summary(results)
    
    # Generate LLM summary if available
    if analyzer.llm_client:
        use_llm = input("\nGenerate LLM summary? [y/N]: ").strip().lower() == 'y'
        if use_llm:
            print("\nGenerating LLM summary...")
            summary = analyzer.generate_summary(results)
            if summary:
                print("\n" + "="*80)
                print("LLM SUMMARY")
                print("="*80)
                print(summary)
    
    # Save results
    save_output = input("\nSave results to file? [y/N]: ").strip().lower() == 'y'
    if save_output:
        output_path = input("Output path (default: analysis_results.json): ").strip() or "analysis_results.json"
        summary_text = summary if 'summary' in locals() and summary else None
        save_results(results, output_path, summary=summary_text)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Audio Analyzer - Extract features and analyze audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific audio files
  python interactive_audio_analyzer.py --audio_files file1.wav file2.wav file3.wav
  
  # Analyze all audio files in a directory
  python interactive_audio_analyzer.py --audio_dir /path/to/audio/files
  
  # Interactive mode
  python interactive_audio_analyzer.py --interactive
  
  # With custom detection settings
  python interactive_audio_analyzer.py --audio_files *.wav --detection_mode metric --metric_mode with_classification
  
  # With Qwen LLM summarization (open-source, no API key needed)
  python interactive_audio_analyzer.py --audio_files *.wav --generate_summary
  
  # With specific Qwen model
  python interactive_audio_analyzer.py --audio_files *.wav --generate_summary --llm_model "Qwen/Qwen2.5-1.5B-Instruct"
        """
    )
    
    parser.add_argument("--audio_files", type=str, nargs='+',
                       help="List of audio file paths to analyze")
    parser.add_argument("--audio_dir", type=str,
                       help="Directory containing audio files to analyze")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--database", type=str, default="database.pkl",
                       help="Path to database pickle file (default: database.pkl)")
    parser.add_argument("--model", type=str, default="espnet/arecho_base_v0",
                       help="Model name (default: espnet/arecho_base_v0)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use (default: cuda)")
    parser.add_argument("--detection_mode", type=str,
                       choices=["feature", "metric", "raw_features"],
                       default="feature",
                       help="Detection mode (default: feature)")
    parser.add_argument("--metric_mode", type=str,
                       choices=["numerical_only", "with_classification",
                               "numerical_only_normalized", "with_classification_normalized"],
                       default="numerical_only",
                       help="Metric mode (only if detection_mode=metric)")
    parser.add_argument("--similarity_metric", type=str,
                       choices=["cosine", "euclidean"],
                       default="cosine",
                       help="Similarity metric (default: cosine)")
    parser.add_argument("--ood_threshold", type=float, default=0.5,
                       help="Out-of-domain threshold (default: 0.5)")
    parser.add_argument("--output", type=str, default="analysis_results.json",
                       help="Output JSON file path (default: analysis_results.json)")
    parser.add_argument("--generate_summary", action="store_true",
                       help="Generate LLM summary (requires API key or Qwen)")
    parser.add_argument("--llm_model", type=str, default=None,
                       help="Qwen model name (e.g., 'Qwen/Qwen2.5-0.5B-Instruct'). Default: Qwen2.5-0.5B-Instruct")
    parser.add_argument("--use_qwen", action="store_true", default=True,
                       help="Use Qwen as default LLM if no API keys found (default: True)")
    parser.add_argument("--no_qwen", action="store_false", dest="use_qwen",
                       help="Disable Qwen LLM (use only API-based LLMs)")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Determine audio files
    audio_files = []
    if args.interactive:
        # Will be handled in interactive_mode
        pass
    elif args.audio_files:
        audio_files = args.audio_files
    elif args.audio_dir:
        audio_files = find_audio_files(args.audio_dir)
        if not audio_files:
            print(f"No audio files found in {args.audio_dir}")
            return
    else:
        parser.print_help()
        print("\nError: Must provide --audio_files, --audio_dir, or --interactive")
        return
    
    # Limit files if specified
    if args.max_files and len(audio_files) > args.max_files:
        print(f"Limiting to {args.max_files} files (found {len(audio_files)})")
        audio_files = audio_files[:args.max_files]
    
    # Initialize analyzer
    try:
        with AudioAnalyzer(
            database_path=args.database,
            model_name=args.model,
            device=args.device,
            llm_model=args.llm_model,
            use_qwen=args.use_qwen
        ) as analyzer:
            
            if args.interactive:
                interactive_mode(analyzer)
            else:
                print(f"\nAnalyzing {len(audio_files)} audio files...")
                
                results = analyzer.analyze_audio_files(
                    audio_files,
                    detection_mode=args.detection_mode,
                    metric_mode=args.metric_mode,
                    similarity_metric=args.similarity_metric,
                    ood_threshold=args.ood_threshold
                )
                
                # Print summary
                print_analysis_summary(results)
                
                # Generate LLM summary if requested
                summary = None
                if args.generate_summary and analyzer.llm_client:
                    print("\nGenerating LLM summary...")
                    summary = analyzer.generate_summary(results)
                    if summary:
                        print("\n" + "="*80)
                        print("LLM SUMMARY")
                        print("="*80)
                        print(summary)
                
                # Save results
                save_results(results, args.output, summary=summary)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

