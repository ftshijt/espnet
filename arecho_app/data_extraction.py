import numpy as np
import torch
import librosa
import soundfile
import json
import os
import random
from collections import defaultdict
from pathlib import Path
import kaldiio
from espnet2.bin.universa_inference import UniversaInference
from tqdm import tqdm


def parse_wav_scp(wav_scp_path):
    """Parse wav.scp file and group utterances by dataset."""
    dataset_utterances = defaultdict(list)
    
    print(f"Reading wav.scp file: {wav_scp_path}")
    with open(wav_scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split utterance_id and audio_path
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            
            utterance_id = parts[0]
            audio_path = parts[1]
            
            # Replace /scratch with /work/hdd for disk space change
            audio_path = audio_path.replace('/scratch', '/work/hdd')
            
            # Extract dataset name (first field when split by "_")
            dataset_name = utterance_id.split('_')[0]
            
            # Special handling for OpenSLR: extract specific sub-dataset name
            if dataset_name == "OpenSLR":
                # Pattern: OpenSLR_...-{sub_dataset}_openslr_...
                # Extract the part between the last '-' and '_openslr'
                if '-' in utterance_id and '_openslr' in utterance_id:
                    # Find the last dash and the _openslr position
                    last_dash_idx = utterance_id.rfind('-')
                    openslr_idx = utterance_id.find('_openslr', last_dash_idx)
                    if last_dash_idx != -1 and openslr_idx != -1:
                        sub_dataset = utterance_id[last_dash_idx + 1:openslr_idx]
                        dataset_name = f"OpenSLR_{sub_dataset}"
            
            dataset_utterances[dataset_name].append({
                'utterance_id': utterance_id,
                'audio_path': audio_path
            })
    
    print(f"Found {len(dataset_utterances)} datasets")
    for dataset, utterances in dataset_utterances.items():
        print(f"  {dataset}: {len(utterances)} utterances")
    
    return dataset_utterances


def read_audio_from_ark(audio_path):
    """Read audio from Kaldi ark file format (path:offset) or wav file."""
    if ':' in audio_path and '.ark' in audio_path:
        # Kaldi ark format with offset
        # The audio_path is already in format: /path/to/file.ark:offset
        # kaldiio.load_mat returns (sample_rate, audio_array) tuple
        try:
            # Pass the path directly (without 'ark:' prefix) to kaldiio.load_mat
            result = kaldiio.load_mat(audio_path)
            
            # kaldiio.load_mat returns (sample_rate, audio_array) for ark files
            if isinstance(result, tuple) and len(result) == 2:
                sr, audio_array = result
            else:
                # Fallback: assume it's just the array
                audio_array = result
                sr = 16000  # Default sample rate
            
            # Ensure audio_array is 1D
            if len(audio_array.shape) > 1:
                # If 2D, might need to flatten or handle differently
                if audio_array.shape[0] == 1:
                    audio_array = audio_array[0]
                else:
                    # Flatten if needed
                    audio_array = audio_array.flatten()
            
            return audio_array, sr
        except Exception as e:
            raise ValueError(f"Could not read ark file {audio_path}: {e}")
    else:
        # Direct wav file path
        audio, sr = soundfile.read(audio_path)
        return audio, sr


def audio_preprocess(audio_array, sr=None):
    """Preprocess audio for UniversaInference."""
    if sr is None:
        # Assume 16kHz if not provided
        sr = 16000
    elif sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    
    audio = audio_array.astype(np.float32)
    audio = torch.from_numpy(audio).unsqueeze(0)
    audio_lengths = torch.tensor([len(audio[0])])
    return audio, audio_lengths


def extract_features(universa_inference, audio_path, device="cuda"):
    """Extract features from audio using UniversaInference."""
    # Read audio
    try:
        audio_array, sr = read_audio_from_ark(audio_path)
        audio, audio_lengths = audio_preprocess(audio_array, sr)
    except Exception as e:
        print(f"Error reading audio from {audio_path}: {e}")
        return None
    
    # Move to device
    audio = audio.to(device)
    audio_lengths = audio_lengths.to(device)
    
    # Placeholder reference audio
    ref_audio = torch.zeros(1, 8000, dtype=torch.float32).to(device)
    ref_audio_lengths = torch.tensor([8000]).to(device)
    
    # Inference
    try:
        with torch.no_grad():
            result = universa_inference(
                audio.float(), 
                audio_lengths, 
                ref_audio=ref_audio.float(), 
                ref_audio_lengths=ref_audio_lengths
            )
        return result
    except Exception as e:
        print(f"Error during inference for {audio_path}: {e}")
        return None


def save_features(result, output_dir, utterance_id, dataset_name=None):
    """Save extracted features to disk."""
    output_dir = Path(output_dir)
    
    # Create subfolder for dataset if specified
    if dataset_name:
        output_dir = output_dir / dataset_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate encoded_feats from other fields (don't modify original dict)
    encoded_feat = result.get('encoded_feat', None)
    
    # Save other fields as JSON (excluding encoded_feats)
    json_path = output_dir / f"{utterance_id}.json"
    # Convert tensors to numpy arrays for JSON serialization
    json_data = {}
    for key, value in result.items():
        if key == 'encoded_feat':
            continue  # Skip encoded_feats, save separately
        if isinstance(value, torch.Tensor):
            json_data[key] = value.cpu().numpy().tolist()
        elif isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        else:
            json_data[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save encoded_feat as npz
    if encoded_feat is not None:
        npz_path = output_dir / f"{utterance_id}_encoded_feat.npz"
        if isinstance(encoded_feat, torch.Tensor):
            encoded_feat = encoded_feat.cpu().numpy()
        np.savez_compressed(npz_path, encoded_feat=encoded_feat.squeeze(0))


def main():
    # Configuration
    wav_scp_path = "/work/hdd/bbjs/shared/owsm_data/owsm_data_v3/train/wav.scp"
    output_dir = "feature"
    model_name = "espnet/arecho_base_v0"
    device = "cuda"
    num_samples_per_dataset = 100
    random_seed = 42
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Initialize model
    print(f"Loading model: {model_name}")
    universa_inference = UniversaInference.from_pretrained(model_name, device=device)
    print("Model loaded successfully")
    
    # Parse wav.scp
    dataset_utterances = parse_wav_scp(wav_scp_path)
    
    # Sample utterances from each dataset
    sampled_utterances = []
    for dataset_name, utterances in dataset_utterances.items():
        if len(utterances) >= num_samples_per_dataset:
            sampled = random.sample(utterances, num_samples_per_dataset)
        else:
            sampled = utterances
            print(f"Warning: {dataset_name} has only {len(utterances)} utterances, using all")
        
        # Add dataset_name to each sampled item
        for item in sampled:
            item['dataset_name'] = dataset_name
        
        sampled_utterances.extend(sampled)
        print(f"Sampled {len(sampled)} utterances from {dataset_name}")
    
    print(f"\nTotal utterances to process: {len(sampled_utterances)}")
    
    # Extract features
    successful = 0
    failed = 0
    
    for item in tqdm(sampled_utterances, desc="Extracting features"):
        utterance_id = item['utterance_id']
        audio_path = item['audio_path']
        dataset_name = item['dataset_name']
        
        result = extract_features(universa_inference, audio_path, device=device)
        
        if result is not None:
            save_features(result, output_dir, utterance_id, dataset_name=dataset_name)
            successful += 1
        else:
            failed += 1
            print(f"Failed to extract features for {utterance_id}")
    
    print(f"\nFeature extraction completed!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

