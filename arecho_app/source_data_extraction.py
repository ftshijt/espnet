import numpy as np
import torch
import random
from pathlib import Path
from espnet2.bin.universa_inference import UniversaInference
from tqdm import tqdm

# Import common functions from data_extraction
from data_extraction import extract_features, save_features


def process_source_data(source_data_dir, universa_inference, output_dir, device="cuda", num_samples_per_dataset=None, random_seed=42):
    """Process audio files from source_data directory where each subfolder is a dataset."""
    source_data_path = Path(source_data_dir)
    
    if not source_data_path.exists():
        print(f"Source data directory not found: {source_data_dir}")
        return 0, 0
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\nProcessing source_data from: {source_data_dir}")
    
    # Find all subdirectories (datasets)
    dataset_dirs = [d for d in source_data_path.iterdir() if d.is_dir()]
    print(f"Found {len(dataset_dirs)} datasets in source_data")
    
    successful = 0
    failed = 0
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Find all audio files (support .wav, .flac, .mp3, etc.)
        audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        audio_files = [f for f in dataset_dir.rglob('*') 
                      if f.is_file() and f.suffix.lower() in audio_extensions]
        
        if not audio_files:
            print(f"  No audio files found in {dataset_name}")
            continue
        
        print(f"  Found {len(audio_files)} audio files")
        
        # Sample if specified
        if num_samples_per_dataset and len(audio_files) > num_samples_per_dataset:
            audio_files = random.sample(audio_files, num_samples_per_dataset)
            print(f"  Sampled {len(audio_files)} files")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"  Processing {dataset_name}"):
            # Use filename (without extension) as utterance_id
            utterance_id = audio_file.stem
            
            result = extract_features(universa_inference, str(audio_file), device=device)
            
            if result is not None:
                save_features(result, output_dir, utterance_id, dataset_name=dataset_name)
                successful += 1
            else:
                failed += 1
                print(f"  Failed to extract features for {utterance_id}")
    
    return successful, failed


def main():
    # Configuration
    source_data_dir = "source_data"
    output_dir = "feature"
    model_name = "espnet/arecho_base_v0"
    device = "cuda"
    num_samples_per_dataset = None  # Set to None to process all files, or a number to sample
    random_seed = 42
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Initialize model
    print(f"Loading model: {model_name}")
    universa_inference = UniversaInference.from_pretrained(model_name, device=device)
    print("Model loaded successfully")
    
    # Process source_data
    successful, failed = process_source_data(
        source_data_dir, 
        universa_inference, 
        output_dir, 
        device=device,
        num_samples_per_dataset=num_samples_per_dataset,
        random_seed=random_seed
    )
    
    print(f"\nFeature extraction from source_data completed!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

