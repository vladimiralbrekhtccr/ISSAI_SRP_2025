import os
import sentencepiece as spm
import json
import pandas as pd
from typing import Generator
import random

input_files_json = [
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/en_data/train_en_fineweb_cleaned_v2_splited.json",
    "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/ru_data/russian/russian/train_ru_wikipedia_cleaned_v2.json",
]

input_files_parquet = [
    "/home/srp/data/kazakh/snapshots/d62ba981d9ba825905753c27ee73ed5814ebb9ed/data/kk_wikipedia_cleaned_v2_split-00000-of-00001.parquet",
]

output_dir = "./spm_bpe_tokenizer_200m"
model_prefix = "tokenizer_multilingual"

# Optimized settings for 200M-1B model scaling
vocab_size = 75000  # Sweet spot for 200M-1B models
target_samples = 1000000  # 1M samples for better coverage
samples_per_file = 333000

def sample_texts() -> Generator[str, None, None]:
    """Sample texts efficiently from all files."""

    print("Running text sampling...")

    # Sample from JSON files
    for file_path in input_files_json:
        print(f"Sampling {samples_per_file:,} from {os.path.basename(file_path)}")
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "").strip()
                        if text and len(text) > 50:  # Filter very short texts
                            lines.append(text)
                    except json.JSONDecodeError:
                        continue
                
                # Random sample
                if len(lines) > samples_per_file:
                    lines = random.sample(lines, samples_per_file)
                
                for text in lines:
                    yield text
                    count += 1
                
                print(f"  Sampled {count:,} texts")
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
    
    # Sample from Parquet files
    for file_path in input_files_parquet:
        print(f"Sampling {samples_per_file:,} from {os.path.basename(file_path)}")
        try:
            df = pd.read_parquet(file_path)
            texts = df["text"].astype(str).str.strip().dropna()
            texts = texts[texts.str.len() > 50]  # Filter short texts
            
            if len(texts) > samples_per_file:
                texts = texts.sample(n=samples_per_file, random_state=42)
            
            count = 0
            for text in texts:
                if text and text != 'nan':
                    yield text
                    count += 1
            
            print(f"  Sampled {count:,} texts")
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")

def train_tokenizer():
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, model_prefix)
    
    print("Training tokenizer...")
    
    spm.SentencePieceTrainer.train(
        sentence_iterator=sample_texts(),
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,  
        model_type="bpe",
        pad_id=0, eos_id=1, bos_id=2, unk_id=3,
        bos_piece='<bos>', eos_piece='<eos>',
        user_defined_symbols=['<mask>', '<start_of_turn>', '<end_of_turn>'],
        num_threads=32,  
        input_sentence_size=750000,  # 1M for better scaling
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,  
        split_by_whitespace=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        byte_fallback=True,
        max_sentence_length=4192,
    )
    
    # Test and save
    sp = spm.SentencePieceProcessor()
    sp.load(f"{prefix}.model")
    print(f"Test: {sp.encode_as_pieces('Hello мир қазақ')}")
    
    # Save config
    config = {
        "tokenizer_class": "SentencePieceTokenizerFast",
        "vocab_size": vocab_size,
        "bos_token": "<bos>", "eos_token": "<eos>",
        "pad_token": "<pad>", "unk_token": "<unk>",
        "model_max_length": 2048,
    }
    
    import json, shutil
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    shutil.copy(f"{prefix}.model", os.path.join(output_dir, "tokenizer.model"))
    shutil.copy(f"{prefix}.vocab", os.path.join(output_dir, "tokenizer.vocab"))
    
    print(f"✅ Tokenizer saved to {output_dir}")

if __name__ == "__main__":
    random.seed(42)  # Reproducible sampling
    train_tokenizer()