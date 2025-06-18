import sentencepiece as spm
import json
import statistics
import re
from pathlib import Path
import random
import pandas as pd

# Load your trained tokenizer
tokenizer_path = "/home/sanzhar/llm_project/llm_tokenizer/spm_bpe_tokenizer_200m/tokenizer.model"
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

# Test texts in different languages
def collect_texts_from_json(data_dir, target_size_mb=10, language_name="Russian"):
    target_size_bytes = target_size_mb * 1024 * 1024
    collected_texts = []
    current_size = 0
    output_file = f"{language_name.lower()}_eval_texts.json"
    
    # Get all JSON files
    json_files = list(Path(data_dir).glob("*.json"))
    random.shuffle(json_files)  # Randomize file order
    
    print(f"Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        if current_size >= target_size_bytes:
            break
            
        print(f"Processing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if current_size >= target_size_bytes:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        # Extract text field (adjust key name if different)
                        text = data.get('text', '') or data.get('content', '') or str(data)
                        
                        if text and len(text.strip()) > 50:  # Skip very short texts
                            collected_texts.append(text.strip())
                            current_size += len(text.encode('utf-8'))
                            
                            if len(collected_texts) % 1000 == 0:
                                print(f"  Collected {len(collected_texts)} texts, {current_size/1024/1024:.1f}MB")
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create structured output
    result = {language_name: collected_texts}
    
    # Save as JSON
    json_output = output_file.replace('.txt', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    final_size_mb = current_size / 1024 / 1024
    print(f"\nCollected {len(collected_texts)} texts ({final_size_mb:.1f}MB)")
    print(f"Saved to: {json_output}")
    
    return result

def collect_texts_from_parquet(data_dir, target_size_mb=10, text_column='text', language_name="Russian"):
    target_size_bytes = target_size_mb * 1024 * 1024
    collected_texts = []
    current_size = 0
    output_file = f"{language_name.lower()}_eval_texts.json"
    
    # Get all Parquet files
    parquet_files = list(Path(data_dir).glob("*.parquet"))
    random.shuffle(parquet_files)  # Randomize file order
    
    print(f"Found {len(parquet_files)} Parquet files")
    
    for file_path in parquet_files:
        if current_size >= target_size_bytes:
            break
            
        print(f"Processing: {file_path.name}")
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path, engine="pyarrow").head(10000)
            
            # Check available columns
            if text_column not in df.columns:
                # Try common text column names
                possible_columns = ['text', 'content', 'body', 'message', 'article']
                text_column_found = None
                
                for col in possible_columns:
                    if col in df.columns:
                        text_column_found = col
                        break
                
                if text_column_found:
                    print(f"  Using column: {text_column_found}")
                    text_column = text_column_found
                else:
                    print(f"  No text column found. Available columns: {list(df.columns)}")
                    continue
            
            # Sample rows randomly
            df_sample = df.sample(frac=1).reset_index(drop=True)
            
            for idx, row in df_sample.iterrows():
                if current_size >= target_size_bytes:
                    break
                
                text = str(row[text_column])
                
                if text and text != 'nan' and len(text.strip()) > 50:  # Skip very short texts
                    collected_texts.append(text.strip())
                    current_size += len(text.encode('utf-8'))
                    
                    if len(collected_texts) % 1000 == 0:
                        print(f"  Collected {len(collected_texts)} texts, {current_size/1024/1024:.1f}MB")
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create structured output
    result = {language_name: collected_texts}
    
    # Save as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    final_size_mb = current_size / 1024 / 1024
    print(f"\nCollected {len(collected_texts)} texts ({final_size_mb:.1f}MB)")
    print(f"Saved to: {output_file}")
    
    return result


data_directories_json = {
    "Russian" : "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/ru_data/russian/russian",
    "English" : "/home/srp/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/en_data",
}

data_directories_parquet = {
    "Kazakh" : "/home/srp/data/kazakh/snapshots/d62ba981d9ba825905753c27ee73ed5814ebb9ed/data",
}

test_texts = {}

for language in data_directories_json:
    text = collect_texts_from_json(data_directories_json[language], target_size_mb=25, language_name=language)
    test_texts.update(text)

for language in data_directories_parquet:
    text = collect_texts_from_parquet(data_directories_parquet[language], target_size_mb=25, language_name=language)
    test_texts.update(text)

def count_words(text):
    """Count words using whitespace splitting."""
    return len(text.split())

def evaluate_tokenizer():
    """Evaluate tokenizer performance on multiple metrics."""
    
    print("=" * 60)
    print("TOKENIZER PERFORMANCE EVALUATION")
    print("=" * 60)
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Vocabulary size: {sp.vocab_size()}")
    print()
    
    all_metrics = {
        "compression_ratios": [],
        "fertility_scores": [],
        "continued_word_ratios": []
    }
    
    for lang, texts in test_texts.items():
        print(f"\n📊 {lang.upper()} EVALUATION")
        print("-" * 40)
        
        lang_compression = []
        lang_fertility = []
        lang_continued = []
        
        for i, text in enumerate(texts, 1):
            # Tokenize
            tokens = sp.encode_as_pieces(text)
            
            # 1. Character-Level Compression Ratio
            total_chars = len(text)
            total_tokens = len(tokens)
            compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            
            # 2. Word-Level Fertility
            total_words = count_words(text)
            fertility = total_tokens / total_words if total_words > 0 else 0
            
            # 3. Subword Segmentation (Continued Words)
            # Count words that are split into multiple tokens
            words = text.split()
            continued_words = 0
            
            for word in words:
                word_tokens = sp.encode_as_pieces(word.strip('.,!?";:()'))
                if len(word_tokens) > 1:
                    continued_words += 1
            
            continued_ratio = continued_words / total_words if total_words > 0 else 0
            
            # Store metrics
            lang_compression.append(compression_ratio)
            lang_fertility.append(fertility)
            lang_continued.append(continued_ratio)
            
            # Print detailed results for first text
            if i == 1:
                print(f"Sample text: \"{text[:60]}...\"")
                print(f"Tokens: {tokens[:8]}...")
                print()
            
            # if i % 100 == 0:
            #     print(f"Text {i}:")
            #     print(f"  Chars: {total_chars:4d} | Tokens: {total_tokens:3d} | Words: {total_words:3d}")
            #     print(f"  Compression: {compression_ratio:.2f} | Fertility: {fertility:.2f} | Continued: {continued_ratio:.1%}")
        
        # Language averages
        avg_compression = statistics.mean(lang_compression)
        avg_fertility = statistics.mean(lang_fertility)
        avg_continued = statistics.mean(lang_continued)
        
        print(f"\n{lang} Averages:")
        print(f"  📏 Compression Ratio: {avg_compression:.2f}")
        print(f"  🌱 Fertility Score:   {avg_fertility:.2f}")
        print(f"  ✂️  Continued Words:   {avg_continued:.1%}")
        
        # Add to overall metrics
        all_metrics["compression_ratios"].extend(lang_compression)
        all_metrics["fertility_scores"].extend(lang_fertility)
        all_metrics["continued_word_ratios"].extend(lang_continued)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    overall_compression = statistics.mean(all_metrics["compression_ratios"])
    overall_fertility = statistics.mean(all_metrics["fertility_scores"])
    overall_continued = statistics.mean(all_metrics["continued_word_ratios"])
    
    print(f"📏 Average Compression Ratio: {overall_compression:.2f}")
    print(f"   (Higher = better compression, ~3-4 is good)")
    print()
    print(f"🌱 Average Fertility Score:   {overall_fertility:.2f}")  
    print(f"   (Lower = better, ~1.3-1.8 is good)")
    print()
    print(f"✂️  Average Continued Words:   {overall_continued:.1%}")
    print(f"   (Moderate = good, ~30-60% is typical)")
    
    # Performance interpretation
    print("\n" + "=" * 60)
    print("PERFORMANCE INTERPRETATION")
    print("=" * 60)
    
    # Compression ratio assessment
    if overall_compression >= 3.5:
        comp_grade = "Excellent"
    elif overall_compression >= 2.8:
        comp_grade = "Good"
    elif overall_compression >= 2.0:
        comp_grade = "Fair"
    else:
        comp_grade = "Poor"
    
    # Fertility assessment  
    if overall_fertility <= 1.4:
        fert_grade = "Excellent"
    elif overall_fertility <= 1.8:
        fert_grade = "Good"
    elif overall_fertility <= 2.2:
        fert_grade = "Fair"
    else:
        fert_grade = "Poor"
    
    # Continued words assessment
    if 0.25 <= overall_continued <= 0.65:
        cont_grade = "Good balance"
    elif overall_continued < 0.25:
        cont_grade = "Too little segmentation"
    else:
        cont_grade = "Too much segmentation"
    
    print(f"Compression Efficiency: {comp_grade}")
    print(f"Word-Level Efficiency:  {fert_grade}")
    print(f"Segmentation Balance:   {cont_grade}")
    
    return {
        "compression_ratio": overall_compression,
        "fertility_score": overall_fertility,
        "continued_word_ratio": overall_continued
    }

def test_specific_examples():
    """Test tokenizer on specific challenging examples."""
    print("\n" + "=" * 60)
    print("CHALLENGING EXAMPLES TEST")
    print("=" * 60)
    
    examples = [
        "preprocessing",
        "антидискриминационный", 
        "ақпараттандыру",
        "COVID-19",
        "machine-learning",
        "сверхъестественный",
        "Hello, world!",
        "100,000.50",
        "email@domain.com"
    ]
    
    print("Word → Tokens → Reconstruction")
    print("-" * 45)
    
    for word in examples:
        tokens = sp.encode_as_pieces(word)
        reconstructed = "".join(tokens).replace("▁", " ").strip()
        print(f"{word:20} → {tokens}")
        print(f"{'':20} → \"{reconstructed}\"")
        print()

if __name__ == "__main__":
    try:
        # Main evaluation
        metrics = evaluate_tokenizer()
        
        # Specific examples
        test_specific_examples()
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your tokenizer model exists at:", tokenizer_path)