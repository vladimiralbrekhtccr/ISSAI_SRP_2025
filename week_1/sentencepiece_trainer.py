# https://github.com/google/sentencepiece/blob/master/doc/options.md
import json
import sentencepiece as spm

# Настройки
INPUT_JSONS = [
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/train_kk_various_cleaned_v2_splited.json",
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/train_kk_books_cleaned_v2_splited.json",
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/txt_tq_kaz.json",
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/HuggingFaceFW_fineweb-edu_CC-MAIN-2024-51_train_6-kk.json",
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/pdf_tq_kaz.json",
    "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/doc_tq_kaz.json",
]
MODEL_PREFIX = "multilang_sp"

print("📥 Загружаем тексты в память...")
all_texts = []

for file_path in INPUT_JSONS:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                text = data.get("text", "").strip()
                if text:
                    all_texts.append(text)
            except:
                continue

print(f"✅ Загружено {len(all_texts)} предложений")


# INPUT_JSON = "/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/data/db_11_v2_splited_kazakh/train_kk_various_cleaned_v2_splited.json"
# TEXT_FILE = "kaz_texts.txt"
# MODEL_PREFIX = "kaz_sp"


# # 1. Извлекаем тексты из JSON
# print("Извлекаем тексты...")
# with open(INPUT_JSON, 'r', encoding='utf-8') as f_in, \
#      open(TEXT_FILE, 'w', encoding='utf-8') as f_out:
#     for line in f_in:
#         try:
#             data = json.loads(line.strip())
#             text = data.get('text', '')
#             if text:
#                 f_out.write(text + '\n')
#         except:
#             continue

# === Обучение SentencePiece напрямую из памяти ===
print("🧠 Обучаем SentencePiece из памяти...")
spm.SentencePieceTrainer.train(
    sentence_iterator=iter(all_texts),
    #input=TEXT_FILE,
    model_prefix=MODEL_PREFIX,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    bos_piece='<bos>',  # Вместо <s>
    eos_piece='<eos>',  # Вместо </s> sentencepiece_trainer
    unk_id=3,
    user_defined_symbols=['<mask>', '<start_of_turn>','<end_of_turn>','<sep>', '<cls>'],
    vocab_size=15000,
    model_type='bpe',
    character_coverage=1,
    num_threads=16,
    input_sentence_size=100000,
    shuffle_input_sentence=True,
    max_sentencepiece_length=16,
    byte_fallback=True,
    use_all_vocab=False,
    remove_extra_whitespaces=False,
    normalization_rule_name='identity',
    split_digits=False,
    split_by_number=False,
    split_by_whitespace=True,
    split_by_unicode_script=False,
    treat_whitespace_as_suffix=False,
    allow_whitespace_only_pieces=False,
    vocabulary_output_piece_score=True,
    train_extremely_large_corpus=True,
    random_seed=42
)

# 3. Тестируем
print("\nТестируем токенайзер...")
sp = spm.SentencePieceProcessor(model_file=f'{MODEL_PREFIX}.model')

test_text = "Қазақстан Республикасының Президенті"
tokens = sp.encode_as_pieces(test_text)
ids = sp.encode_as_ids(test_text)
decoded = sp.decode_pieces(tokens)

print(f"Текст: {test_text}")
print(f"Токены: {tokens}")
print(f"IDs: {ids}")
print(f"Декодированный: {decoded}")
print(f"Размер словаря: {sp.get_piece_size()}")

print(f"\nГотово! Модель сохранена в {MODEL_PREFIX}.model")



# To save .json and so on but first add the tokenizer_config.json to this folder (example tokenizer_config.json)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from transformers import AutoTokenizer

# text_data = "Жаксым"

# tokenizer = AutoTokenizer.from_pretrained("/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/bpe_spm")
# #tokenizer.save_pretrained("/raid/vladimir_albrekht/projects/ISSAI_vladimir_projects/smollm/bpe_spm/multilang_sp")
# encoded_text = tokenizer.encode(text_data)
# print(encoded_text)

# decoded_text = tokenizer.decode(encoded_text)
# print(decoded_text)
