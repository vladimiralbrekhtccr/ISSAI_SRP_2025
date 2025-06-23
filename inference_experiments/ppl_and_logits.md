## For Beibars

Basically there is cool stuff that you can try to do with model.

Step by step.

### 1. Load model

```python 
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import os 
os.environ["CUDA_VISIBLE_DEVICE"] = "0,1"
# Загрузка твоей обученной модели
model_path = "SRP-base-model-training/gemma_3_2B_base_v1_kk_only_5B-data"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Gemma3ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### 2. Simple inference

```python
prompt = ("Сәлем")

model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(
        **model_inputs, 
        max_new_tokens=50, 
        do_sample=True,
        #temperature=0.2,
        top_p=0.9,                # nucleus sampling
        #top_k=50,                 # отфильтровать всё, кроме 50 лучших токенов
        repetition_penalty=1.2, 
        pad_token_id=tokenizer.eos_token_id
    )
    generation = generation[0][input_len:]

decoded = tokenizer.decode(generation, skip_special_tokens=True)
print(decoded)
```

### 3. Next token predicton to actually see what model think is the next.

```python
tok = tokenizer

context = "Астан"                  # любое слово / фраза
inputs  = tok(context, return_tensors="pt").to(model.device)

with torch.no_grad():
    out   = model(**inputs)        # logits shape: (1, seq_len, vocab)
    logits_next = out.logits[0, -1]  # вектор на последний токен

# Top-k самых вероятных токенов
k = 50
probs = torch.softmax(logits_next, dim=-1)
topk  = torch.topk(probs, k)

for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
    raw_tok = tokenizer.convert_ids_to_tokens(idx)
    print(f"{repr(raw_tok):<12}  {p:.3f}")

```

### 4. Perplexity score

```python
from torch.nn import functional as F

text = "Кітап – білімнің кілті."
enc  = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out   = model(**enc)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = enc.input_ids[..., 1:].contiguous()
    loss  = F.cross_entropy(
              shift_logits.view(-1, shift_logits.size(-1)),
              shift_labels.view(-1),
              reduction="mean"
            )
    ppl = torch.exp(loss)
print(f"loss={loss.item():.4f}  |  perplexity={ppl.item():.2f}")

```

### 5. Model architecture 

```python
model
```

### 6. Trainable params

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```



### Some cool stuff todo

* Try to play with inference params to find the optimal for this specific model.

```python
    max_new_tokens=50, 
    do_sample=True,
    #temperature=0.2,
    top_p=0.9,                # nucleus sampling
    #top_k=50,                 # отфильтровать всё, кроме 50 лучших токенов
    repetition_penalty=1.2, 
    pad_token_id=tokenizer.eos_token_id
```
* Mask heads of the model and check the inference.
