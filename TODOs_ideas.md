### 1

1. Train 100 M model with base training.

2. Train 100 M model with SFT trianing.

But dataset for both is the same and we only apply masking for the second one.

### 2 

1. Train 100M -> 200M -> 300M -> 400 etc. with same dataset and SFT only approach on base model and check the performance on the benches.

2. Train 100M -> 200M -> ... on [KazSparc](https://huggingface.co/datasets/issai/kazparc) for translation task.
