# Project: *Training small LLM (200M-500M) from scratch on Kazakh English and Russian languages.*

## General rules:
- Each week will have a brief presentation of the work done and sharing knowledge with others.
- If someone finished the task early, he should help others. 
- If you have any questions, please ask me immediately in our <%might_be_changed%> telegram group. <%/might_be_changed%>

## Introduction to the project and the goal of it.
Project is about training small LLM (200M-500M) from scratch on Kazakh English and Russian languages using custom tokenizer.

Our team will consist of 3-6 students: (for now plan is for 3 students)
- Tokenizer/helper student.
- Base model training student.
- SFT (Supervised fine-tuning) student.

Why it's important:
- We don't have any small models trained on Kazakh and Russian languages.
- We can share this model with others on Hugging Face such that other will be able to further fine-tune it for their needs.

On which tasks we can fine-tune it later on:
- Chatbots
- Summarization
- Translation
- Question answering
- Add image understanding to the model
- Add audio understanding to the model
- etc. 

Why we are building it from scratch and not using existing models:
- We will use efficient tokenizer that will improve the generation speed of the model.
- But we may consider training existing models (qwen, llama, gemm) as well.

## Tokenizer/helper student.
- **[Week1]** Research week. 
    - Understand what is tokenizer and how to train it. 
      - Read paper and articles about tokenization, watch youtube videos about tokenization.
        - [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=uscBgajGLQJ8ToQE)
        - [Tokenizers library](https://huggingface.co/learn/llm-course/en/chapter6)
        - [Qwen tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B)
        - [Gemma tokenizer](https://huggingface.co/google/gemma-3-1b-pt)
- **[Week2]** Experiment with tokenizer from scratch and tokenizer expansion techniques.
    - Train tokenizer from scratch using BPE algorithm, for that you will need to:
        - Collect the data (kk, en, ru).
            - Create diverse proportions of dataset consisting of different domains for 3 languages (kk, en, ru) with proportions (50%, 25%, 25% or similar)
        - Train tokenizer using tokenizers library.
        - Evaluate tokenizer fertility and compare to qwen and llama tokenizers.
    - *Trained tokenizer will be used for base model training.*
    - Prepare explanation of what is tokenizer and how it works and how you trained it.
- **[Week3]** Start working on quantization techniques.
    - Research about quantization techniques. (AWQ, GTPQ, GGUF or new ones)
- **[Week4]** Apply quantization techniques to the model.
    - Evaluate the performance of the quantized model ask SFT student to help with it.
- **[Week5]** Help with base model training.
- **[Week6]** Help with SFT.
- **[Week7]** Help with DPO.
- **[Week8]** Final presentation.

## Base model training.
- **[Week1]** Reserach week. 
    - Read papers of training from scratch write down important points about it: 
        - Data
        - Model
        - Training
        - Evaluation
- **[Week2]** Understand the nemotron framework and how to use it.
- **[Week3]** Prepare the data for the initial training of base model: 
    - "1 phase base training on dirty data"
    - "2 phase base training on clean data"
- **[Week4]** Start experimenting with the training process.
    - While model is training, prepare the evaluation strategy.
    - Evaluate the model after each "n" step.
    - *Checkpoints of the model will be provided to the SFT student.*
- **[Week5]** Improve the dataset for the base model training.
    - Collect more data.
    - Clean the data.
- **[Week6]** Explore ways to optimize the training process.
    - Write cuda optimization tricks.
    - Use triton kernels for better performance.
    - Use flash attention.
- **[Week7]** Optimize the training code: Research/implement/test.
    - Optimize the training code.
    - Test the optimized code.
    - Compare the performance of the optimized code with the original code.
- **[Week8]** Final presentation.
    - Present the results of the work.
    - Share the code.
    - Share the model.

## SFT (Supervised fine-tuning)
- **[Week1]** Research week. 
    - Read papers and articles about SFT how other did it for small models and big models and for low resource languages.
        - [SFT paper](https://arxiv.org/abs/2109.08668)
        - [SFT paper](https://arxiv.org/abs/2109.08668)
        - [SFT paper](https://arxiv.org/abs/2109.08668)
- **[Week2]** Prepare the data for the SFT.
- **[Week3]** Train the model.
    - Evaluate the model.
- **[Week4]** DPO training.
    - Read papers and articles about DPO.
    - Prepare the data for the DPO.
    - Train the model.
    - Evaluate the model.
- **[Week5]** Function calling training.
    - Read papers and articles about function calling.
    - Prepare the data for the function calling.
    - Train the model.
    - Evaluate the model.
- **[Week6]** Reward model training.
    - Read papers and articles about reward model.
    - Prepare the data for the reward model.
    - Train the model.
    - Evaluate the model.
- **[Week7]** Audio and image understanding integration.
    - Image understanding integration.
        - Read papers and articles about image understanding.
        - Prepare the data for the image understanding.
        - Train the model.
        - Evaluate the model.
    - Audio understanding integration.
        - Read papers and articles about audio understanding.
        - Prepare the data for the audio understanding.
        - Train the model.
        - Evaluate the model.
- **[Week8]** Final presentation.
    - Present the results of the work.
    - Share the code.
    - Share the model.

<!-- #### Other tasks to consider:
    - DPO training.
    - Function calling training/usage.
    - Reward model training. -->
