# Multilingual Small LLM Training Program
*Training a 200M-500M parameter LLM from scratch on Kazakh, English, and Russian languages*

## Program Philosophy
**"Peer learning while doing something together is the most potent form of learning."**

This 8-week intensive program follows a collaborative learning approach where all students work together on each major component, bringing different perspectives and expertise to create a comprehensive understanding of the entire LLM training pipeline.

## Program Overview

### Goal
Train a small multilingual LLM (200M-500M parameters) from scratch on Kazakh, English, and Russian languages with a custom tokenizer, making it available on Hugging Face for the community.

### Team Structure
4 students working collaboratively through each phase:
- **Collaborative Sprints**: All students work on the same component each week
- **Specialized Perspectives**: Each student takes a different angle on the same problem
- **Knowledge Sharing**: Weekly presentations and peer teaching sessions
- **Continuous Integration**: Students help each other and share discoveries daily

### Why training small LLM matters
- No existing small models trained specifically on Kazakh and Russian
- Efficient custom tokenizer for improved generation speed
- Open-source contribution for further fine-tuning by the community
- Foundation for chatbots, summarization, translation, QA, and multimodal extensions

## Weekly Program Structure

### **Week 1: Tokenization Foundations**
*Theme: Understanding and Building Tokenizers*

**Collaborative Goal**: All students understand tokenization process and contribute to building our custom tokenizer.

**Study Resources**:

`NOTE: below is only resorcses that I found, but I encourage you to find more resources and share them with the group.`
- Play with tokenizer online to receive some high level understnading [Tiktokenizer](https://tiktokenizer.vercel.app/)
- Watch "[Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=LVfIh85FUiFT4v1p)" by Andrej Karpathy
- Study HuggingFace [Tokenizers library](https://huggingface.co/learn/llm-course/en/chapter6) by huggingface
- Analyze existing tokenizers (Qwen, Gemma, LLaMA)
    - [Qwen tokenizer](https://huggingface.co/Qwen/Qwen2.5-0.5B)
    - [Gemma tokenizer](https://huggingface.co/google/gemma-3-1b-pt)
    - [LLaMA tokenizer](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

`TASK: Create a srcipt/tool to compare the tokenizers in terms of efficiency for 3 languages.`

- Collect and prepare multilingual training data (?% Kazakh, ?% English, ?% Russian)
- Papers to read:
    - [MorphBPE](https://arxiv.org/pdf/2502.00894)
    - [VRCP](https://aclanthology.org/2025.sumeval-2.5.pdf)


`END_GOAL_OF_THE_WEEK: Everyone will understnad what the process of tokenization and we can create initial first version of tokenizer and provide tokenizer to start base model training`

By the end of the week we should answer the following questions:
1. What is tokenization?
2. Why we need different tokenizers for different languages?
3. How tokenizer affects the performance of the model?
4. How tokenizer affects the model training?
5. Which proportions of the data we need to create tokenizer?
