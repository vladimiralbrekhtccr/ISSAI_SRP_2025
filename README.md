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


### **Week 2: Base Model Architecture & Training Setup**
*Theme: Understanding the process of training base model*

**Collaborative Goal**: Understanding the process of training base model and set up training infrastructure.

`NOTE: here is huge gap in terms of resources, so we need to find different frameworks that can help us to train the model`

**Study Resources**:

-- Main frameworks that we can use:
- Gitgub [nanotron](https://github.com/huggingface/nanotron/tree/main)

-- Papers to read to better understand the process of training base model:
- Qwen3 paper
- Gemma3 paper
- SmolLM2 paper 
- Deepseek paper




`NOTE: everything below is rough plan that will change based on how fisrt 2 weeks will go`

### **Week 3: Data Preparation & Training Launch**
*Theme: Data Processing and Initial Model Training*

**Collaborative Goal**: Launch base model training with clean, well-processed multilingual data

**Individual Perspectives**:
- **Student A**: Data cleaning and quality filtering (deduplication, language detection, quality scoring)
- **Student B**: Data preprocessing pipeline (tokenization, sequence packing, batch preparation)
- **Student C**: Training monitoring and logging (loss curves, learning rate scheduling, gradient norms)
- **Student D**: Distributed training optimization (data parallelism, gradient accumulation, mixed precision)

**Shared Activities**:
- Implement data processing pipeline
- Launch initial training runs
- Monitor training stability and convergence
- Debug training issues collaboratively

**Week 3 Deliverables**:
- Clean training dataset (multiple domains, balanced languages)
- Stable training pipeline
- Initial model checkpoints
- Training monitoring dashboard

### **Week 4: Training Optimization & Evaluation**
*Theme: Improving Training Efficiency and Model Performance*

**Collaborative Goal**: Optimize training process and establish comprehensive evaluation

**Individual Perspectives**:
- **Student A**: Training optimization (flash attention, gradient checkpointing, memory optimization)
- **Student B**: Learning rate scheduling and hyperparameter tuning
- **Student C**: Evaluation on downstream tasks (text generation, basic reasoning)
- **Student D**: Multilingual evaluation (cross-lingual transfer, language-specific performance)

**Shared Activities**:
- Implement training optimizations
- Run systematic hyperparameter experiments
- Evaluate intermediate checkpoints
- Prepare base model for fine-tuning

**Week 4 Deliverables**:
- Optimized training configuration
- Comprehensive evaluation results
- Best base model checkpoint
- Performance analysis report

### **Week 5: Supervised Fine-Tuning (SFT)**
*Theme: Teaching the Model to Follow Instructions*

**Collaborative Goal**: Transform base model into instruction-following assistant

**Individual Perspectives**:
- **Student A**: Instruction dataset creation and curation (multilingual instructions, diverse tasks)
- **Student B**: SFT training implementation (learning rates, data formatting, loss functions)
- **Student C**: Conversation and chat formatting (system prompts, multi-turn dialogues)
- **Student D**: SFT evaluation (instruction following, safety, multilingual capabilities)

**Shared Activities**:
- Study SFT methodologies and best practices
- Curate high-quality instruction datasets
- Implement SFT training pipeline
- Evaluate instruction-following capabilities

**Week 5 Deliverables**:
- Multilingual instruction dataset
- SFT training pipeline
- Instruction-tuned model checkpoint
- SFT evaluation results

### **Week 6: Advanced Fine-Tuning Techniques**
*Theme: DPO, Function Calling, and Specialized Capabilities*

**Collaborative Goal**: Implement advanced alignment and capability techniques

**Individual Perspectives**:
- **Student A**: Direct Preference Optimization (DPO) implementation and training
- **Student B**: Function calling capability (tool use, API integration, structured outputs)
- **Student C**: Reward model training for RLHF pipeline
- **Student D**: Safety alignment and evaluation (harmful content detection, refusal training)

**Shared Activities**:
- Study DPO and RLHF methodologies
- Implement preference learning
- Design function calling interfaces
- Test advanced capabilities

**Week 6 Deliverables**:
- DPO-aligned model
- Function calling capabilities
- Reward model
- Safety evaluation results

### **Week 7: Model Optimization & Deployment**
*Theme: Making Models Production-Ready*

**Collaborative Goal**: Optimize models for deployment and real-world use

**Individual Perspectives**:
- **Student A**: Model quantization (INT8, INT4, GGUF formats)
- **Student B**: Inference optimization (KV caching, batching, serving frameworks)
- **Student C**: Model packaging and distribution (HuggingFace Hub, model cards, documentation)
- **Student D**: Performance benchmarking (latency, throughput, memory usage)

**Shared Activities**:
- Implement quantization techniques
- Set up model serving infrastructure
- Create comprehensive documentation
- Benchmark model performance

**Week 7 Deliverables**:
- Quantized model variants
- Deployment-ready model packages
- Complete documentation and model cards
- Performance benchmark results

### **Week 8: Multimodal Integration & Final Presentation**
*Theme: Extending to Vision and Audio (If Time Permits)*

**Collaborative Goal**: Explore multimodal capabilities and present final results

**Individual Perspectives**:
- **Student A**: Vision encoder integration (image understanding, vision-language alignment)
- **Student B**: Audio encoder integration (speech recognition, audio understanding)
- **Student C**: Multimodal training pipeline (interleaved data, cross-modal attention)
- **Student D**: Multimodal evaluation and demonstration applications

**Shared Activities**:
- Research multimodal architectures
- Experiment with vision/audio integration
- Prepare comprehensive final presentation
- Plan community release and documentation

**Week 8 Deliverables**:
- Multimodal model experiments (if achieved)
- Complete project documentation
- Final presentation and demo
- Open-source release on HuggingFace




# Some notes


## Daily Structure

### **Morning Standup** (15 minutes)
- What did you accomplish yesterday?
- What are you working on today?
- Any blockers or questions?
- Knowledge sharing opportunities

### **Deep Work Sessions** (3-4 hours)
- Individual focused work on specialized perspectives
- Pair programming when beneficial
- Research and implementation time

### **Afternoon Sync** (30 minutes)
- Progress updates and problem-solving
- Code review sessions
- Planning next steps
- Cross-pollination of ideas

### **Friday Knowledge Sharing** (90 minutes)
- Weekly presentations by each student
- Demonstration of achievements
- Group discussion and learning
- Planning for next week

## Success Metrics

### **Learning Outcomes**
- Deep understanding of entire LLM training pipeline
- Hands-on experience with state-of-the-art techniques
- Collaborative problem-solving skills
- Open-source contribution experience

### **Technical Deliverables**
- Custom multilingual tokenizer
- Trained base model (200M-500M parameters)
- Instruction-tuned assistant model
- Quantized and optimized model variants
- Comprehensive documentation and benchmarks

### **Community Impact**
- First open-source small LLM trained on Kazakh
- Reusable training pipeline and methodologies
- Educational resources for multilingual LLM training
- Foundation for future research and applications

## Resources and Infrastructure

### **Computing Requirements**
- GPU access for training (recommended: A100 or V100)
- Distributed training setup for larger experiments
- Storage for datasets and model checkpoints

### **Key Technologies**
- HuggingFace Transformers and Tokenizers
- NeMo or Megatron for distributed training
- PyTorch for model implementation
- Weights & Biases for experiment tracking

### **Learning Resources**
- Research papers on tokenization, training, and fine-tuning
- Video tutorials and online courses
- Open-source codebases and implementations
- Community forums and discussion groups
