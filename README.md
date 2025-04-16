
# Project : *Training small model (200M-500M) from scratch on Kazakh English and Russian languages.*

## General rules:
    - Each week will have a brief presentation of the work done and sharing knowledge with others.
    - If someone finished the task early, he should help others. 


### Week 1 (Training tokenizer) (1 student):
## Introduction to the project and the goal of it.
    - 

## Training tokenizer process.
    . Understand what is tokenizer and how to train it. 
        . Read paper and articles about tokenization, watch youtube videos about tokenization
            . [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=uscBgajGLQJ8ToQE)
        . Experiment with tokenizer from scratch and tokenizer expansion techniques.
    . Create proportions of dataset (kk, en, ru) for training tokenizer. (50%, 25%, 25% etc.)
    . Train tokenizer.
    . Evaluate tokenizer fertility and compare to qwen and llama tokenizers.
    . Repeat the process to get better results.
    . Provide tokenizer to the model training student.
    . Explain to others how to train tokenizer and what it is.

### Week 1 (Training model) (1 student):
    - Training model process. (As base we take qwen and llama architectures and nemotron framework for training)
    . Understand what is model and how to train it. 
        . Read papers of training from scratch write down important points about: (data, model, training, evaluation)
    . Understand the nemotron framework and how to use it.
    . Prepare the data for the initial training of base model: 
        . "1 phase base training on dirty data"
        . "2 phase base training on clean data"
    . Start experimenting with the training process.
    . While model is training, prepare the evaluation strategy.
    . Evaluate the model after each "n" step.
    . Provide final model to the SFT student. 
    . Explain to others how to train model and what it is.

### Week 1 (SFT) (1 student):
    - SFT process.
    . Understand what is SFT and how to do it. (Read paper and articles about SFT how other did it for small models and big models and for low resource languages)
    . Prepare the data for the SFT.
    . Train the model.
    . Evaluate the model.
    . Explain to others how to do SFT and what it is.

#### Other tasks:
    - DPO training.
    - Function calling training.
    - 
    - Reward model training.
