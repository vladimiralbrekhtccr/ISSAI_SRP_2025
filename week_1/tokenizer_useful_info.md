# How to Measure Tokenizer Performance

## 1. Character-Level Compression Ratio

The **compression ratio** measures how efficiently a tokenizer compresses text by comparing the number of characters to the number of tokens produced.

**Formula:** `Compression Ratio = Total Characters / Total Tokens`

**Example:**
- Text: "I want to say hello" (19 characters, including spaces)
- Tokenizer A produces: 6 tokens → Compression ratio = 19/6 = 3.17
- Tokenizer B produces: 12 tokens → Compression ratio = 19/12 = 1.58

| Tokenizer | Tokens | Compression Ratio | Efficiency |
|-----------|--------|------------------|------------|
| A         | 6      | 3.17             | High       |
| B         | 12     | 1.58             | Low        |

**Interpretation:** Higher compression ratio indicates better efficiency, as each token represents more characters on average.

## 2. Word-Level Fertility

**Fertility** measures how many tokens each word gets split into on average. It indicates how well a tokenizer preserves word boundaries.

**Formula:** `Fertility = Total Tokens / Total Words`

**Example:**
- Text: "I want to say hello" (5 words)
- Tokenizer A: 6 tokens → Fertility = 6/5 = 1.2
- Tokenizer B: 12 tokens → Fertility = 12/5 = 2.4

| Tokenizer | Tokens | Fertility | Word Preservation |
|-----------|--------|-----------|------------------|
| A         | 6      | 1.2       | Good             |
| B         | 12     | 2.4       | Poor             |

**Interpretation:** Lower fertility indicates better word preservation, as words are split into fewer tokens.

## 3. Subword Segmentation Metrics

### Proportion of Continued Words
This measures what percentage of words are split into multiple tokens (subwords). A "continued word" is any word that gets broken down into 2 or more subword tokens instead of remaining as a single token.

**Formula:** `Proportion of Continued Words = Words Split into 2+ Tokens / Total Words`

**Detailed Example:**
- Text: "I want to say hello programming"
- Tokenization results:
  - "I" → ["I"] (1 token - not continued)
  - "want" → ["want"] (1 token - not continued)  
  - "to" → ["to"] (1 token - not continued)
  - "say" → ["say"] (1 token - not continued)
  - "hello" → ["hel", "lo"] (2 tokens - **continued**)
  - "programming" → ["program", "ming"] (2 tokens - **continued**)

- Analysis: 2 out of 6 words were split (continued)
- Proportion of continued words = 2/6 = 0.33 (33%)

**Why This Matters:**
- **Lower proportion** indicates the tokenizer preserves more complete words
- **Higher proportion** suggests more aggressive subword splitting, which may:
  - Break semantic meaning
  - Require longer sequences to represent the same text
  - Impact model performance on word-level tasks

**Comparison Example:**
| Tokenizer | Continued Words | Total Words | Proportion | Word Preservation |
|-----------|----------------|-------------|------------|------------------|
| A         | 2              | 10          | 20%        | Good             |
| B         | 7              | 10          | 70%        | Poor             |

### Average Subwords per Word
For words that are split, this measures the average number of subword tokens generated.

**Formula:** `Avg Subwords = Total Subword Tokens / Words That Were Split`

## Key Performance Indicators Summary

| Metric | Formula | Good Performance | Interpretation |
|--------|---------|------------------|----------------|
| Compression Ratio | Characters / Tokens | Higher values | More efficient encoding |
| Fertility | Tokens / Words | Lower values | Better word preservation |
| Continued Words % | Split Words / Total Words | Lower values | Fewer broken words |
| Avg Subwords | Subword Tokens / Split Words | Lower values | Simpler word splitting |

## Practical Considerations

When evaluating tokenizers, consider the trade-offs between these metrics based on your specific use case:

- **High compression ratio** reduces computational costs but may lose semantic information
- **Low fertility** preserves word boundaries but may result in larger vocabularies
- **Language-specific performance** varies significantly between tokenizers
- **Domain adaptation** may require different optimization priorities

### 🧠 TODO:

What other metrics can we use to measure tokenizer performance?

* Perplexity after tokenization

Data for training.

* KazLLM base filtering

## 2. We will train multilingual because:

1. 3 main lanugages in kazakhstand because a lot of different stuff also written in english

2. It will allow us to SFT on transaltion tasks.

3. Better for the long run.


