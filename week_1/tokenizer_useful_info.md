## 1. How to measure tokenizer performance?
**Formula:** `Fertility = Characters / Tokens`

### Example
**Text:** "I want to say the he" *(20 characters)*

| Tokenizer | Tokens | Fertility | Efficiency |
|-----------|--------|-----------|------------|
| Tokenizer 1 | 20 | 1.0 | Low |
| Tokenizer 2 | 10 | 2.0 | High |

> **Note:** Higher fertility = better efficiency. Each token represents more characters, reducing storage and processing costs.

> **TODO:** What other metrics we can use to measure the tokenizer performance?
