## 1. How to measure tokenizer performance?
**Formula:** `Fertility = Characters / Tokens`

### 🔢 **Character-level efficiency**
**Text:** "I want to say the he" *(20 characters)*

Fertility: 1.0 = 20/10

| Tokenizer | Tokens | Fertility | Efficiency |
|-----------|--------|-----------|------------|
| Tokenizer 1 | 20 | 1.0 | Low |
| Tokenizer 2 | 10 | 2.0 | High |

> **Note:** Higher fertility = better efficiency. Each token represents more characters, reducing storage and processing costs.

### 🧩 **Word-level fertility**

"**Fertility**" refers to how many tokens each **word** gets split into on average.

**Formula:**
`Fertility = Total Tokens / Total Words`

**Example (5 words total):**
Text: `"I want to say the he"`

| Tokenizer   | Tokens | Fertility (Tokens/Words) | Efficiency |
| ----------- | ------ | ------------------------ | ---------- |
| Tokenizer 1 | 20     | 4.0                      | Low        |
| Tokenizer 2 | 10     | 2.0                      | High       |

> **TODO:** What other metrics we can use to measure the tokenizer performance?
