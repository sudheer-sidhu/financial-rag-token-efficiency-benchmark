
**TOON vs JSON for Structured Financial Tables (10-K Filings)**

**Overview**

Large Language Models operate under strict context window limits.
Financial documents such as SEC 10-K filings contain dense, multi-column tables that rapidly exhaust these token budgets when encoded in JSON.

This project evaluates whether Token-Oriented Object Notation (TOON) can significantly reduce token consumption compared to JSON while preserving structural integrity for financial RAG systems.

Using 90 large financial tables extracted from Apple Inc. 10-K filings (2023–2025), we benchmark token usage, byte size, structural correlations, and effective context window capacity.

**The result:**
TOON reduces token usage by 47.65% on average, nearly doubling effective evidence capacity within fixed LLM context windows.

**Problem**

JSON is the de facto standard for structured data, but it repeats field names in every row:

{"Revenue": 50000000, "Net Income": 5000000}
{"Revenue": 52000000, "Net Income": 5200000}

For wide financial tables, this redundancy creates severe token inefficiency in LLM pipelines.

In financial RAG systems, wasted tokens directly reduce:

Historical coverage

Multi-table comparison capacity

Cross-company reasoning

Numerical synthesis accuracy

Cost efficiency

This project quantifies that inefficiency and evaluates a structured alternative.

**Methodology**

**Dataset:**

90 large financial tables

Apple Inc. 10-K filings (2023–2025)

Extracted from SEC EDGAR

**Evaluation:**

Token counting via cl100k_base tokenizer

Byte size comparison

Pearson correlation analysis

Context window simulation under:

4K tokens

8K tokens

16K tokens

**Metrics:**

Absolute token count

Percentage token reduction

Context window table capacity

Structural correlations (rows, columns, precision)

**Key Results**
**1. Aggregate Token Efficiency**

Average Token Reduction: 47.65%

Average Byte Reduction: 36.14%

Median reduction: 48.81%

Consistent gains across all 90 tables

**2. Structural Correlation Findings**

Moderate positive correlation with number of columns (r = 0.4465)

Moderate positive correlation with total cells (r = 0.3929)

Strong negative correlation with numerical precision (r = −0.5149)

These results indicate that structural redundancy drives compression gains.

**3. Context Window Simulation**
Token Budget	JSON Tables	TOON Tables	Gain
4,096	            0	          2	+2
8,192            	1	          3	+2
16,384	          3	          6	+3

TOON approximately doubles effective evidence capacity in constrained LLM environments.

**Why This Matters**

**In financial AI systems, token efficiency directly impacts:**

Evidence completeness

Numerical reasoning depth

Multi-document synthesis

API cost efficiency

Context window scalability

This benchmark demonstrates that structured encoding strategy is not a formatting detail — it is a systems-level performance lever.

**Future Work**

Evaluate downstream QA accuracy impact

Apply TOON to multi-company comparative analysis

Benchmark under 128K context models

Integrate into production financial RAG systems

