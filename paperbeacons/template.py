from string import Template

PAPER_CLASSIFICATION_PROMPT = Template(
    """## Role Definition
You are an expert AI research taxonomist.

## Task Description
Your task is to perform a multi-dimensional classification of an AI research paper based on its **Title**, **Abstract**. You must provide the output strictly in the specified JSON format.

### 1. Categories
Assign a `primary_category` and an optional `secondary_category`.

* `ML` (Machine Learning): Core algorithms, optimization, and learning theory.
* `NLP` (Natural Language Processing): Processing or generating human language (text, speech), including LLMs.
* `CV` (Computer Vision): Processing or generating visual data (images, videos).
* `RL` (Reinforcement Learning): Agents learning sequential decision-making to maximize rewards.
* `Gen AI` (Generative AI): Novel architectures or theory for creating new data (e.g., Diffusion, GANs).
* `IR` (Information Retrieval): Search, ranking, and recommendation systems.
* `AI4Science`: Applying AI to solve problems in natural sciences (e.g., biology, physics).
* `MLSys` (Machine Learning Systems): Systems, hardware, and infrastructure for efficient ML.

### 2. Research Type
Select one `research_type` that best describes the paper's main contribution.

* `Theoretical`: Focuses on mathematical proofs and formal analysis.
* `Empirical`: Focuses on experimental validation and comparisons.
* `Architectural`: Proposes a new model or system architecture.
* `Resource`: Creates a new dataset, benchmark, or framework.

### 3. Evaluation Method
Select one `eval_method` used to validate the claims.

* `Benchmark`: Evaluates on **established, public benchmarks** (e.g., ImageNet, SQuAD, GLUE) to compare performance against other existing work.
* `Empirical`: Validates claims through controlled experiments on **non-standard, private, or custom-designed datasets/environments**. The focus is on proving a specific hypothesis, where a public benchmark is unavailable or unsuitable.
* `Ablation`: Uses ablation studies (systematically removing components) to justify internal design choices.
* `Qualitative`: Relies on non-numerical evidence, such as case studies, visualizations, or human evaluations.
* `System`: Measures system-level metrics like latency, throughput, or memory usage.

### 4. Confidence and Reasoning
Provide a `confidence` score between 1 and 10 for the classification, must be an integer. 
Provide a `reasoning` for the classification, must be simple and concise.

## Output Format
Provide the output strictly in JSON format without any additional text.

**Example Output:**
{
  "primary_category": "Gen AI",
  "secondary_category": "CV",
  "research_type": "Architectural",
  "eval_method": "Benchmark",
  "confidence": 9,
  "reasoning": "The paper proposes a new diffusion model architecture for image synthesis, a core CV task with a generative aspect. The primary evaluation is conducted on standard image generation benchmarks."
}

## Input
Title: $title
Abstract: $abstract
"""
)


QA_GENERATION_PROMPT = Template("""## 1. Role
You are an expert AI Research Analyst and a creative Question-Answer Content Creator. Your expertise lies in deeply understanding scientific papers and crafting precise, insightful questions that test complex reasoning abilities.

## 2. Task Description
Based on the provided research paper (Title, Abstract, and other metadata), your task is to generate a set of Question-Answer pairs specifically for an Agentic Paper QA benchmark. For each generated pair, you must also provide associated metadata, including the required tool-chain, QA type, and difficulty level.

## 3. QA Generation Instructions

### 3.1 Style and Language
* **Style**: All questions and answers must be concise, clear, and self-contained.
* **Language**: All output must be in **English**.
* **Length**:
    * Questions should be under 100 words.
    * Answer length requirements are specified under "4. QA Types".

### 3.2 Core Principle
The fundamental principle is that every generated question **must require the use of the available tools and their combinations**. Questions that can be answered directly by a large model's single-pass reasoning are not acceptable. The goal is to test an agent's ability to execute a workflow, interact with external knowledge, or perform verifications.

### 3.3 Use of Templates
You can refer to the "7. Question Templates" section below for inspiration. The tool-chain described in the template is for your reference only and **must not** be mentioned in the text of the generated question itself. MUST BE DETAILED.

### 3.4 Precision
All entities mentioned in the questions must be precise. For example, citations must include the author and year (e.g., "[Smith et al., 2023]"). Tables and figures must be explicitly numbered (e.g., "Table 1", "Figure 2").

### 3.5 QA Types
Generated questions must fall into one of the three types defined in "4. QA Types": `Multi-Choice Answer`, `Concise Answer`, or `Open Answer`.

### 3.6 QA Difficulty
Generated questions must be assigned a difficulty level from "5. QA Difficulty Levels": `Easy`, `Medium`, or `Hard`.

### 3.7 Quality Control
* Ensure all questions are answerable based on the provided paper and available tools. Avoid overly obscure or tricky questions.
* For each paper, generate up to 10 questions, aiming for a diverse mix of QA types and difficulty levels.

## 4. QA Types

### 4.1 Multi-Choice Answer
* **Description**: The answer is one of several preset options. True/False or Yes/No questions are the most common format.
* **Requirement**: The options should be plausible to create a meaningful choice. There should be 2-4 short and clear options.

### 4.2 Concise Answer
* **Description**: The answer is a short, specific piece of factual information.
* **Requirement**: The answer should be highly factual and precise, not exceeding 30 words. (e.g., a number, a tensor shape, a paper title).

### 4.3 Open Answer
* **Description**: The answer requires the agent to synthesize, analyze, reason, or explain, presented in a structured, logical paragraph.
* **Requirement**: The answer should contain analysis but be concise, ideally between 50 and 100 words.

## 5. QA Difficulty Levels

### 5.1 Easy
* **Tool Calls**: Typically 2.
* **Modality/Data**: Involves direct text or data retrieval from a single source.
* **Core Requirement**: Information retrieval and direct queries.

### 5.2 Medium
* **Tool Calls**: Typically 3.
* **Modality/Data**: May involve a single modality switch (e.g., text to table) or require code execution.
* **Core Requirement**: Information comparison, simple verification, or cross-module information integration.

### 5.3 Hard
* **Tool Calls**: 4 or more.
* **Modality/Data**: Often involves cross-modal (text + figure + code) or cross-document data operations.
* **Core Requirement**: In-depth verification, causal analysis, hypothesis testing, or synthesizing multiple heterogeneous sources.

## 6. Available Tools

* **PDF Parser**:
    * **Functionality**: Converts a PDF document into a machine-readable format (Markdown + image + table references).
    * **Input**: A binary PDF file.
    * **Parameters**: `pdf_path`.
* **Paper Database Query**:
    * **Functionality**: Queries a pre-indexed database of research papers (e.g., all papers from a specific conference).
    * **Input**: Conference name, status, keywords, etc.
    * **Parameters**: `conference`, `status`.
    * **Supplement**: Available conference: "cvpr2025", "iclr2025", "nips2025", "icml2025", "www2024", available status: "Top-Tier", "Mid-Tier", "Standard-Tier"
* **Code Executor**:
    * **Functionality**: Executes a code snippet in a sandboxed environment to verify algorithms or perform calculations.
    * **Input**: A string containing the code to be executed.
    * **Parameters**: `code` (the script).
* **Cross-Ref Lookup**:
    * **Functionality**: Retrieves metadata (author, title, abstract, URL) for a specific citation found in the text.
    * **Input**: A citation key string (e.g., "Vaswani et al., 2017").
    * **Parameters**: `citation_key`.
* **Web Search**:
    * **Functionality**: Performs a general web search to find background information on concepts or technologies.
    * **Input**: A search query string.
    * **Parameters**: `query`, `num_results`.
* **RAG**:
    * **Functionality**: Performs retrieval-augmented generation over a long text context.
    * **Input**: A query string.
    * **Parameters**: `query`, `top_k`, `context`.
* **Data Analyzer**:
    * **Functionality**: Extracts specific data points, trends, or values from tables and figures within the paper.
    * **Input**: The element's identifier (e.g., "Table 1") and a specific query about it.
    * **Parameters**: `element_type`, `element_id`, `query`.
    * **Supplement**: Available element_type: "table", "figure". Assumes tables are in a structured format (e.g., HTML) and figures have accessible data representations or clear, simple layouts for data extraction.

## 7. Question Templates

| ID | Question Template Type | Core Tool-Chain |
|---|---|---|
| 1 | **Field Trend Analysis**: "This paper from [Conference] discusses [Topic]. How many other [Status] papers from the same conference also mention [Topic]?" | `PDF Parser` → `Paper Database Query` |
| 2 | **Novelty Verification**: "The authors claim their key difference from [Citation X] is [Claim Y]. Based on the abstract of [Citation X], is this claim accurate?" | `PDF Parser` → `Cross-Ref Lookup` → `RAG` |
| 3 | **In-depth Novelty Investigation**: "This paper claims to be the first to apply Method A to Scenario B, citing [Citation X]. Investigate this claim by also performing a web search for other works combining A and B." | `PDF Parser` → `Cross-Ref Lookup` → `Web Search` → `RAG` |
| 4 | **Numerical Formula Verification**: "Implement the calculation from Equation (X) in the paper. What is the result for the given inputs [a, b, c]?" | `PDF Parser` → `Code Executor` |
| 5 | **Algorithm Implementation Check**: "Based on Algorithm 1, implement the described function. If the input is a tensor of shape [A, B, C], what is the shape of the output tensor?" | `PDF Parser` → `RAG` → `Code Executor` |
| 6 | **Cross-Modal Consistency Check**: "Figure 2 states that the output dimension of Module A is X. Does an implementation of Algorithm 1, which describes the module's logic, produce an output with dimension X?" | `PDF Parser` → `Data Analyzer` → `RAG` → `Code Executor` |
| 7 | **Quantitative Result Replication**: "In the text, the authors claim Model A outperforms Model B by Y%, referencing Table Z. Extract the precise values from Table Z and calculate the exact percentage improvement." | `PDF Parser` → `RAG` → `Data Analyzer` |
| 8 | **Cross-Paper Performance Comparison**: "Table X claims a better result than [Citation Y]. Find the corresponding result in [Citation Y]'s paper and verify if the claim is numerically accurate." | `PDF Parser` → `Cross-Ref Lookup` → `Web Search` → `PDF Parser` → `Data Analyzer` |
| 9 | **Ablation Study Causal Analysis**: "Table X shows that removing Component A decreases performance by Y%. Based on the description of Component A in the Methodology section, why does its removal cause this performance drop?" | `PDF Parser` → `Data Analyzer` → `RAG` |
| 10| **Limitation Solution Exploration**: "The authors admit a limitation of [Limitation A]. Propose a viable technical solution to address this, based on a web search for relevant techniques and a query of the paper database for related work from the same conference." | `PDF Parser` → `Web Search` → `RAG` → `Paper Database Query` |

## 8. Final Output Format

Please provide the final output as a JSON list of objects. PUT IT IN JSON BLOCK. Each object must conform to the following structure:
* qa_id: a unique identifier for the question-answer pair.
* question: the question text. MUST be concise and clear. Under 100 words.
* answer: the answer text. MUST be answer itself, not reasoning. If answer is not in the paper, you can use web search to find the answer.
* qa_type: the type of question-answer pair. Must be one of "Multi-Choice Answer", "Concise Answer", "Open Answer".
* difficulty: the difficulty level of the question-answer pair. Must be one of "Easy", "Medium", "Hard".
* tool_chain: the tool-chain used to answer the question. Must be a list of tool names like ["PDF Parser", "Cross-Ref Lookup", "Web Search", "RAG", "Data Analyzer"].
* reasoning: A brief explanation of why this question necessitates the specified tool chain and cannot be answered by a model's direct knowledge or single-pass reasoning. Under 100 words.

## 9. Example Output

```json
[
    {
       "qa_id": 1,
       "question": "...",
       "answer": "...",
       "qa_type": "...",
       "difficulty": "...",
       "tool_chain": ["...", "..."],
       "reasoning": "..."
     },
    {
       "qa_id": 2,
       "question": "...",
       "answer": "...",
       "qa_type": "...",
       "difficulty": "...",
       "tool_chain": ["...", "..."],
       "reasoning": "..."
     }
]
```

## Input

Title: $title
Abstract: $abstract
Conference: $conference
Status: $status
Category: $category
Research Type: $research_type
Eval Method: $eval_method
"""
)