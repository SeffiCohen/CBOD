# Chameleon-BOD: Detecting Overfit in LLM Evaluations

Welcome to **Chameleon-BOD** – a repository for our paper **"Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon."** This project provides a meta-evaluation framework for large language models (LLMs) that reveals whether a model’s performance is based on memorized prompt patterns rather than genuine language understanding.

---

## Abstract

Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the **Chameleon Benchmark Overfit Detector (C-BOD)**, a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model’s performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD’s dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation.

---

## Repository Structure

```
Chameleon-BOD/
├── .gitignore
├── README.md          <-- (This file)
├── requirements.txt
├── code/
│   ├── MMLU_Eval.py
│   └── mmlu_rephrase_DS.py
├── Data/
│   └── rephrased_mmlu_test_parallel_temp1_0.json
├── Results/
│   └── all the results json
└── paper/
    └── paper.pdf
```

- **`code/`**: Contains our experimental Python scripts:
  - **`MMLU_Eval.py`** – Evaluates LLM predictions on both original and rephrased prompts.
  - **`mmlu_rephrase_DS.py`** – Uses the DeepSeek API to generate a perturbed (rephrased) dataset.
- **`paper/`**: Contains the LaTeX source files for the paper and all required figures.
- **`requirements.txt`**: Lists all Python dependencies.
- **`.gitignore`** & **`LICENSE`**: Standard files for repository management and licensing.

---

## Requirements

To run the code, you need the following packages:

- **torch** (>= 1.10.0)
- **transformers** (>= 4.25.0)
- **tqdm** (>= 4.60.0)
- **requests** (>= 2.25.0)
- **datasets** (>= 2.0.0)

These dependencies are all listed in the [requirements.txt](requirements.txt) file.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Chameleon-BOD.git
cd Chameleon-BOD
```

### 2. Set Up a Virtual Environment (Optional)

It’s a good idea to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### 4. Running the Experiments

#### a. Evaluate LLM Predictions

To run the evaluation script with a Hugging Face model (for example, using `microsoft/phi-4`):

```bash
python code/MMLU_Eval.py --model microsoft/phi-4 --batch_size 1
```

This command loads the specified model from Hugging Face and evaluates it on the original and rephrased MMLU questions.

#### b. Generate the Rephrased Dataset

Before evaluation, generate a rephrased dataset using the DeepSeek API:

1. **Update the API Key:**  
   Open the file `code/mmlu_rephrase_DS.py` and replace the placeholder `"XXXXXXXXXXXXXX"` with your actual DeepSeek API key.

2. **Run the Script:**

```bash
python code/mmlu_rephrase_DS.py
```

This script processes the MMLU test set and saves a rephrased version to `rephrased_mmlu_test_parallel_temp1_0.json`.

---

