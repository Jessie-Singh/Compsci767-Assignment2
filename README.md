# OpenAI-Powered CSV Analysis Agent

## Overview

This project is an intelligent CSV analysis agent that uses OpenAI and local Python tools to analyze tabular data.
The agent performs a full **Perceive → Decide → Act → Summarize** cycle, then supports interactive follow-up questions.

Key behavior:

- Loads CSV data into memory
- Uses GPT-4o-mini to choose the best analysis steps
- Executes deterministic Python tools locally
- Generates a human-readable summary
- Answers follow-up questions with optional extra analysis

## Setup and Reproducibility

### 1. Prepare the environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure OpenAI credentials

Copy the example environment file and set your API key:

```bash
cp .env.example .env
```

Edit `.env` and add:

```text
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the agent

```bash
python main.py
```

Then enter the sample CSV path when prompted:

```text
Enter CSV file path: data/Gaming_Academic_Performance.csv
```

### 4. Reproduce the results

After the initial analysis, you can ask follow-up questions interactively.
The agent keeps the dataset and findings in memory, so follow-ups are answered using prior results plus additional tools if needed.

Example follow-up sequence:

```text
Follow-up question: What column has the strongest correlation with Grades?
```

Press Enter on an empty prompt to exit.

## Code Workflow

### 1. Perceive

- `DataAnalysisAgent.perceive(csv_path)` loads the CSV into shared memory
- The agent records dataset shape, columns, and any load issues

### 2. Decide

- `DataAnalysisAgent.decide()` sends the dataset summary to the OpenAI model
- The model returns a JSON array of tool names to execute
- Default fallback: `inspect_data` and `summary_statistics`

### 3. Act

- `DataAnalysisAgent.act()` executes selected functions from `TOOL_MAP`
- Tools include:
  - `inspect_data`
  - `missing_value_report`
  - `summary_statistics`
  - `category_counts`
  - `create_histograms`
  - `correlation_heatmap`
- Outputs are appended to `self.findings`

### 4. Summarize

- `DataAnalysisAgent.summarize()` sends findings to the LLM
- Returns a concise AI-generated summary

### 5. Follow-up

- `DataAnalysisAgent.follow_up(question)` evaluates whether extra tools are needed
- Optionally runs more analysis
- Generates a clear answer using prior findings and extra results

## Reproducible Results

To reproduce the exact analysis flow shown in this repository:

1. Activate the virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Set `OPENAI_API_KEY` in `.env`
4. Run `python main.py`
5. Provide `data/Gaming_Academic_Performance.csv` when prompted
6. Review the printed analysis summary and generated images

Generated output files may include histogram and heatmap PNGs in the repository root.

Demo Video: https://github.com/user-attachments/assets/743080f4-42bc-4c68-bf4b-0bdc87f6e9f3




