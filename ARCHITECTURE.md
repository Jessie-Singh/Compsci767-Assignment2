# Agent Architecture & Flow

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSV Analysis Agent                           │
│        (OpenAI-assisted local tool execution agent)             │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                      PERCEIVE PHASE                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ load_csv() → Read file                                  │   │
│  │           → Store in memory                             │   │
│  │           → Report shape & columns                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                      DECIDE PHASE                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OpenAI LLM (GPT-4o-mini)                               │   │
│  │ ├─ Input: Dataset summary                              │   │
│  │ ├─ Output: Recommended tool actions                   │   │
│  │ └─ Action parsing via parse_action_list()              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                        ACT PHASE                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Local tool dispatch using TOOL_MAP                       │   │
│  │  (deterministic Python function map)                     │   │
│  │                                                         │   │
│  │  Executes selected actions:                              │   │
│  │  - inspect_data                                          │   │
│  │  - missing_value_report                                  │   │
│  │  - summary_statistics                                    │   │
│  │  - category_counts                                       │   │
│  │  - create_histograms                                     │   │
│  │  - correlation_heatmap                                   │   │
│  │                                                         │   │
│  │  Result: Analysis outputs appended to findings          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                      SUMMARIZE PHASE                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OpenAI LLM (GPT-4o-mini)                               │   │
│  │ ├─ Input: All collected findings                        │   │
│  │ ├─ Process: Synthesize insights                          │   │
│  │ └─ Output: Human-readable summary                        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────┐
│                   FOLLOW-UP QUESTION PHASE                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OpenAI LLM decides whether additional tools are needed   │   │
│  │ ├─ Input: prior findings + user question                │   │
│  │ ├─ Output: extra tool actions or []                     │   │
│  │ └─ Result: refined answer and optional extra analysis    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

                              ↓

            User gets interactive follow-up answers
```

## Data Flow

```
CSV File
   ↓
   └─→ load_csv()
        ↓
        └─→ pandas.read_csv()
             ↓
             └─→ Global memory
                  ├─ memory.df (DataFrame)
                  ├─ memory.csv_path
                  ├─ memory.findings (List)
                  └─ memory.plots (List)
                       ↓
                       └─→ get_data_summary()
                            ↓
                            └─→ LLM decision prompt
                                 ↓
                                 └─→ parse_action_list()
                                      ↓
                                      └─→ TOOL_MAP dispatch
                                           ├─ inspect_data()
                                           ├─ missing_value_report()
                                           ├─ summary_statistics()
                                           ├─ category_counts()
                                           ├─ create_histograms()
                                           └─ correlation_heatmap()
                                                ↓
                                                └─→ Findings collected
                                                     ↓
                                                     └─→ LLM summarization
                                                          ↓
                                                          └─→ User output
```

## Component Interactions

### DataAnalysisAgent

```python
┌─────────────────────────────────────┐
│    DataAnalysisAgent                │
├─────────────────────────────────────┤
│ Methods:                            │
│ • perceive()    → Load data         │
│ • decide()      → LLM recommends tool actions
│ • act()         → Execute local tools
│ • summarize()   → LLM synthesizes findings
│ • follow_up()   → Handle interactive follow-up questions
│ • run()         → Full cycle        │
├─────────────────────────────────────┤
│ Attributes:                         │
│ • llm          → ChatOpenAI()       │
│ • actions      → Selected tool list │
│ • findings     → Results collected  │
│ • TOOL_MAP     → Local action map   │
└─────────────────────────────────────┘
         ↓         ↓         ↓
        LLM      Tools    Memory
```

### OpenAI Integration

```
┌───────────────────────────────────────────┐
│     ChatOpenAI Instance                  │
│  (GPT-4o-mini, temperature=0.2)          │
├───────────────────────────────────────────┤
│                                           │
│  Used for:                                │
│  1. Decision making                       │
│     └─→ decide()                           │
│                                           │
│  2. Summarization                         │
│     └─→ summarize()                        │
│                                           │
│  3. Follow-up question processing         │
│     └─→ follow_up()                        │
│                                           │
│  4. Tool selection parsing                │
│     └─→ parse_action_list()                │
└───────────────────────────────────────────┘
```

## Tool System

```
┌─────────────────────────────────────────┐
│         Local Tool Dispatch            │
├─────────────────────────────────────────┤
│                                         │
│ inspect_data()                          │
│ └─→ Reports data types, missing values, │
│     numeric and categorical columns     │
│                                         │
│ missing_value_report()                  │
│ └─→ Summarizes missing values           │
│                                         │
│ summary_statistics()                    │
│ └─→ Generates numeric statistics        │
│                                         │
│ category_counts()                       │
│ └─→ Returns top categorical frequencies │
│                                         │
│ create_histograms()                     │
│ └─→ Saves histogram PNG files           │
│                                         │
│ correlation_heatmap()                   │
│ └─→ Saves a correlation heatmap PNG     │
│                                         │
│ get_data_summary()                      │
│ └─→ Provides high-level dataset info    │
└─────────────────────────────────────────┘
```

## Follow-up Support

The follow-up loop in `main.py` allows a user to ask additional questions after the initial analysis. Each follow-up:

- uses prior `findings` and the current dataset state
- asks the LLM whether more tool-based analysis is needed
- optionally executes extra actions from `TOOL_MAP`
- then returns a concise answer

## Memory System

```
┌─────────────────────────────────────────┐
│  DataAnalysisMemory                    │
├─────────────────────────────────────────┤
│ df          → pandas DataFrame         │
│ csv_path    → input CSV path           │
│ findings    → Collected result strings │
│ plots       → Generated PNG filenames  │
└─────────────────────────────────────────┘
```

## Error Handling

```
Perceive Phase:
  load_csv() handles file and parse errors
  returns a friendly error message if CSV loading fails

Decide Phase:
  if LLM returns no valid actions,
  defaults to ["inspect_data", "summary_statistics"]

Act Phase:
  unknown tool names produce a warning result,
  but do not crash the agent

Follow-up Phase:
  if no dataset exists, user is prompted to load a CSV first
  extra tool analysis is only executed when the LLM requests it
```

## API Calls

```
┌───────────────────────────────────────────┐
│        OpenAI API Calls                   │
├───────────────────────────────────────────┤
│                                           │
│ Call 1: DECIDE Phase                      │
│ ├─ Model: gpt-4o-mini                     │
│ ├─ Input: dataset summary                 │
│ └─ Output: selected tool names            │
│                                           │
│ Call 2: SUMMARIZE Phase                   │
│ ├─ Model: gpt-4o-mini                     │
│ ├─ Input: analysis findings               │
│ └─ Output: human-readable summary         │
│                                           │
│ Call 3: FOLLOW-UP Phase                   │
│ ├─ Model: gpt-4o-mini                     │
│ ├─ Input: follow-up question + findings   │
│ └─ Output: optional actions + answer      │
└───────────────────────────────────────────┘
```

## Execution Flow Example

```
User Input: "data/Gaming_Academic_Performance.csv"
                              ↓
                    perceive(csv_path)
                      ├─ load_csv()
                      ├─ memory.df updated
                      └─ finding recorded
                              ↓
                         decide()
                      ├─ get_data_summary()
                      ├─ LLM prompt
                      ├─ OpenAI API call
                      ├─ parse_action_list()
                      └─ selected actions
                              ↓
                         act()
                      ├─ Execute local tools via TOOL_MAP
                      ├─ Append results to findings
                      └─ Optional plots written to disk
                              ↓
                      summarize()
                      ├─ LLM prompt with findings
                      ├─ OpenAI API call
                      └─ Summary returned
                              ↓
            User sees initial summary and prompt for follow-up
                              ↓
                      follow_up(question)
                      ├─ Evaluate need for extra tools
                      ├─ Optionally execute more tools
                      ├─ LLM answer prompt
                      └─ Follow-up answer returned
```

## Technology Stack

```
┌─────────────────────────────────┐
│      Python 3.8+               │
├─────────────────────────────────┤
│ Data Processing:               │
│ ├─ pandas              (CSV I/O)
│ ├─ matplotlib   (Visualization)
│                                │
│ AI/LLM:                        │
│ ├─ openai       (API client)   │
│ ├─ langchain-openai (ChatOpenAI integration)
│                                │
│ Configuration:                 │
│ ├─ python-dotenv    (.env mgmt)
│                                │
└─────────────────────────────────┘
```

---
