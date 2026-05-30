import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.tools import memory, load_csv, TOOL_REGISTRY, TOOL_ALIASES

load_dotenv()


def get_data_summary() -> str:
    """Get a quick summary of the loaded data (shape, columns, types)."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    rows, cols = memory.df.shape
    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing = memory.df.isnull().sum()
    missing_info = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
    missing_text = json.dumps(missing_info) if missing_info else "None"
    return (
        f"Data shape: {rows} rows × {cols} columns\n"
        f"Numeric columns: {len(numeric_cols)}\n"
        f"Categorical columns: {len(categorical_cols)}\n"
        f"Missing values: {missing_text}"
    )


class DataAnalysisAgent:
    """An agent that uses OpenAI API plus local tool execution for CSV analysis."""

    TOOL_MAP = TOOL_REGISTRY
    TOOL_DESCRIPTIONS = {
        name: (tool.__doc__ or "").strip().replace("\n", " ")
        for name, tool in TOOL_REGISTRY.items()
    }

    def __init__(self):
        self.actions = []
        self.findings = []
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2,
            verbose=False,
        )

    def perceive(self, csv_path: str) -> str:
        """Load the CSV file into memory and record the initial data state."""
        result = load_csv(csv_path)
        self.findings.append(result)
        return result

    def parse_action_list(self, text: str) -> list[str]:
        """Parse a list of tool action names from an LLM response."""
        text = text.strip()

        try:
            candidate = json.loads(text)
            if isinstance(candidate, list):
                actions = []
                for item in candidate:
                    if not isinstance(item, (str, int)):
                        continue
                    action = str(item).strip().lower()
                    for key, aliases in TOOL_ALIASES.items():
                        if action in aliases:
                            action = key
                            break
                    if action in self.TOOL_MAP and action not in actions:
                        actions.append(action)
                return actions
        except json.JSONDecodeError:
            pass

        lower_text = text.lower()
        actions = []
        for key, aliases in TOOL_ALIASES.items():
            for alias in aliases:
                if alias in lower_text and key not in actions:
                    actions.append(key)
                    break
        return actions

    def decide(self) -> list[str]:
        """Use the LLM to select the best actions for dataset analysis."""
        if memory.df is None:
            return []

        available_actions = "\n".join(
            f"- {name}: {description}"
            for name, description in self.TOOL_DESCRIPTIONS.items()
        )
        prompt = f"""
            You are an intelligent data analysis assistant. Based on the dataset summary below, choose the most useful analysis actions for this dataset.

            Available actions:
            {available_actions}

            Dataset summary:
            {get_data_summary()}

            Return only a JSON array with the selected action names in order. Example:
            [\"inspect_data\", \"summary_statistics\", \"create_histograms\"]
            """
        decision_text = self.llm.invoke(prompt).content
        selected = self.parse_action_list(decision_text)

        baseline = ["inspect_data", "summary_statistics"]
        self.actions = baseline.copy()
        for action in selected:
            if action not in self.actions:
                self.actions.append(action)

        numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
        if numeric_cols and "create_histograms" not in self.actions:
            self.actions.append("create_histograms")
        if len(numeric_cols) >= 2 and "correlation_heatmap" not in self.actions:
            self.actions.append("correlation_heatmap")

        self.findings.append(f"LLM Decision: {', '.join(self.actions)}")
        return self.actions

    def act(self) -> list[str]:
        """Execute the selected analysis tools and collect their outputs."""
        if memory.df is None:
            return []

        if not self.actions:
            self.decide()

        results = []
        for action in self.actions:
            tool_obj = self.TOOL_MAP.get(action)
            if not tool_obj:
                error_msg = f"Unknown action: {action}"
                self.findings.append(error_msg)
                results.append(error_msg)
                continue

            result = tool_obj()
            self.findings.append(result)
            results.append(result)

        return results

    def summarize(self) -> str:
        """Generate a compact summary from the collected analysis findings."""
        if not self.findings:
            return "No findings to summarize."

        findings_text = "\n\n".join(str(item) for item in self.findings)
        prompt = f"""
            You are an analytical assistant. Summarize the following CSV data analysis findings clearly and concisely.

            Dataset summary:
            {get_data_summary()}

            Findings:
            {findings_text}

            Include:
            1. Key data characteristics,
            2. Important patterns or anomalies,
            3. Any missing data or quality issues,
            4. Recommendations for follow-up analysis.

            Output a short list with bullet points.
            """
        summary_text = self.llm.invoke(prompt).content
        summary_lines = [
            "📊 Analysis Summary:",
            "=" * 50,
            summary_text.strip(),
            "=" * 50,
        ]
        if memory.plots:
            summary_lines.append(f"\n📈 Visualizations generated: {', '.join(memory.plots)}")

        return "\n".join(summary_lines)

    def follow_up(self, question: str) -> str:
        """Answer a follow-up question using existing findings and optional additional tool analysis."""
        if memory.df is None:
            return "No data has been loaded. Please load a CSV first."

        findings_text = "\n\n".join(str(item) for item in self.findings)
        available_actions = "\n".join(
            f"- {name}: {description}"
            for name, description in self.TOOL_DESCRIPTIONS.items()
        )
        prompt = f"""
            You are a helpful data analysis assistant. A user has asked the follow-up question below about the dataset and the previous analysis findings.

            Previous analysis findings:
            {findings_text}

            Follow-up question:
            {question}

            Decide whether you need additional analysis tools to answer the question accurately. Available tool actions are:
            {available_actions}

            If further analysis is needed, return only a JSON array of action names to execute in order. If no additional tools are required, return [] only.
            """
        action_text = self.llm.invoke(prompt).content
        actions = self.parse_action_list(action_text)

        extra_results = []
        if actions:
            for action in actions:
                tool_obj = self.TOOL_MAP.get(action)
                if not tool_obj:
                    continue
                result = tool_obj()
                self.findings.append(result)
                extra_results.append(f"[{action}] {result}")

        extra_results_text = "\n\n".join(extra_results) if extra_results else "None"
        answer_prompt = f"""
            You are a helpful data analysis assistant. Use the dataset summary, the previous analysis findings, and the results of any additional tool execution below to answer the user's follow-up question.

            Dataset summary:
            {get_data_summary()}

            Previous findings:
            {findings_text}

            Additional analysis results:
            {extra_results_text}

            Follow-up question:
            {question}

            Provide a clear, concise answer, and mention if more analysis is recommended.
            """
        answer_text = self.llm.invoke(answer_prompt).content
        self.findings.append(f"Follow-up: {question}")
        self.findings.append(answer_text)
        return answer_text

    def run(self, csv_path: str) -> str:
        """Execute the full perceive-decide-act-summarize cycle."""
        print("🤖 Starting CSV analysis agent...\n")
        print("📥 Perceiving data...")
        self.perceive(csv_path)

        print("🧠 Deciding which analysis steps to run...")
        self.decide()

        print("⚙️  Executing analysis tools...\n")
        self.act()

        print("✍️  Generating summary...\n")
        return self.summarize()
