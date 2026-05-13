from src.tools import (
    memory,
    load_csv,
    inspect_data,
    missing_value_report,
    summary_statistics,
    category_counts,
    create_histograms,
    correlation_heatmap,
)


class DataAnalysisAgent:
    """A simple local agent that chooses CSV analysis steps without APIs."""

    def __init__(self):
        self.actions = []
        self.findings = []

    def perceive(self, csv_path: str) -> str:
        """Perceive input by loading the CSV file into memory."""
        result = load_csv(csv_path)
        self.findings.append(result)
        return result

    def decide(self) -> list:
        """Decide which steps to run based on the loaded dataset."""
        if memory.df is None:
            return []

        numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
        categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        missing = memory.df.isnull().sum()

        actions = ["inspect_data"]
        if missing.any():
            actions.append("missing_value_report")
        if numeric_cols:
            actions.append("summary_statistics")
        if categorical_cols:
            actions.append("category_counts")
        if numeric_cols:
            actions.append("create_histograms")
        if len(numeric_cols) >= 2:
            actions.append("correlation_heatmap")

        self.actions = actions
        return actions

    def act(self) -> list:
        """Execute the chosen analysis steps and save the results."""
        results = []
        for action in self.actions:
            if action == "inspect_data":
                results.append(inspect_data())
            elif action == "missing_value_report":
                results.append(missing_value_report())
            elif action == "summary_statistics":
                results.append(summary_statistics())
            elif action == "category_counts":
                results.append(category_counts())
            elif action == "create_histograms":
                results.append(create_histograms())
            elif action == "correlation_heatmap":
                results.append(correlation_heatmap())
        self.findings.extend(results)
        return results

    def summarize(self) -> str:
        """Summarize the agent’s findings after action execution."""
        summary_lines = ["Agent Summary:"]
        summary_lines.extend(self.findings)
        if memory.plots:
            summary_lines.append(f"Plots generated: {', '.join(memory.plots)}")
        return "\n\n".join(summary_lines)

    def run(self, csv_path: str) -> str:
        """Run the full agent cycle: perceive, decide, act, summarize."""
        self.perceive(csv_path)
        self.decide()
        self.act()
        return self.summarize()