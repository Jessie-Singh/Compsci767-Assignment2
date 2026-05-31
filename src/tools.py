import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from src.memory import DataAnalysisMemory

# Global memory instance for data analysis
memory = DataAnalysisMemory()


def load_csv(csv_path: str) -> str:
    """Load a CSV file and report its shape and columns."""
    try:
        memory.csv_path = csv_path
        memory.df = pd.read_csv(csv_path)
        rows, cols = memory.df.shape
        return f"Loaded {csv_path}\n   Shape: {rows} rows × {cols} columns\n   Columns: {', '.join(memory.df.columns)}"
    except Exception as e:
        return f"Error loading CSV: {str(e)}"


def inspect_data() -> str:
    """Inspect data types, missing values, numeric columns, and categorical columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing = memory.df.isnull().sum()
    missing_details = missing[missing > 0].to_dict()

    result = [
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}",
        f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}",
        f"Missing values: {missing_details if missing_details else 'None'}",
    ]

    memory.findings.append("Data inspection completed")
    return "\n".join(result)


def missing_value_report() -> str:
    """Report missing values and their percentages."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    missing = memory.df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "No missing values found."

    missing_pct = (missing / len(memory.df) * 100).round(2)
    rows = [f"{col}: {count} missing ({missing_pct[col]}%)" for col, count in missing.items()]

    memory.findings.append("Missing value report generated")
    return "❗ Missing Values Report:\n" + "\n".join(rows)


def summary_statistics() -> str:
    """Generate summary statistics for numeric columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return "No numeric columns found for statistics."

    stats = memory.df[numeric_cols].describe()
    result = "📊 Summary Statistics:\n" + str(stats)

    memory.findings.append("Summary statistics generated")
    return result


def category_counts() -> str:
    """Display top category counts for categorical columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not categorical_cols:
        return "No categorical columns found."

    outputs = []
    for col in categorical_cols:
        counts = memory.df[col].value_counts().head(10).to_string()
        outputs.append(f"Top values for {col}:\n{counts}")

    memory.findings.append("Category counts generated")
    return "\n\n".join(outputs)


def dataset_context() -> str:
    """Provide dataset structure, missing values, and a sample of rows for inspection."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    cols = list(memory.df.columns)
    types = memory.df.dtypes.astype(str).to_dict()
    missing = memory.df.isnull().sum().to_dict()
    sample_rows = memory.df.head(5).to_dict(orient="records")

    lines = [
        "Dataset columns and types:",
        "  " + ", ".join(f"{col} ({types[col]})" for col in cols),
        "Missing values by column:",
        "  " + ", ".join(f"{col}: {int(missing[col])}" for col in cols if missing[col] > 0) if any(missing[col] > 0 for col in cols) else "  None",
        "Sample rows (first 5):",
    ]
    lines.extend(f"  {row}" for row in sample_rows)

    memory.findings.append("Dataset context generated")
    return "\n".join(lines)


def query_dataset(question: str) -> str:
    """Answer a natural language question by querying the loaded DataFrame."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    cols = list(memory.df.columns)
    sample_rows = memory.df.head(5).to_dict(orient="records")
    prompt = f"""
You are a Python data assistant. A pandas DataFrame named df is loaded with the following columns:
{cols}

Use only df and pd to answer the user's question.
Do not import any other modules.
Do not include markdown or explanation.
Assign the final answer to a variable named result.

Sample rows:
{sample_rows}

Question:
{question}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
        verbose=False,
    )

    response = llm.invoke(prompt).content.strip()
    code = response

    # Remove Markdown fences if the model returned code blocks.
    if code.startswith("```"):
        code_lines = code.splitlines()
        if code_lines[0].startswith("```"):
            code_lines = code_lines[1:]
        if code_lines and code_lines[-1].strip() == "```":
            code_lines = code_lines[:-1]
        code = "\n".join(code_lines)

    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "round": round,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "abs": abs,
        "any": any,
        "all": all,
    }
    exec_globals = {"df": memory.df, "pd": pd, "__builtins__": safe_builtins}
    exec_locals = {}
    try:
        exec(code, exec_globals, exec_locals)
    except Exception as exc:
        return f"Error executing dataset query code: {exc}\nCode:\n{code}"

    if "result" not in exec_locals:
        return f"The dataset query did not assign a 'result' variable. Code:\n{code}"

    result = exec_locals["result"]
    memory.findings.append(f"Dataset query executed: {question}")
    return str(result)


def data_insights() -> str:
    """Generate high-level insights from the loaded dataset."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    rows, cols = memory.df.shape
    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing = memory.df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)

    insights = [f"Dataset contains {rows} rows and {cols} columns."]

    if not missing_cols.empty:
        top_missing_col = missing_cols.index[0]
        insights.append(
            f"{len(missing_cols)} column(s) contain missing values, with {int(missing_cols.iloc[0])} missing in '{top_missing_col}'."
        )
    else:
        insights.append("No missing values detected.")

    if numeric_cols:
        highest_variance = memory.df[numeric_cols].var().idxmax()
        insights.append(
            f"{len(numeric_cols)} numeric column(s) present. Column '{highest_variance}' has the highest variance."
        )
        if len(numeric_cols) > 1:
            corr_matrix = memory.df[numeric_cols].corr().abs()
            top_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for col2 in corr_matrix.columns[i + 1 :]:
                    top_pairs.append(((col1, col2), corr_matrix.loc[col1, col2]))
            top_pairs = sorted(top_pairs, key=lambda item: -item[1])[:2]
            insights.append(
                "Strongest numeric relationships: "
                + ", ".join(
                    f"{pair[0]} vs {pair[1]}={value:.2f}" for pair, value in top_pairs
                )
            )

    if categorical_cols:
        top_values = []
        for col in categorical_cols[:3]:
            value_counts = memory.df[col].value_counts().head(3)
            top_values.append(
                col + ": " + ", ".join(
                    f"{label} ({count})" for label, count in value_counts.items()
                )
            )
        insights.append("Top categories: " + "; ".join(top_values))

    memory.findings.append("High-level dataset insights generated")
    return "\n".join(insights)


def create_histograms() -> str:
    """Create histogram plots for numeric columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return "No numeric columns found for plotting."

    plots_created = []
    for col in numeric_cols[:5]:
        plt.figure(figsize=(8, 6))
        memory.df[col].hist(bins=30, edgecolor='black')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        filename = f"histogram_{col.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        plots_created.append(filename)

    memory.plots.extend(plots_created)
    return f"📈 Created {len(plots_created)} histogram plots: {', '.join(plots_created)}"


def correlation_heatmap() -> str:
    """Create a correlation heatmap for numeric columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return "Need at least two numeric columns for correlation analysis."

    corr = memory.df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    filename = "correlation_heatmap.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    memory.plots.append(filename)
    memory.findings.append("Correlation heatmap generated")
    return f"Correlation heatmap saved as {filename}"


def top_correlations() -> str:
    """Show the strongest numeric correlations in the dataset."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return "Need at least two numeric columns for correlation analysis."

    corr = memory.df[numeric_cols].corr().abs()
    pairs = []
    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1:]:
            pairs.append((corr.loc[col1, col2], col1, col2))
    pairs.sort(reverse=True)

    top_pairs = pairs[:5]
    summary = [f"{col1} ⟷ {col2}: correlation {corr_value:.2f}" for corr_value, col1, col2 in top_pairs]
    memory.findings.append("Top correlations identified")
    return "Top numeric correlations:\n" + "\n".join(summary)


def outlier_report() -> str:
    """Detect numeric outliers using the IQR method."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return "No numeric columns found for outlier analysis."

    details = []
    for col in numeric_cols:
        series = memory.df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        details.append(f"{col}: {len(outliers)} outliers ({len(outliers) / len(series) * 100:.2f}%)")

    memory.findings.append("Outlier report generated")
    return "Outlier detection summary:\n" + "\n".join(details)


def value_ranges() -> str:
    """Summarize min/max ranges for numeric columns and top categories for categorical columns."""
    if memory.df is None:
        return "No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    outputs = []
    if numeric_cols:
        outputs.append("Numeric range summary:")
        for col in numeric_cols:
            outputs.append(f"{col}: min={memory.df[col].min()}, max={memory.df[col].max()}")

    if categorical_cols:
        outputs.append("\nCategorical top values:")
        for col in categorical_cols:
            top = memory.df[col].value_counts().head(5).to_string()
            outputs.append(f"{col}:\n{top}")

    memory.findings.append("Value ranges summarized")
    return "\n".join(outputs)


TOOL_REGISTRY = {
    "load_csv": load_csv,
    "inspect_data": inspect_data,
    "missing_value_report": missing_value_report,
    "summary_statistics": summary_statistics,
    "category_counts": category_counts,
    "dataset_context": dataset_context,
    "query_dataset": query_dataset,
    "data_insights": data_insights,
    "create_histograms": create_histograms,
    "correlation_heatmap": correlation_heatmap,
    "top_correlations": top_correlations,
    "outlier_report": outlier_report,
    "value_ranges": value_ranges,
}

TOOL_ALIASES = {
    "load_csv": ["load_csv", "load csv", "read csv", "open csv"],
    "inspect_data": ["inspect_data", "inspect", "inspect data", "overview", "data overview"],
    "missing_value_report": ["missing_value_report", "missing values", "missing value report", "missing-value report"],
    "summary_statistics": ["summary_statistics", "summary stats", "summary statistics", "describe"],
    "category_counts": ["category_counts", "category counts", "category distribution", "category frequency"],
    "dataset_context": ["dataset_context", "data context", "dataset context", "preview data", "preview rows", "data preview", "raw data"],
    "query_dataset": ["query_dataset", "query data", "query dataset", "ask dataset", "ask data", "data question", "csv question", "query csv", "explore data"],
    "data_insights": ["data_insights", "data insights", "dataset insights", "insights", "insight summary", "data summary"],
    "create_histograms": ["create_histograms", "histogram", "histograms", "create histogram", "create histograms"],
    "correlation_heatmap": ["correlation_heatmap", "heatmap", "correlation heatmap", "corr heatmap"],
    "top_correlations": ["top_correlations", "top correlations", "strong correlations", "highest correlations"],
    "outlier_report": ["outlier_report", "outlier report", "outliers", "outlier detection"],
    "value_ranges": ["value_ranges", "value ranges", "value range", "min max", "range summary"],
}

# Keep tools list for backward compatibility (though not used in non-LLM version)
tools = list(TOOL_REGISTRY.values())
