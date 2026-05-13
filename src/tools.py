import pandas as pd
import matplotlib.pyplot as plt
from src.memory import DataAnalysisMemory

# Global memory instance for data analysis
memory = DataAnalysisMemory()


def load_csv(csv_path: str) -> str:
    """Load a CSV file and report its shape and columns."""
    try:
        memory.csv_path = csv_path
        memory.df = pd.read_csv(csv_path)
        rows, cols = memory.df.shape
        return f"✅ Loaded {csv_path}\n   📊 Shape: {rows} rows × {cols} columns\n   📋 Columns: {', '.join(memory.df.columns)}"
    except Exception as e:
        return f"❌ Error loading CSV: {str(e)}"


def inspect_data() -> str:
    """Inspect data types, missing values, numeric columns, and categorical columns."""
    if memory.df is None:
        return "❌ No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    missing = memory.df.isnull().sum()
    missing = missing[missing > 0].to_dict()

    result = f"🔢 Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'None'}\n"
    result += f"📝 Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}\n"
    result += f"❓ Missing values: {missing if missing else 'None'}"

    memory.findings.append("Data inspection completed")
    return result


def missing_value_report() -> str:
    """Report missing values and their percentages."""
    if memory.df is None:
        return "❌ No CSV has been loaded yet."

    missing = memory.df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "✅ No missing values found."

    missing_pct = (missing / len(memory.df) * 100).round(2)
    rows = [f"{col}: {count} missing ({missing_pct[col]}%)" for col, count in missing.items()]

    memory.findings.append("Missing value report generated")
    return "❗ Missing Values Report:\n" + "\n".join(rows)


def summary_statistics() -> str:
    """Generate summary statistics for numeric columns."""
    if memory.df is None:
        return "❌ No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return "❌ No numeric columns found for statistics."

    stats = memory.df[numeric_cols].describe()
    result = "📊 Summary Statistics:\n" + str(stats)

    memory.findings.append("Summary statistics generated")
    return result


def category_counts() -> str:
    """Display top category counts for categorical columns."""
    if memory.df is None:
        return "❌ No CSV has been loaded yet."

    categorical_cols = memory.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not categorical_cols:
        return "❌ No categorical columns found."

    outputs = []
    for col in categorical_cols:
        counts = memory.df[col].value_counts().head(10).to_string()
        outputs.append(f"Top values for {col}:\n{counts}")

    memory.findings.append("Category counts generated")
    return "\n\n".join(outputs)


def create_histograms() -> str:
    """Create histogram plots for numeric columns."""
    if memory.df is None:
        return "❌ No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return "❌ No numeric columns found for plotting."

    plots_created = []
    for col in numeric_cols[:5]:  # Limit to first 5 columns to avoid too many plots
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
        return "❌ No CSV has been loaded yet."

    numeric_cols = memory.df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return "❌ Need at least two numeric columns for correlation analysis."

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
    return f"📉 Correlation heatmap saved as {filename}"


# Keep tools list for backward compatibility (though not used in non-LLM version)
tools = [load_csv, inspect_data, missing_value_report, summary_statistics, category_counts, create_histograms, correlation_heatmap]