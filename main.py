import os
from src.agent import DataAnalysisAgent


def main():
    csv_path = input("Enter CSV file path: ")

    if not os.path.exists(csv_path):
        print(f"❌ Error: File '{csv_path}' not found.")
        return

    agent = DataAnalysisAgent()

    try:
        print(f"\n🔍 Starting agent for: {csv_path}")
        print("=" * 70)

        summary = agent.run(csv_path)

        print(summary)

        while True:
            print("\nYou can now ask a follow-up question, or press Enter to exit.")
            question = input("Follow-up question: ").strip()
            if not question:
                break
            answer = agent.follow_up(question)
            print("\n" + answer)

        print("\n✅ Agent session complete.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()