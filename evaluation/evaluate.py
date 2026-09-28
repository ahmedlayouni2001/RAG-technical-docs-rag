"""
Phase 4: Evaluation harness.

Runs the REAL compiled LangGraph agent (src/agent/graph.py) against the golden
question set, capturing both its retrieved context and final answer for each
question, then scores the results with RAGAS.

Requires OPENAI_API_KEY — both your agent and RAGAS's own judge model make
real API calls. Nothing here has been run yet (no key available while
building this) — syntax-checked only. RAGAS's API has shifted across
versions; if anything below doesn't match, run `pip show ragas` and check
the current docs for the exact class/method names in your installed version.

Usage:
    python -m evaluation.evaluate
"""

import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,NoiseSensitivity

from src.agent.graph import app as rag_agent

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.csv"


def run_agent_on_question(question: str) -> dict:
    """Run the real agent end to end, capturing what it retrieved AND what it answered."""
    result = rag_agent.invoke({"question": question, "retries": 0})
    retrieved_contexts = [doc.page_content for doc in result["documents"]]
    return {
        "answer": result["generation"],
        "contexts": retrieved_contexts,
    }


def build_evaluation_dataset(golden_set: list[dict]) -> EvaluationDataset:
    rows = []
    for i, item in enumerate(golden_set, 1):
        print(f"  [{i}/{len(golden_set)}] Running agent on: {item['question'][:60]}...")
        agent_output = run_agent_on_question(item["question"])
        rows.append(
            {
                "user_input": item["question"],
                "response": agent_output["answer"],
                "retrieved_contexts": agent_output["contexts"],
                "reference": item["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(rows)


def main():
    golden_set = json.loads(GOLDEN_SET_PATH.read_text())
    print(f"Loaded {len(golden_set)} golden questions.\n")

    print("Step 1: running the real agent on every question (this hits the LLM + retriever)...")
    dataset = build_evaluation_dataset(golden_set)

    print("\nStep 2: scoring with RAGAS (this runs a separate judge-model LLM call per metric)...")
    results =evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ContextPrecision(), ContextRecall(), AnswerRelevancy(),NoiseSensitivity()],
    )

    print("\n=== Aggregate results ===")
    print(results)

    results_df = results.to_pandas()
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved per-question breakdown to {RESULTS_PATH}")

    # A quick honest sanity check worth glancing at manually: the 4 "out_of_scope"
    # questions in the golden set SHOULD show low context_precision/recall (nothing
    # relevant exists to retrieve) but ideally still reasonably high faithfulness
    # (the agent admitting "not covered" rather than confidently hallucinating).
    print("\nWorth checking manually: do the out_of_scope questions show the agent")
    print("admitting it doesn't know, rather than confidently making something up?")


if __name__ == "__main__":
    main()