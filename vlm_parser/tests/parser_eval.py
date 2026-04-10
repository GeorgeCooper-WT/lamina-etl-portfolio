"""
Evaluates the accuracy of the LLM parser output against a manually verified ground truth JSON.

Compares the latest parser output for a given project against a manually verified ground truth,
reporting field-level error counts and error percentage using recursive leaf-node diffing.

Usage:
    python tests/parser_eval.py <project> <ground_truth>

Arguments:
    project         Project folder name under db/
    ground_truth    Path to manually verified ground truth JSON

Requires: deepdiff
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from deepdiff import DeepDiff
import pprint
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_leaves(obj) -> int:
    """Recursively count the number of leaf nodes in a nested dict/list structure."""
    if isinstance(obj, dict):
        return sum(count_leaves(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_leaves(v) for v in obj)
    else:
        return 1


def count_diff_leaves(diff: DeepDiff) -> int:
    """
    Count every differing leaf field across all DeepDiff change types.
    Recursively counts into removed/added subtrees so a missing inverter
    (with 20+ nested fields) is counted as 20+ differences, not 1.
    """
    total = 0

    # Simple value changes — one diff per entry
    total += len(diff.get("values_changed", {}))
    total += len(diff.get("type_changes", {}))

    # Removed/added items — recurse into the subtree to count all leaves
    for removed in diff.get("iterable_item_removed", {}).values():
        total += count_leaves(removed)
    for added in diff.get("iterable_item_added", {}).values():
        total += count_leaves(added)
    for removed in diff.get("dictionary_item_removed", {}).values():
        total += count_leaves(removed)
    for added in diff.get("dictionary_item_added", {}).values():
        total += count_leaves(added)

    return total


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
def evaluate_llm_output(ground_truth_path: str, llm_output_path: str) -> None:
    """
    Compares the LLM's parsed JSON against the manually verified Ground Truth JSON.
    Provides a summary of error counts and error percentage.
    """
    # Load ground truth
    logger.info(f"Loading Ground Truth: {ground_truth_path}")
    try:
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in ground truth: {e}")
        sys.exit(1)

    gt_leaf_count = count_leaves(ground_truth)
    logger.info(f"Ground Truth leaf fields: {gt_leaf_count}")

    # Load LLM output
    logger.info(f"Loading LLM Output:   {llm_output_path}")
    try:
        with open(llm_output_path, "r") as f:
            llm_output = json.load(f)
    except FileNotFoundError:
        logger.error(f"LLM output file not found: {llm_output_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in LLM output: {e}")
        sys.exit(1)

    llm_leaf_count = count_leaves(llm_output)
    logger.info(f"LLM Output leaf fields: {llm_leaf_count}")

    # Run diff
    logger.info("\nRunning Diff Analysis...")
    print("-" * 40)

    diff = DeepDiff(ground_truth, llm_output, ignore_order=True)

    if not diff:
        logger.info(
            "✅ No discrepancies found. LLM output matches ground truth exactly."
        )
        return

    print("DISCREPANCIES FOUND:")
    pprint.pprint(diff.to_dict(), indent=2)

    # Calculate accurate error percentage using recursive leaf count
    diff_count = count_diff_leaves(diff)
    error_pct = 100 * diff_count / gt_leaf_count if gt_leaf_count > 0 else 0.0

    print(f"\nTotal differing leaf fields : {diff_count}")
    print(f"Total ground truth leaf fields: {gt_leaf_count}")
    print(f"Error percentage            : {error_pct:.2f}%")

    # Breakdown by diff type
    print("\nBreakdown by diff type:")
    for diff_type, items in diff.items():
        print(f"  {diff_type}: {len(items)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM parser output against a ground truth JSON."
    )
    parser.add_argument("project", help="Project folder name under db/")
    parser.add_argument("ground_truth", help="Path to ground truth JSON")
    args = parser.parse_args()

    _ROOT = Path(__file__).resolve().parent.parent
    output_files = sorted((_ROOT / "db" / args.project / "json").glob("*.json"))

    if not output_files:
        logger.error(f"No JSON output files found for project: {args.project}")
        sys.exit(1)

    evaluate_llm_output(args.ground_truth, str(output_files[-1]))


if __name__ == "__main__":
    main()
