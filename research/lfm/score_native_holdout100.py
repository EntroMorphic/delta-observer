#!/usr/bin/env python3
"""Score the locked 100-task comparison without fitting thresholds to outputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from research.lfm.evaluate_pretrained_development import content_validation, strict_json, validate


def format_pass(task, response):
    text = response.strip()
    if task["validator"] == "json_exact":
        try:
            strict_json(text)
            return True
        except (ValueError, TypeError):
            return False
    if task["category"] == "arithmetic":
        return re.fullmatch(r"-?\d+", text) is not None
    if task["category"] == "transitive_ordering":
        return text in re.findall(r"\b[RST]\d+\b", task["prompt"])
    if task["category"] == "spanish_number_words":
        return text.isalpha() and text == text.lower()
    raise ValueError("Unsupported format contract")


def score(tasks, rows):
    if len(tasks) != 100 or len({task['id'] for task in tasks}) != 100:
        raise ValueError("Expected 100 distinct task IDs")
    if len(rows) != 100:
        raise ValueError("Expected exactly 100 result rows; duplicates are forbidden")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != 100 or set(by_id) != {task["id"] for task in tasks}:
        raise ValueError("Missing, duplicate, or unexpected task IDs")
    results = []
    for task in tasks:
        row = by_id[task["id"]]
        if row["prompt"] != task["prompt"]:
            raise ValueError("Prompt differs from frozen task")
        strict = validate(task, row["response"])["pass"]
        content = content_validation(task, row["response"])["pass"]
        if strict:
            content = True
        results.append({"id": task["id"], "category": task["category"],
            "strict": strict, "format": format_pass(task, row["response"]),
            "content": content,
            "truncated": row["hit_token_limit_without_eos"]})
    categories = {}
    for category in sorted({task["category"] for task in tasks}):
        selected = [row for row in results if row["category"] == category]
        categories[category] = counts(selected)
    return {**counts(results), "categories": categories, "tasks": results}


def counts(rows):
    return {"total": len(rows), "strict_successes": sum(row["strict"] for row in rows),
        "format_successes": sum(row["format"] for row in rows),
        "content_confirmed": sum(row["content"] is True for row in rows),
        "content_parser_failures": sum(row["content"] is False for row in rows),
        "content_unresolved": sum(row["content"] is None for row in rows),
        "truncated": sum(row["truncated"] for row in rows)}


def compare(tasks_path, freeze_path, paths, output,
            protocol="research/reports/native_holdout100_protocol.md"):
    task_bytes = tasks_path.read_bytes()
    digest = hashlib.sha256(task_bytes).hexdigest()
    freeze = json.loads(freeze_path.read_text())
    if digest != freeze["tasks_sha256"]:
        raise ValueError("Task file changed after freeze")
    tasks = json.loads(task_bytes)["tasks"]
    if len(tasks) != 100:
        raise ValueError("Expected exactly 100 tasks")
    artifacts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline = artifacts["baseline"]
    for name, artifact in artifacts.items():
        if not artifact["complete"] or artifact["tasks_sha256"] != digest:
            raise ValueError(f"Incomplete or unmatched evaluation: {name}")
        for key in ("model", "revision", "dtype", "generation_config"):
            if artifact[key] != baseline[key]:
                raise ValueError(f"Evaluation settings differ: {key}")
    result = {"protocol": protocol,
        "tasks_sha256": digest,
        "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for name, path in paths.items()},
        "arms": {name: score(tasks, artifact["results"]) for name, artifact in artifacts.items()}}
    base_rows = {row["id"]: row for row in result["arms"]["baseline"]["tasks"]}
    for name in paths:
        if name == "baseline":
            continue
        arm = result["arms"][name]
        arm["gains"] = [row["id"] for row in arm["tasks"] if row["strict"] and not base_rows[row["id"]]["strict"]]
        arm["regressions"] = [row["id"] for row in arm["tasks"] if not row["strict"] and base_rows[row["id"]]["strict"]]
        arm["net_change"] = len(arm["gains"]) - len(arm["regressions"])
        arm["content_gains"] = [row["id"] for row in arm["tasks"]
            if row["content"] is True and base_rows[row["id"]]["content"] is not True]
        arm["lost_confirmed_content"] = [row["id"] for row in arm["tasks"]
            if row["content"] is not True and base_rows[row["id"]]["content"] is True]
        for category, totals in arm["categories"].items():
            totals["net_change"] = totals["strict_successes"] - result["arms"]["baseline"]["categories"][category]["strict_successes"]
    with output.open("x") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps({name: {key: value for key, value in arm.items() if key != "tasks"}
                      for name, arm in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--protocol", default="research/reports/native_holdout100_protocol.md")
    for name in ("baseline", "repeated", "output"):
        parser.add_argument(f"--{name}", required=True)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--unique")
    candidate.add_argument("--remediation")
    args = parser.parse_args()
    names = ("baseline", "repeated", "remediation" if args.remediation else "unique")
    compare(Path(args.tasks), Path(args.freeze),
            {name: Path(getattr(args, name)) for name in names},
            Path(args.output), args.protocol)
