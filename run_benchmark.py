import os
import sys
import glob
import json
import time
import csv
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kg_rag.evaluation import score_results, load_results

def run_single_dataset(benchmark_input_path, output_dir, models, api=True, reasoning_effort=None, rescore_only=False, skip_existing=False, base_dir=None, context_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = os.path.basename(benchmark_input_path)
    seed_prefix = os.path.splitext(dataset_name)[0]
    
    print(f"\n" + "#" * 70)
    print(f"DATA SET PROCESSESSING: {dataset_name}")
    print("#" * 70)

    summary_stats = []

    for model_id in models:
        safe_model_name = model_id.replace('/', '_').replace(':', '_')
        output_file_path = os.path.join(output_dir, f"{seed_prefix}_{safe_model_name}.json")

        start_time = time.time()
        # An already finished file is not sent to the LLM again; it is only re-scored
        already_done = skip_existing and not rescore_only and os.path.exists(output_file_path)
        if already_done:
            print(f"[SKIP] {output_file_path} already exists, not calling the LLM again.")

        try:
            if rescore_only or already_done:
                if not os.path.exists(output_file_path):
                    raise FileNotFoundError(f"No existing result file: {output_file_path}")
            elif base_dir and context_dir:
                from kg_rag.rag_based_generation.GPT.text_generation import run_with_reference_contexts
                reference_files = sorted(glob.glob(os.path.join(context_dir, f"{seed_prefix}_*.json")))
                if not reference_files:
                    raise FileNotFoundError(f"No reference file for {seed_prefix} in {context_dir}")
                run_with_reference_contexts(
                    base_json_file=os.path.join(base_dir, f"{seed_prefix}_{safe_model_name}.json"),
                    reference_json_file=reference_files[0],
                    output_json_file=output_file_path,
                    model_id=model_id,
                    reasoning_effort=reasoning_effort
                )
            elif base_dir:
                from kg_rag.rag_based_generation.GPT.text_generation import run_own_knowledge_fallback
                run_own_knowledge_fallback(
                    base_json_file=os.path.join(base_dir, f"{seed_prefix}_{safe_model_name}.json"),
                    output_json_file=output_file_path,
                    model_id=model_id,
                    reasoning_effort=reasoning_effort
                )
            else:
                from kg_rag.rag_based_generation.GPT.text_generation import run_text_generation
                run_text_generation(
                    input_questions_file=benchmark_input_path,
                    output_json_file=output_file_path,
                    model_id=model_id,
                    api=api,
                    reasoning_effort=reasoning_effort
                )
            elapsed_time = time.time() - start_time

            total_input_tokens = 0
            total_output_tokens = 0
            total_reasoning_tokens = 0
            total_tokens = 0
            question_count = 0
            metrics = score_results([])
            run_effort = reasoning_effort or "default"

            if os.path.exists(output_file_path):
                data_list = load_results(output_file_path)
                question_count = len(data_list)

                for item in data_list:
                    total_input_tokens += item.get("input_tokens") or item.get("prompt_tokens") or 0
                    total_output_tokens += item.get("output_tokens") or item.get("completion_tokens") or 0
                    total_reasoning_tokens += item.get("reasoning_tokens") or 0
                    total_tokens += item.get("total_tokens") or (
                        (item.get("input_tokens") or 0) + (item.get("output_tokens") or 0))

                metrics = score_results(data_list)
                # New result files keep the reasoning setting once in run_info, older ones per record
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                efforts = {raw["run_info"].get("reasoning_effort")} if isinstance(raw, dict) else \
                    {item.get("reasoning_effort") for item in data_list if item.get("reasoning_effort")}
                efforts.discard(None)
                if len(efforts) == 1:
                    run_effort = efforts.pop()

            summary_stats.append({
                "dataset": dataset_name,
                "model_id": model_id,
                "reasoning_effort": run_effort,
                "status": "Rescored" if rescore_only else ("Existing" if already_done else "Success"),
                "question_count": question_count,
                **metrics,
                "elapsed_time_sec": None if (rescore_only or already_done) else round(elapsed_time, 2),
                "avg_time_per_q_sec": None if (rescore_only or already_done) else round(elapsed_time / max(question_count, 1), 2),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_tokens
            })

        except Exception as e:
            print(f"[ERROR] {model_id} during ({dataset_name}): {e}")
            summary_stats.append({
                "dataset": dataset_name,
                "model_id": model_id,
                "reasoning_effort": reasoning_effort or "default",
                "status": f"Failed: {str(e)}",
                "question_count": 0,
                **score_results([]),
                "elapsed_time_sec": 0,
                "avg_time_per_q_sec": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_reasoning_tokens": 0,
                "total_tokens": 0
            })

    return summary_stats

def _summary_key(row):
    return (row.get("dataset"), row.get("model_id"), row.get("reasoning_effort"))

def merge_summaries(old_rows, new_rows):
    """
    Combine an existing summary with this run's rows instead of overwriting it.
    A row from this run replaces the old row for the same (dataset, model, reasoning),
    except that a skipped ('Existing') or failed row never replaces an earlier
    successful one, so timings from the original run are kept.
    """
    merged = {_summary_key(r): r for r in old_rows}
    for row in new_rows:
        key = _summary_key(row)
        old = merged.get(key)
        keep_old = old is not None and old.get("status") == "Success" and (
            row.get("status") == "Existing" or str(row.get("status", "")).startswith("Failed")
        )
        if keep_old:
            if str(row.get("status", "")).startswith("Failed"):
                print(f"[WARN] {key} failed in this run; keeping the earlier successful summary row.")
            continue
        merged[key] = row
    return sorted(merged.values(), key=lambda r: tuple(str(x) for x in _summary_key(r)))

def save_summary_to_csv(summaries, csv_path):
    if not summaries:
        return

    fieldnames = []
    for row in summaries:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

def main():
    parser = argparse.ArgumentParser(description="KG-RAG Benchmark Runner")
    parser.add_argument("-i", "--input", type=str, default="./benchmark_dataset")
    parser.add_argument("-o", "--output_dir", type=str, default="./results/benchmark_runs")
    parser.add_argument("-g", "--models", nargs="+", default=["gpt-5.4-mini"])
    parser.add_argument("-r", "--reasoning_effort", type=str, default=None,
                        help="e.g. none/low/medium/high; omitted = API default")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Do not call the LLM for files that already have a result file in --output_dir")
    parser.add_argument("--rescore", action="store_true",
                        help="Do not call the LLM; recompute metrics from existing result files in --output_dir")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="v4 result folder: re-ask only its empty-answer questions, allowing the model's own knowledge")
    parser.add_argument("--context_dir", type=str, default=None,
                        help="With --base_dir: re-ask (v4 instruction) only the questions whose context differs from "
                             "this result folder, using its contexts, so all runs share one SPOKE snapshot")
    
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path) and os.path.exists(os.path.join(PROJECT_ROOT, "benchmark_dataset")):
        input_path = os.path.join(PROJECT_ROOT, "benchmark_dataset")

    if os.path.isdir(input_path):
        input_files = sorted(glob.glob(os.path.join(input_path, "*.json")) + glob.glob(os.path.join(input_path, "*.jsonl")))
    elif os.path.isfile(input_path):
        input_files = [input_path]
    else:
        print(f"[ERROR] No input path: {input_path}")
        return

    print(f"[{time.strftime('%H:%M:%S')}] Benchmark starting...")
    print(f"Number of seed files found: {len(input_files)}")

    all_summaries = []

    for file_path in input_files:
        summary = run_single_dataset(
            benchmark_input_path=file_path,
            output_dir=args.output_dir,
            models=args.models,
            reasoning_effort=args.reasoning_effort,
            rescore_only=args.rescore,
            skip_existing=args.skip_existing,
            base_dir=args.base_dir,
            context_dir=args.context_dir
        )
        all_summaries.extend(summary)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # One summary per model set / mode so different runs do not overwrite each other
    tag = "_".join(m.replace('/', '_').replace(':', '_') for m in args.models)
    if args.rescore:
        tag += "_rescored"
    json_summary_path = os.path.join(args.output_dir, f"overall_benchmark_summary_{tag}.json")

    # Every run's rows are also appended to a history file, so nothing is ever lost
    run_stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    history_path = os.path.join(args.output_dir, f"summary_history_{tag}.jsonl")
    with open(history_path, 'a', encoding='utf-8') as f:
        for row in all_summaries:
            f.write(json.dumps({"run_started": run_stamp, **row}, ensure_ascii=False) + "\n")

    # Merge with the existing summary instead of overwriting it
    if os.path.exists(json_summary_path):
        with open(json_summary_path, 'r', encoding='utf-8') as f:
            old_rows = json.load(f)
        all_summaries = merge_summaries(old_rows, all_summaries)

    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    csv_summary_path = os.path.join(args.output_dir, f"overall_benchmark_summary_{tag}.csv")
    save_summary_to_csv(all_summaries, csv_summary_path)

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print(f"JSON Summary: {json_summary_path}")
    print(f"CSV Summary:  {csv_summary_path}")
    print(f"Run history:  {history_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()