import ast
import re


def parse_to_list(val):
    """Gold answer field (list or string) to a Python list of strings."""
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if not val:
        return []
    val_str = str(val).strip()
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in val_str.split(',') if x.strip()]


def normalize_entity(name):
    """Lowercase, drop type suffixes like '(disease)', unify quotes/punctuation."""
    s = str(name).lower().strip()
    s = s.replace('’', "'").replace('‘', "'")
    s = re.sub(r"\((disease|drug|gene|protein|gene/protein)\)", " ", s)
    s = re.sub(r"'s\b", "", s)
    # Drop entity-type prefixes copied from the context ('Disease type 2 diabetes mellitus')
    s = re.sub(r"^(disease|drug|compound|gene|protein)\s+(?=\S)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def parse_model_output(output):
    """Split a model answer into individual predicted entities."""
    text = str(output or "")
    if text.startswith("Error:"):
        return []
    # Prefer semicolons / new lines: gold names can contain commas
    if re.search(r"[;\n]", text):
        items = re.split(r"[;\n]+", text)
    else:
        items = re.split(r",", text)
    preds = []
    for item in items:
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", item)  # bullets / numbering
        item = item.strip().strip('.').strip()
        if item:
            preds.append(item)
    return preds


def score_item(model_output, gold_answers):
    """
    Exact match after normalization (no substring matching, which gives false hits
    such as 'gout' inside 'gouty arthritis').
    model_output is either the parsed answer list or a raw text answer.
    Returns hit, precision, recall, counts and the matched / missed gold names.
    """
    gold = {}
    for g in parse_to_list(gold_answers):
        gold.setdefault(normalize_entity(g), g)
    gold.pop("", None)

    raw_preds = model_output if isinstance(model_output, list) else parse_model_output(model_output)
    preds = []
    for p in raw_preds:
        n = normalize_entity(p)
        if n and n not in preds:
            preds.append(n)

    matched = [p for p in preds if p in gold]
    return {
        "hit": bool(matched),
        "precision": len(matched) / len(preds) if preds else 0.0,
        "recall": len(matched) / len(gold) if gold else 0.0,
        "n_pred": len(preds),
        "n_gold": len(gold),
        "n_matched": len(matched),
        "matched": [gold[m] for m in matched],
        "missed": [g for n, g in gold.items() if n not in matched],
    }


def gold_in_context(spoke_result, context, gold_answers):
    """Gold answers that SPOKE returned as targets and that are in the context given to the model."""
    if not spoke_result:
        return []
    names = {normalize_entity(t) for b in spoke_result.get("bridges", [])
             for t in b.get("targets", []) if t and t in context}
    return [g for g in parse_to_list(gold_answers) if normalize_entity(g) in names]


def load_results(path):
    """Result records from a result file (a list, or {'run_info': ..., 'results': [...]})."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("results", []) if isinstance(data, dict) else data


def score_results(data_list):
    """Aggregate metrics (percent) over a list of result records."""
    scored = []
    with_context = []
    empty_context = 0
    errors = 0
    truncated_outputs = 0
    gold_in_ctx = []
    statuses = {}
    for item in data_list:
        gold = item.get("correct_answer") or item.get("normalized_target_answer") or []
        if not parse_to_list(gold):
            continue
        has_context = bool(str(item.get("retrieved_context", "")).strip())
        if not has_context:
            empty_context += 1
        if str(item.get("model_output", "")).startswith("Error:"):
            errors += 1
        if item.get("finish_reason") == "length":
            truncated_outputs += 1
        if "retrieval_status" in item:
            statuses[item["retrieval_status"]] = statuses.get(item["retrieval_status"], 0) + 1
        if "gold_in_context" in item:
            gold_in_ctx.append(bool(item["gold_in_context"]))
        prediction = item["model_output_list"] if isinstance(item.get("model_output_list"), list) else item.get("model_output", "")
        result = score_item(prediction, gold)
        scored.append(result)
        if has_context:
            with_context.append(result)

    n = len(scored)
    if n == 0:
        return {"n_scored": 0, "hit_rate_percent": 0.0, "precision_percent": 0.0,
                "recall_percent": 0.0, "avg_predictions": 0.0, "hit_rate_with_context_percent": 0.0,
                "gold_in_context_percent": None, "empty_context": empty_context, "llm_errors": errors,
                "outputs_cut_at_token_limit": truncated_outputs}

    def pct(key):
        return round(100 * sum(float(s[key]) for s in scored) / n, 2)

    summary = {
        "n_scored": n,
        "hit_rate_percent": pct("hit"),
        "precision_percent": pct("precision"),
        "recall_percent": pct("recall"),
        "avg_predictions": round(sum(s["n_pred"] for s in scored) / n, 2),
        # Hit rate restricted to questions where SPOKE returned context
        "hit_rate_with_context_percent": round(100 * sum(s["hit"] for s in with_context) / len(with_context), 2) if with_context else 0.0,
        # Questions where SPOKE returned at least one gold answer (upper bound of the hit rate)
        "gold_in_context_percent": round(100 * sum(gold_in_ctx) / len(gold_in_ctx), 2) if gold_in_ctx else None,
        "empty_context": empty_context,
        "llm_errors": errors,
        "outputs_cut_at_token_limit": truncated_outputs,
    }
    for status, count in sorted(statuses.items()):
        summary[f"retrieval_{status}"] = count
    return summary
