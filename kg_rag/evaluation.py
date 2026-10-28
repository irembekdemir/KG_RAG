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
    Returns dict with hit, precision, recall and counts.
    """
    gold = {normalize_entity(g) for g in parse_to_list(gold_answers)}
    gold.discard("")
    preds = []
    for p in parse_model_output(model_output):
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
    }


def score_results(data_list):
    """Aggregate metrics (percent) over a list of result records."""
    scored = []
    with_context = []
    empty_context = 0
    errors = 0
    for item in data_list:
        gold = item.get("correct_answer") or item.get("normalized_target_answer") or []
        if not parse_to_list(gold):
            continue
        has_context = bool(str(item.get("retrieved_context", "")).strip())
        if not has_context:
            empty_context += 1
        if str(item.get("model_output", "")).startswith("Error:"):
            errors += 1
        result = score_item(item.get("model_output", ""), gold)
        scored.append(result)
        if has_context:
            with_context.append(result)

    n = len(scored)
    if n == 0:
        return {"n_scored": 0, "hit_rate_percent": 0.0, "precision_percent": 0.0,
                "recall_percent": 0.0, "avg_predictions": 0.0, "hit_rate_with_context_percent": 0.0,
                "empty_context": empty_context, "llm_errors": errors}

    def pct(key):
        return round(100 * sum(float(s[key]) for s in scored) / n, 2)

    return {
        "n_scored": n,
        "hit_rate_percent": pct("hit"),
        "precision_percent": pct("precision"),
        "recall_percent": pct("recall"),
        "avg_predictions": round(sum(s["n_pred"] for s in scored) / n, 2),
        # Hit rate restricted to questions where SPOKE returned context
        "hit_rate_with_context_percent": round(100 * sum(s["hit"] for s in with_context) / len(with_context), 2) if with_context else 0.0,
        "empty_context": empty_context,
        "llm_errors": errors,
    }
