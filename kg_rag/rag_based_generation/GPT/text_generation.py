import json
import os
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from kg_rag.utility import *
from kg_rag.evaluation import parse_to_list, score_item, gold_in_context

CHAT_MODEL_ID_DEFAULT = 'gpt-5.4-mini'
SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

# Names may contain commas (e.g. 'pregnancy loss, recurrent, susceptibility to'), so the
# answer is requested as a JSON list through OpenAI structured output (ANSWER_LIST_SCHEMA)
INSTRUCTION_VERSION = "v4-structured-output"
INSTRUCTION_TEMPLATE = (
    "\n\nInstruction: Answer the question using only the retrieved context. "
    "Put the names of ALL {target_type}s that answer the question in the \"answers\" list, "
    "copying each name exactly as it appears in the context, without entity-type prefixes. "
    "If the context is empty or does not contain the answer, return an empty list."
)

# Second pass: only questions that got an empty list from a v4 run are asked again,
# allowing the model's own knowledge
OWN_KNOWLEDGE_INSTRUCTION_VERSION = "v4-structured-output+own-knowledge-fallback"
OWN_KNOWLEDGE_INSTRUCTION_TEMPLATE = (
    "\n\nInstruction: Answer the question using the retrieved context. "
    "Put the names of ALL {target_type}s that answer the question in the \"answers\" list, "
    "copying each name exactly as it appears in the context, without entity-type prefixes. "
    "If the context is empty or does not contain the answer, answer from your own biomedical knowledge."
)


# Field order of a result record (answer_source right after model_output_list)
RECORD_FIELDS = [
    "question_id", "question", "seed_entity", "seed_type", "target_type", "options",
    "correct_answer", "retrieval_status", "retrieved_context", "spoke_result", "gold_in_context",
    "model_output", "model_output_list", "answer_source", "finish_reason", "is_hit",
    "precision", "recall", "matched", "input_tokens", "output_tokens", "reasoning_tokens",
    "total_time_sec",
]


def answer_source_for(context, answers):
    # An answer given although SPOKE returned no context can only come from the model's own knowledge
    return "own_knowledge" if answers and not str(context).strip() else "context"


def build_prompt(context, question, options, instruction):
    if options:
        return f"Context: {context}\n\nQuestion: {question}\nOptions: {options}{instruction}"
    return f"Context: {context}\n\nQuestion: {question}{instruction}"


def ask_llm(prompt, model_id, reasoning_effort):
    try:
        return get_GPT_list_response(
            prompt, SYSTEM_PROMPT, model_id, temperature=TEMPERATURE,
            reasoning_effort=reasoning_effort
        )
    except Exception as e:
        return {"raw_output": f"Error: {str(e)}", "answers": [], "finish_reason": None,
                "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}

def run_text_generation(
    input_questions_file,
    output_json_file,
    model_id=CHAT_MODEL_ID_DEFAULT,
    api=True,
    edge_evidence=False,
    reasoning_effort=None
):
    print(f"\n--- Benchmark Starting with {model_id}: {input_questions_file} ---")
    
    data = []
    if input_questions_file.endswith('.jsonl'):
        with open(input_questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(input_questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    embedding_model_name = (
        config_data.get('VECTOR_DB_SENTENCE_EMBEDDING_MODEL') or
        config_data.get('SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL') or
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_function = load_sentence_transformer(embedding_model_name)
    vectorstore = load_chroma(config_data['VECTOR_DB_PATH'], embedding_model_name)
    
    node_context_path = config_data['NODE_CONTEXT_PATH']
    node_context_df = pd.read_csv(node_context_path) if os.path.exists(node_context_path) else pd.DataFrame()

    results = []
    run_start_time = time.time()
    dataset_name = os.path.splitext(os.path.basename(input_questions_file))[0]

    for index, row in enumerate(tqdm(data, desc=f"Processing questions ({model_id})...")):
        # The answers are full lists, so ask the "Name all ..." variant. The 'prompt'
        # field asks for a single answer ("Just give me the answer"), which contradicts
        # the list instruction below and caps recall at 1/len(answers).
        question = (
            row.get('question') or row.get('hop2_question_multi') or
            row.get('hop2_question') or row.get('prompt') or
            row.get('hop1_question_multi') or row.get('hop1_question') or
            row.get('text') or row.get('Question')
        )

        if not question:
            continue

        question_start_time = time.time()
        options = row.get('options') or row.get('choices') or row.get('Options') or ""
        raw_correct_answer = row.get('answer') or row.get('correct_answer') or row.get('Correct_Answer') or ""
        correct_answer_list = parse_to_list(raw_correct_answer)

        target_type = row.get('target_type', 'disease')
        hop2_entity = row.get('hop2', '')
        hop1_entity = row.get('hop1', '')
        
        seed_entity = hop2_entity if hop2_entity else hop1_entity
        seed_type = row.get('hop2_type') if hop2_entity else row.get('hop1_type')

        # 1. Context 
        retrieval = retrieve_context_with_details(
            question=question,
            seed_entity=seed_entity,
            target_type=target_type,
            seed_type=seed_type,
            vectorstore=vectorstore,
            embedding_function=embedding_function,
            node_context_df=node_context_df,
            context_volume=config_data['CONTEXT_VOLUME'],
            context_sim_threshold=config_data['QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD'],
            context_sim_min_threshold=config_data['QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY'],
            edge_evidence=edge_evidence or config_data.get('EDGE_EVIDENCE', False),
            api=api
        )
        context = retrieval["context"]

        # 2. Instruction
        instruction = INSTRUCTION_TEMPLATE.format(target_type=target_type)
        full_prompt = build_prompt(context, question, options, instruction)

        # 3. LLM Call
        response = ask_llm(full_prompt, model_id, reasoning_effort)

        score = score_item(response["answers"], correct_answer_list)

        record = {
            "question_id": f"{dataset_name}#{index}",
            "question": question,
            "seed_entity": seed_entity,
            "seed_type": seed_type,
            "target_type": target_type,
        }
        if options:
            record["options"] = str(options)
        record.update({
            "correct_answer": correct_answer_list,
            "retrieval_status": retrieval["retrieval_status"],
            "retrieved_context": context,
            "spoke_result": retrieval["spoke_result"],
            "gold_in_context": gold_in_context(retrieval["spoke_result"], context, correct_answer_list),
            "model_output": response["raw_output"],
            "model_output_list": response["answers"],
            "answer_source": answer_source_for(context, response["answers"]),
            "finish_reason": response["finish_reason"],
            "is_hit": score["hit"],
            "precision": round(score["precision"], 4),
            "recall": round(score["recall"], 4),
            "matched": score["matched"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "reasoning_tokens": response["reasoning_tokens"],
            "total_time_sec": round(time.time() - question_start_time, 2),
        })
        results.append(record)

    run_info = {
        "model_id": model_id,
        "reasoning_effort": reasoning_effort or "default",
        "instruction_version": INSTRUCTION_VERSION,
        "total_time_sec": round(time.time() - run_start_time, 2),
    }

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({"run_info": run_info, "results": results}, f, ensure_ascii=False, indent=4)

    return results


def run_with_reference_contexts(base_json_file, reference_json_file, output_json_file, model_id, reasoning_effort=None):
    """
    SPOKE data changes over time. Re-ask (v4 instruction) only the questions whose context
    in base_json_file differs from reference_json_file, using the reference context, so
    that all runs are evaluated on the same SPOKE snapshot. Other records are copied.
    """
    with open(base_json_file, 'r', encoding='utf-8') as f:
        base = json.load(f)
    with open(reference_json_file, 'r', encoding='utf-8') as f:
        reference = {r["question_id"]: r for r in json.load(f)["results"]}
    base_info = base["run_info"]
    run_effort = reasoning_effort or "default"
    if base_info.get("instruction_version") != INSTRUCTION_VERSION:
        raise ValueError(f"{base_json_file} is not a {INSTRUCTION_VERSION} result file")
    if base_info.get("model_id") != model_id or base_info.get("reasoning_effort") != run_effort:
        raise ValueError(
            f"{base_json_file} was run with {base_info.get('model_id')} / {base_info.get('reasoning_effort')}, "
            f"not {model_id} / {run_effort}")

    print(f"\n--- Re-asking with reference contexts ({model_id}): {base_json_file} ---")
    run_start_time = time.time()
    results = []
    for item in base["results"]:
        ref = reference[item["question_id"]]
        record = dict(item)
        if item["retrieved_context"] != ref["retrieved_context"]:
            question_start_time = time.time()
            instruction = INSTRUCTION_TEMPLATE.format(target_type=item["target_type"])
            prompt = build_prompt(ref["retrieved_context"], item["question"], item.get("options"), instruction)
            response = ask_llm(prompt, model_id, reasoning_effort)
            score = score_item(response["answers"], item["correct_answer"])
            record.update({
                "retrieval_status": ref["retrieval_status"],
                "retrieved_context": ref["retrieved_context"],
                "spoke_result": ref["spoke_result"],
                "gold_in_context": ref["gold_in_context"],
                "model_output": response["raw_output"],
                "model_output_list": response["answers"],
                "finish_reason": response["finish_reason"],
                "is_hit": score["hit"],
                "precision": round(score["precision"], 4),
                "recall": round(score["recall"], 4),
                "matched": score["matched"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "reasoning_tokens": response["reasoning_tokens"],
                "total_time_sec": round(time.time() - question_start_time, 2),
            })
            print(f"Re-asked {item['question_id']}")
        record["answer_source"] = answer_source_for(record["retrieved_context"], record["model_output_list"])
        results.append({k: record[k] for k in RECORD_FIELDS if k in record})

    run_info = dict(base_info)
    run_info["total_time_sec"] = round((base_info.get("total_time_sec") or 0) + time.time() - run_start_time, 2)

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({"run_info": run_info, "results": results}, f, ensure_ascii=False, indent=4)

    return results


def run_own_knowledge_fallback(base_json_file, output_json_file, model_id, reasoning_effort=None):
    """
    Re-ask only the questions that got an empty answer list in a v4 result file, this
    time allowing the model's own knowledge. Other records are copied unchanged.
    SPOKE is not queried again: the context stored in the v4 record is reused, so the
    contexts stay identical to the v4 run.
    """
    with open(base_json_file, 'r', encoding='utf-8') as f:
        base = json.load(f)
    base_info = base["run_info"]
    run_effort = reasoning_effort or "default"
    if base_info.get("instruction_version") != INSTRUCTION_VERSION:
        raise ValueError(f"{base_json_file} is not a {INSTRUCTION_VERSION} result file")
    if base_info.get("model_id") != model_id or base_info.get("reasoning_effort") != run_effort:
        raise ValueError(
            f"{base_json_file} was run with {base_info.get('model_id')} / {base_info.get('reasoning_effort')}, "
            f"not {model_id} / {run_effort}")

    print(f"\n--- Own-knowledge fallback with {model_id}: {base_json_file} ---")
    run_start_time = time.time()
    results = []
    for item in tqdm(base["results"], desc=f"Re-asking empty answers ({model_id})..."):
        record = dict(item)
        if item["model_output_list"]:
            record["answer_source"] = answer_source_for(item["retrieved_context"], item["model_output_list"])
            results.append(record)
            continue

        question_start_time = time.time()
        instruction = OWN_KNOWLEDGE_INSTRUCTION_TEMPLATE.format(target_type=item["target_type"])
        prompt = build_prompt(item["retrieved_context"], item["question"], item.get("options"), instruction)
        response = ask_llm(prompt, model_id, reasoning_effort)
        score = score_item(response["answers"], item["correct_answer"])

        record.update({
            "model_output": response["raw_output"],
            "model_output_list": response["answers"],
            "answer_source": "own_knowledge",
            "finish_reason": response["finish_reason"],
            "is_hit": score["hit"],
            "precision": round(score["precision"], 4),
            "recall": round(score["recall"], 4),
            "matched": score["matched"],
            "input_tokens": response["input_tokens"],
            "output_tokens": response["output_tokens"],
            "reasoning_tokens": response["reasoning_tokens"],
            "total_time_sec": round(time.time() - question_start_time, 2),
        })
        results.append(record)

    results = [{k: r[k] for k in RECORD_FIELDS if k in r} for r in results]

    run_info = {
        "model_id": model_id,
        "reasoning_effort": run_effort,
        "instruction_version": OWN_KNOWLEDGE_INSTRUCTION_VERSION,
        # v4 run time plus the time of this second pass
        "total_time_sec": round((base_info.get("total_time_sec") or 0) + time.time() - run_start_time, 2),
    }

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({"run_info": run_info, "results": results}, f, ensure_ascii=False, indent=4)

    return results