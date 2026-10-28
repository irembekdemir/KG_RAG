import json
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from kg_rag.utility import *
from kg_rag.evaluation import parse_to_list, score_item

CHAT_MODEL_ID_DEFAULT = 'gpt-5.4-mini'
SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]

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

    for row in tqdm(data, desc=f"Processing questions ({model_id})..."):
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

        options = row.get('options') or row.get('choices') or row.get('Options') or ""
        raw_correct_answer = row.get('answer') or row.get('correct_answer') or row.get('Correct_Answer') or ""
        correct_answer_list = parse_to_list(raw_correct_answer)

        target_type = row.get('target_type', 'disease')
        hop2_entity = row.get('hop2', '')
        hop1_entity = row.get('hop1', '')
        
        seed_entity = hop2_entity if hop2_entity else hop1_entity
        seed_type = row.get('hop2_type') if hop2_entity else row.get('hop1_type')

        # 1. Context 
        try:
            context = retrieve_context(
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
        except Exception as e:
            print(f"\nContext retrieval ERROR: {e}")
            context = ""

        # 2. Instruction
        # Names may contain commas (e.g. 'pregnancy loss, recurrent, susceptibility to'),
        # so answers are separated by semicolons
        instruction = (
            f"\n\nInstruction: Answer the question using the retrieved context. "
            f"List the names of ALL {target_type}s that answer the question, copying the names exactly as they appear in the context. "
            f"Output ONLY the names, separated by semicolons (;), with no explanations, no entity-type prefixes and no other text."
        )

        full_prompt = f"Context: {context}\n\nQuestion: {question}\nOptions: {options}{instruction}" if options else f"Context: {context}\n\nQuestion: {question}{instruction}"

        # 3. LLM Call
        try:
            output, p_tokens, c_tokens, r_tokens, t_tokens = get_GPT_response(
                full_prompt, SYSTEM_PROMPT, model_id, chat_deployment_id=model_id, temperature=TEMPERATURE, n=1,
                reasoning_effort=reasoning_effort
            )
        except Exception as e:
            output, p_tokens, c_tokens, r_tokens, t_tokens = f"Error: {str(e)}", 0, 0, 0, 0

        is_hit = score_item(output, correct_answer_list)["hit"] if correct_answer_list else False

        results.append({
            "question": question,
            "seed_entity": seed_entity,
            "seed_type": seed_type,
            "bridge_entity": hop1_entity if hop2_entity else "",
            "reasoning_effort": reasoning_effort or "default",
            "options": str(options),
            "correct_answer": correct_answer_list,        
            "retrieved_context": context,
            "context_found": bool(str(context).strip()),
            "model_output": output,
            "is_hit": is_hit,
            "input_tokens": p_tokens,                         
            "output_tokens": c_tokens,                        
            "reasoning_tokens": r_tokens,
            "total_tokens": t_tokens,
            "timestamp": datetime.now().isoformat()
        })

    output_dir = os.path.dirname(output_json_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    return results
