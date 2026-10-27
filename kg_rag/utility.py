import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory
import json
import os
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
import time
from dotenv import load_dotenv, find_dotenv
import torch

from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GPTQConfig
from kg_rag.config_loader import *
import ast
import requests
from openai import OpenAI

load_dotenv()

memory = Memory("cachegpt", verbose=0)

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

torch.cuda.empty_cache()
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

def get_spoke_api_resp(base_uri, end_point, params=None):
    uri = base_uri + end_point
    try:
        if params:
            return requests.get(uri, params=params, timeout=30)
        else:
            return requests.get(uri, timeout=30)
    except Exception as e:
        print(f"SPOKE API HTTP Request Exception: {e}")
        return None

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_context_using_spoke_api(node_value, node_type="Disease"):
    empty_df = pd.DataFrame(columns=["source", "edge_type", "target", "provenance", "evidence", "predicate", "context"])
    
    type_end_point = "/api/v1/types"
    result = get_spoke_api_resp(config_data['BASE_URI'], type_end_point)

    if not result or result.status_code != 200:
        return "", empty_df

    try:
        data_spoke_types = result.json()
    except Exception:
        return "", empty_df

    if not isinstance(data_spoke_types, dict) or "nodes" not in data_spoke_types:
        return "", empty_df

    node_types = list(data_spoke_types["nodes"].keys())
    edge_types = list(data_spoke_types["edges"].keys())

    node_types_to_remove = ["DatabaseTimestamp", "Version"]
    filtered_node_types = [nt for nt in node_types if nt not in node_types_to_remove]
    
    api_params = {
        'node_filters' : filtered_node_types,
        'edge_filters': edge_types,
        'cutoff_Compound_max_phase': config_data['cutoff_Compound_max_phase'],
        'cutoff_Protein_source': config_data['cutoff_Protein_source'],
        'cutoff_DaG_diseases_sources': config_data['cutoff_DaG_diseases_sources'],
        'cutoff_DaG_textmining': config_data['cutoff_DaG_textmining'],
        'cutoff_CtD_phase': config_data['cutoff_CtD_phase'],
        'cutoff_PiP_confidence': config_data['cutoff_PiP_confidence'],
        'cutoff_ACTeG_level': config_data['cutoff_ACTeG_level'],
        'cutoff_DpL_average_prevalence': config_data['cutoff_DpL_average_prevalence'],
        'depth' : config_data['depth']
    }
    
    attribute = "name"
    nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{node_value}"
    result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point, params=api_params)
    
    if not result or result.status_code != 200:
        return "", empty_df

    try:
        node_context = result.json()
    except Exception:
        return "", empty_df

    if not node_context or not isinstance(node_context, list) or len(node_context) == 0:
        return "", empty_df

    nbr_nodes = []
    nbr_edges = []
    
    for item in node_context:
        if not isinstance(item, dict) or "data" not in item:
            continue
            
        neo4j_type = item["data"].get("neo4j_type", "")
        
        if "_" not in neo4j_type:
            try:
                if neo4j_type == "Protein":
                    nbr_nodes.append((neo4j_type, item["data"]["id"], item["data"]["properties"]["description"]))
                else:
                    nbr_nodes.append((neo4j_type, item["data"]["id"], item["data"]["properties"]["name"]))
            except Exception:
                nbr_nodes.append((neo4j_type, item["data"]["id"], item["data"]["properties"].get("identifier", "")))
        else:
            try:
                provenance = ", ".join(item["data"]["properties"]["sources"])
            except Exception:
                provenance = "SPOKE-KG"     
            evidence = item["data"].get("properties", None)
            nbr_edges.append((item["data"].get("source"), neo4j_type, item["data"].get("target"), provenance, evidence))

    if not nbr_nodes or not nbr_edges:
        return "", empty_df

    nbr_nodes_df = pd.DataFrame(nbr_nodes, columns=["node_type", "node_id", "node_name"])
    nbr_edges_df = pd.DataFrame(nbr_edges, columns=["source", "edge_type", "target", "provenance", "evidence"])
    
    merge_1 = pd.merge(nbr_edges_df, nbr_nodes_df, left_on="source", right_on="node_id").drop("node_id", axis=1)
    merge_1.loc[:,"node_name"] = merge_1.node_type + " " + merge_1.node_name
    merge_1.drop(["source", "node_type"], axis=1, inplace=True)
    merge_1 = merge_1.rename(columns={"node_name":"source"})
    
    merge_2 = pd.merge(merge_1, nbr_nodes_df, left_on="target", right_on="node_id").drop("node_id", axis=1)
    merge_2.loc[:,"node_name"] = merge_2.node_type + " " + merge_2.node_name
    merge_2.drop(["target", "node_type"], axis=1, inplace=True)
    merge_2 = merge_2.rename(columns={"node_name":"target"})
    
    merge_2 = merge_2[["source", "edge_type", "target", "provenance", "evidence"]]
    merge_2.loc[:, "predicate"] = merge_2.edge_type.apply(lambda x: str(x).split("_")[0])
    
    merge_2.loc[:, "context"] = (
        "- Relationship: " + merge_2.source + " **" + merge_2.predicate.str.lower() + "** " + merge_2.target + 
        " | Source/Provenance: " + merge_2.provenance.astype(str) + "."
    )
    
    context = merge_2.context.str.cat(sep='\n')
    
    try:
        first_props = node_context[0]["data"].get("properties", {})
        source_info = first_props.get("source", "SPOKE")
        id_info = first_props.get("identifier", "")
        context += f"{node_value} has a {source_info} identifier of {id_info} and Provenance of this is from {source_info}."
    except Exception:
        pass

    return context, merge_2

def get_prompt(instruction, new_system_prompt):
    system_prompt = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + system_prompt + instruction + E_INST
    return prompt_template

def llama_model(model_name, branch_name, cache_dir, temperature=0, top_p=1, max_new_tokens=512, stream=False, method='method-1'):
    if method == 'method-1':
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir
        )
    elif method == 'method-2':
        import transformers
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, revision=branch_name, cache_dir=cache_dir, legacy=False)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, revision=branch_name, cache_dir=cache_dir)        
    
    if not stream:
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=max_new_tokens, do_sample=True)
    else:
        streamer = TextStreamer(tokenizer)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto", max_new_tokens=max_new_tokens, do_sample=True, streamer=streamer)        
    
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": temperature, "top_p": top_p})
    return llm

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_GPT_response(instruction, system_prompt, chat_model_id='gpt-5.4-mini', chat_deployment_id=None, temperature=0, n=1):
    try:

        req_temperature = 1 if "5.4" in chat_model_id or "mini" in chat_model_id else temperature
        
        response = client.chat.completions.create(
            model = chat_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            temperature = req_temperature,
            n = n,

            #reasoning_effort = "medium" #remove for reasonin off mode
        )
        
        outputs = [choice.message.content.strip() for choice in response.choices]
        
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            reasoning_tokens = 0
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0
        else:
            prompt_tokens, completion_tokens, reasoning_tokens, total_tokens = 0, 0, 0, 0

        combined_output = ", ".join(outputs) if n > 1 else outputs[0]

        return combined_output, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens

    except Exception as e:
        print(f"\nAPI call ERROR ({chat_model_id}): {e}")
        raise e

def stream_out(output):
    if not output:
        return
    CHUNK_SIZE = max(1, int(round(len(output)/50)))
    SLEEP_TIME = 0.05
    for i in range(0, len(output), CHUNK_SIZE):
        print(output[i:i+CHUNK_SIZE], end='')
        sys.stdout.flush()
        time.sleep(SLEEP_TIME)
    print("\n")

def get_gpt35():
    return 'gpt-5.4-mini', None

def disease_entity_extractor(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    try:
        resp_content, _, _, _, _ = get_GPT_response(text, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0, n=1)
        entity_dict = json.loads(resp_content)
        return entity_dict.get("Diseases", [])
    except Exception:
        return []

def disease_entity_extractor_v2(text):
    chat_model_id, chat_deployment_id = get_gpt35()
    prompt_updated = system_prompts["DISEASE_ENTITY_EXTRACTION"] + "\nSentence : " + text
    try:
        resp_content, _, _, _, _ = get_GPT_response(prompt_updated, system_prompts["DISEASE_ENTITY_EXTRACTION"], chat_model_id, chat_deployment_id, temperature=0, n=1)
        entity_dict = json.loads(resp_content)
        res = entity_dict.get("Diseases", [])
        return res if isinstance(res, list) else []
    except Exception:
        return []

def load_sentence_transformer(sentence_embedding_model):
    return SentenceTransformerEmbeddings(model_name=sentence_embedding_model)

def load_chroma(vector_db_path, sentence_embedding_model):
    embedding_function = load_sentence_transformer(sentence_embedding_model)
    return Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)

def retrieve_context(question, seed_entity, target_type, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
    node_hits = []

    if api and seed_entity:
        node_hits = [str(seed_entity).strip()]
    
    if not node_hits:
        try:
            node_hits_raw = vectorstore.similarity_search_with_score(question, k=3)
            node_hits = [node[0].page_content for node in node_hits_raw]
        except Exception:
            node_hits = []

    if not node_hits:
        return ""

    node_context_extracted = ""

    for node_name in node_hits:
        context_table = None
        if not api:
            if node_context_df is not None and not node_context_df.empty and "node_name" in node_context_df.columns:
                matched_rows = node_context_df[node_context_df.node_name == node_name]
                if matched_rows.empty:
                    continue
                node_context_extracted += matched_rows.node_context.values[0] + " "
        else:
            node_context = ""

            candidate_types = ["Compound", "Protein", "Disease"]
            for candidate_type in candidate_types:
                try:
                    node_context, context_table = get_context_using_spoke_api(node_name, node_type=candidate_type)
                    if node_context:
                        break
                except Exception:
                    continue

            if edge_evidence and context_table is not None and not context_table.empty:
                context_table.loc[:, "context"] = (
                    "- Relationship: " + context_table.source + " **" + context_table.predicate.str.lower() + "** " + context_table.target + 
                    " | Provenance: " + context_table.provenance.astype(str) + "."
                )
                node_context_extracted += context_table.context.str.cat(sep='\n') + "\n"
            else:
                if node_context:
                    node_context_extracted += node_context + "\n"

    MAX_CONTEXT_CHARS = 35000  
    if len(node_context_extracted) > MAX_CONTEXT_CHARS:
        node_context_extracted = node_context_extracted[:MAX_CONTEXT_CHARS] + " [Context truncated due to length...]"

    return node_context_extracted

def retrieve_context_with_raw(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True):
    """
    Both raw SPOKE API JSON data and processed text context
    """
    node_hits = []
    raw_spoke_json_data = [] 

    if api:
        entities = []
        if "gene/protein" in question:
            try:
                parts = question.split("gene/protein")
                if len(parts) > 1:
                    target_entity = parts[1].split(".")[0].split("\n")[0].strip()
                    if target_entity:
                        entities.append(target_entity)
            except Exception:
                pass
        elif "drug" in question:
            try:
                parts = question.split("drug")
                if len(parts) > 1:
                    target_entity = parts[1].split(".")[0].split("\n")[0].strip()
                    if target_entity:
                        entities.append(target_entity)
            except Exception:
                pass
        
        if not entities:
            entities = disease_entity_extractor_v2(question) or []

        if entities and isinstance(entities, list) and len(entities) > 0:
            for entity in entities:
                try:
                    node_search_result = vectorstore.similarity_search_with_score(entity, k=2)
                    if node_search_result:
                        for res in node_search_result:
                            if res[0].page_content not in node_hits:
                                node_hits.append(res[0].page_content)
                except Exception:
                    continue
    
    if not node_hits:
        try:
            node_hits_raw = vectorstore.similarity_search_with_score(question, k=5)
            node_hits = [node[0].page_content for node in node_hits_raw]
        except Exception:
            node_hits = []

    if not node_hits:
        return "", []

    node_context_extracted = ""

    for node_name in node_hits:
        context_table = None
        if not api:
            if node_context_df is not None and not node_context_df.empty and "node_name" in node_context_df.columns:
                matched_rows = node_context_df[node_context_df.node_name == node_name]
                if matched_rows.empty:
                    continue
                node_context_extracted += matched_rows.node_context.values[0] + " "
        else:
            try:
                node_type = "Disease" 
                attribute = "name"
                nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{node_name}"
                
                api_result = get_spoke_api_resp(config_data['BASE_URI'], nbr_end_point)
                if api_result and api_result.status_code == 200:
                    raw_json = api_result.json()
                    raw_spoke_json_data.append({
                        "node_name": node_name,
                        "raw_data": raw_json
                    })
                
                node_context, context_table = get_context_using_spoke_api(node_name)
            except Exception:
                node_context, context_table = "", None

            if edge_evidence and context_table is not None and not context_table.empty:
                context_table.loc[:, "context"] = (
                    context_table.source + " " + 
                    context_table.predicate.str.lower() + " " + 
                    context_table.target + 
                    " (Provenance: " + context_table.provenance.astype(str) + "). "
                )
                node_context_extracted += context_table.context.str.cat(sep=' ')
            else:
                if node_context:
                    node_context_extracted += node_context + " "

    MAX_CONTEXT_CHARS = 35000  
    if len(node_context_extracted) > MAX_CONTEXT_CHARS:
        node_context_extracted = node_context_extracted[:MAX_CONTEXT_CHARS] + " [Context truncated due to length...]"

    return node_context_extracted, raw_spoke_json_data

def interactive(question, vectorstore, node_context_df, embedding_function_for_context_retrieval, llm_type, edge_evidence, system_prompt, api=True, llama_method="method-1"):
    print(" ")
    input("Press enter for Step 1 - Disease entity extraction using GPT")
    print("Processing ...")
    entities = disease_entity_extractor_v2(question) or []
    print("Extracted entity from the prompt = '{}'".format(", ".join(entities)))
    print(" ")
    
    input("Press enter for Step 2 - Match extracted Disease entity to SPOKE nodes")
    print("Finding vector similarity ...")
    node_hits = []
    for entity in entities:
        node_search_result = vectorstore.similarity_search_with_score(entity, k=1)
        if node_search_result:
            node_hits.append(node_search_result[0][0].page_content)
    print("Matched entities from SPOKE = '{}'".format(", ".join(node_hits)))
    print(" ")
    
    input("Press enter for Step 3 - Context extraction from SPOKE")
    node_context = []
    for node_name in node_hits:
        if not api:
            if node_context_df is not None and not node_context_df.empty:
                matched = node_context_df[node_context_df.node_name == node_name]
                if not matched.empty:
                    node_context.append(matched.node_context.values[0])
        else:
            context, context_table = get_context_using_spoke_api(node_name)
            if context:
                node_context.append(context)
    print("Extracted Context is : ")
    print(". ".join(node_context))
    print(" ")

    input("Press enter for Step 4 - Context pruning")
    node_context_extracted = retrieve_context(
        question, vectorstore, embedding_function_for_context_retrieval, 
        node_context_df, config_data["CONTEXT_VOLUME"], 
        config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"], 
        config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"], 
        edge_evidence, api=api
    )
    print("Pruned Context is : ")
    print(node_context_extracted)
    print(" ")
    
    input("Press enter for Step 5 - LLM prompting")
    print("Prompting ", llm_type)
    if llm_type == "llama":
        template = get_prompt("Context:\n\n{context} \n\nQuestion: {question}", system_prompt)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = llama_model(config_data["LLAMA_MODEL_NAME"], config_data["LLAMA_MODEL_BRANCH"], config_data["LLM_CACHE_DIR"], stream=True, method=llama_method) 
        chain = prompt | llm
        output = chain.invoke({"context": node_context_extracted, "question": question})
    elif "gpt" in llm_type:
        enriched_prompt = "Context: "+ node_context_extracted + "\n" + "Question: " + question
        output, _, _, _, _ = get_GPT_response(enriched_prompt, system_prompt, llm_type, llm_type, temperature=config_data["LLM_TEMPERATURE"], n=1)
        stream_out(output)
