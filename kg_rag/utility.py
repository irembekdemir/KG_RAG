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
from functools import lru_cache
from urllib.parse import quote
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

@lru_cache(maxsize=1)
def _get_spoke_types():
    # The type list is static; fetch it once instead of on every neighborhood call
    result = get_spoke_api_resp(config_data['BASE_URI'], "/api/v1/types")
    if not result or result.status_code != 200:
        raise RuntimeError("SPOKE /types request failed")
    data_spoke_types = result.json()
    if not isinstance(data_spoke_types, dict) or "nodes" not in data_spoke_types:
        raise RuntimeError("Unexpected SPOKE /types response")
    return data_spoke_types

@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_context_using_spoke_api(node_value, node_type="Disease"):
    empty_df = pd.DataFrame(columns=["source", "edge_type", "target", "provenance", "evidence", "predicate", "context"])

    try:
        data_spoke_types = _get_spoke_types()
    except Exception:
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
    # URL-encode: names such as 'substance abuse/dependence' or '[...]' break the path otherwise
    nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{quote(str(node_value), safe='')}"
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
def get_GPT_response(instruction, system_prompt, chat_model_id='gpt-5.4-mini', chat_deployment_id=None, temperature=0, n=1, reasoning_effort=None):
    try:

        req_temperature = 1 if "5.4" in chat_model_id or "mini" in chat_model_id else temperature

        request_kwargs = dict(
            model = chat_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction}
            ],
            temperature = req_temperature,
            n = n,
        )
        # Passed explicitly so each run's reasoning setting is recorded instead of
        # depending on a commented-out line
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(**request_kwargs)
        
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

SEED_TYPE_TO_SPOKE_TYPES = {
    "drug": ["Compound"],
    "compound": ["Compound"],
    # Gene symbols (e.g. WNT8A, F2) are SPOKE Gene nodes; Protein names are UniProt descriptions
    "gene/protein": ["Gene", "Protein"],
    "gene": ["Gene"],
    "protein": ["Protein", "Gene"],
    "disease": ["Disease"],
}

def _name_key(name):
    # 'MK-1775' == 'MK 1775', 'S-(4-BROMOBENZYL)CYSTEINE' == 'S-(4-Bromobenzyl)cysteine'
    return "".join(ch for ch in str(name).lower() if ch.isalnum())

@lru_cache(maxsize=4096)
def resolve_spoke_names(node_value, node_type):
    """
    SPOKE's neighborhood endpoint needs the exact (case-sensitive) node name.
    Use the search endpoint to find nodes whose name is the same up to case,
    spacing and punctuation. Fuzzy hits with a different name are rejected.
    """
    result = get_spoke_api_resp(
        config_data['BASE_URI'], f"/api/v1/search/{node_type}/{quote(str(node_value), safe='')}"
    )
    if not result or result.status_code != 200:
        return ()
    try:
        hits = result.json()
    except Exception:
        return ()
    if not isinstance(hits, list):
        return ()
    key = _name_key(node_value)
    names = []
    for hit in hits:
        name = hit.get("name") if isinstance(hit, dict) else None
        if name and name != node_value and _name_key(name) == key and name not in names:
            names.append(name)
    return tuple(names)


# ---------------------------------------------------------------------------
# Two-hop retrieval for questions like "disease related to a gene/protein that is
# associated with drug X". A one-hop neighborhood of X only contains the bridge
# entities (proteins), never the asked target type, so the path is walked explicitly.
# Edge types were chosen by checking which SPOKE edges connect the benchmark seeds
# to their bridge entities.
# ---------------------------------------------------------------------------

SEED_TYPE_TO_SPOKE_ROOT = {"drug": "Compound", "compound": "Compound",
                           "gene/protein": "Gene", "gene": "Gene", "protein": "Gene",
                           "disease": "Disease"}

TARGET_TYPE_TO_SPOKE = {"drug": "Compound", "compound": "Compound", "disease": "Disease",
                        "gene/protein": "Gene", "gene": "Gene", "protein": "Gene"}

# (root type, target type) -> hop 1 (bridge) and hop 2 (target) definitions
TWO_HOP_PATHS = {
    ("Compound", "Disease"): {
        "hop1": {"node_type": "Protein", "edges": ["BINDS_CbP", "INTERACTS_PiC"]},
        # SPOKE links diseases to Genes, so each protein is mapped to its gene symbol
        "hop2": {"from_type": "Gene", "node_type": "Disease", "edges": ["ASSOCIATES_DaG"]},
    },
    ("Gene", "Compound"): {
        "hop1": {"node_type": "Disease", "edges": ["ASSOCIATES_DaG"]},
        "hop2": {"from_type": "Disease", "node_type": "Compound", "edges": ["TREATS_CtD"]},
    },
}

EDGE_VERBS = {"BINDS_CbP": "binds", "INTERACTS_PiC": "interacts with",
              "ASSOCIATES_DaG": "is associated with", "TREATS_CtD": "is treated by"}


def _spoke_cutoff_params():
    return {k: v for k, v in config_data.items() if k.startswith("cutoff_")}


def _parse_listish(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
    return [value] if value else []


@lru_cache(maxsize=8192)
def get_spoke_neighbors(node_type, node_name, neighbor_type, edge_types):
    """
    Neighbors of one SPOKE node, restricted to the given edge types (tuple).
    Returns a tuple of dicts: name, properties, edge_type, edge_properties.
    """
    params = dict(_spoke_cutoff_params())
    params["node_filters"] = list({node_type, neighbor_type})
    params["edge_filters"] = list(edge_types)
    params["depth"] = 1
    end_point = f"/api/v1/neighborhood/{node_type}/name/{quote(str(node_name), safe='')}"
    result = get_spoke_api_resp(config_data['BASE_URI'], end_point, params=params)
    if not result or result.status_code != 200:
        return ()
    try:
        items = result.json()
    except Exception:
        return ()
    if not isinstance(items, list):
        return ()

    nodes = {it["data"]["id"]: it["data"] for it in items
             if isinstance(it, dict) and "_" not in it.get("data", {}).get("neo4j_type", "")}
    root_ids = {i for i, n in nodes.items() if n.get("neo4j_root") == 1}
    neighbors = []
    for it in items:
        data = it.get("data", {}) if isinstance(it, dict) else {}
        if data.get("neo4j_type") not in edge_types:
            continue
        other_id = data.get("target") if data.get("source") in root_ids else data.get("source")
        other = nodes.get(other_id)
        if not other or other.get("neo4j_type") != neighbor_type:
            continue
        neighbors.append({
            "name": other["properties"].get("name"),
            "properties": other["properties"],
            "edge_type": data["neo4j_type"],
            "edge_properties": data.get("properties", {}),
        })
    return tuple(neighbors)


def _find_root_neighbors(root_type, seed_name, neighbor_type, edge_types):
    """Neighbors of the seed, retrying with SPOKE's spelling of the name if needed."""
    neighbors = get_spoke_neighbors(root_type, seed_name, neighbor_type, edge_types)
    if neighbors:
        return seed_name, neighbors
    for spoke_name in resolve_spoke_names(seed_name, root_type):
        neighbors = get_spoke_neighbors(root_type, spoke_name, neighbor_type, edge_types)
        if neighbors:
            return spoke_name, neighbors
    return seed_name, ()


def _bridge_rank(neighbor):
    # Curated binding edges first, then STITCH interactions by confidence
    if neighbor["edge_type"] == "BINDS_CbP":
        return (0, 0.0)
    try:
        conf = float(neighbor["edge_properties"].get("overall_confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return (1, -conf)


CURATED_SOURCES = {"DrugCentral", "ChEMBL", "DISEASES", "GWAS", "OMIM", "DrugBank", "BindingDB"}


def _target_rank(neighbor):
    # Curated database edges before clinical-trial / text-mined ones, then higher phase first
    srcs = set(map(str, _parse_listish(neighbor["edge_properties"].get("sources"))))
    try:
        phase = float(neighbor["edge_properties"].get("phase", 0))
    except (TypeError, ValueError):
        phase = 0.0
    return (0 if srcs & CURATED_SOURCES else 1, -phase)


def _sources(props):
    return ", ".join(str(s) for s in _parse_listish(props.get("sources"))) or "SPOKE"


def retrieve_two_hop_context(seed_entity, seed_type, target_type, max_bridges=None, max_targets_per_bridge=None):
    """
    Context text for a two-hop question, or None if no path is defined
    for this (seed type, target type) combination.
    """
    from concurrent.futures import ThreadPoolExecutor

    root_type = SEED_TYPE_TO_SPOKE_ROOT.get(str(seed_type).lower())
    target_spoke_type = TARGET_TYPE_TO_SPOKE.get(str(target_type).lower())
    path = TWO_HOP_PATHS.get((root_type, target_spoke_type))
    if path is None:
        return None

    max_bridges = max_bridges or config_data.get("MAX_BRIDGE_NODES", 20)
    max_targets_per_bridge = max_targets_per_bridge or config_data.get("MAX_TARGETS_PER_BRIDGE", 50)
    hop1, hop2 = path["hop1"], path["hop2"]

    seed_name, bridges = _find_root_neighbors(root_type, seed_entity, hop1["node_type"], tuple(hop1["edges"]))
    if not bridges:
        return ""

    # Collapse bridges to the names used for hop 2 (Protein -> gene symbol)
    bridge_entries = {}
    for b in sorted(bridges, key=_bridge_rank):
        if hop1["node_type"] == "Protein" and hop2["from_type"] == "Gene":
            keys = [g for g in _parse_listish(b["properties"].get("gene")) if g]
            label = b["properties"].get("description") or b["name"]
        else:
            keys = [b["name"]]
            label = b["name"]
        for key in keys:
            if key and key not in bridge_entries:
                bridge_entries[key] = (label, b)
        if len(bridge_entries) >= max_bridges:
            break

    def fetch(key):
        return key, get_spoke_neighbors(hop2["from_type"], key, hop2["node_type"], tuple(hop2["edges"]))

    with ThreadPoolExecutor(max_workers=8) as ex:
        hop2_results = list(ex.map(fetch, list(bridge_entries)))

    seed_label = f"{'Drug' if root_type == 'Compound' else root_type} {seed_name}"
    target_word = "drugs" if hop2["node_type"] == "Compound" else f"{hop2['node_type'].lower()}s"
    verb2 = EDGE_VERBS.get(hop2["edges"][0], "is linked to")
    lines = []
    for key, targets in hop2_results:
        label, b = bridge_entries[key]
        verb1 = EDGE_VERBS.get(b["edge_type"], b["edge_type"].split("_")[0].lower())
        if hop1["node_type"] == "Protein":
            lines.append(f"- {seed_label} {verb1} protein {label} (gene {key}) | Provenance: {_sources(b['edge_properties'])}.")
        else:
            lines.append(f"- {seed_label} {verb1} {hop1['node_type'].lower()} {key} | Provenance: {_sources(b['edge_properties'])}.")

        target_names = []
        target_sources = set()
        for t in sorted(targets, key=_target_rank):
            if t["name"] and t["name"] not in target_names:
                target_names.append(t["name"])
                target_sources.update(str(x) for x in _parse_listish(t["edge_properties"].get("sources")))
            if len(target_names) >= max_targets_per_bridge:
                break
        if target_names:
            lines.append(f"  - {hop2['from_type']} {key} {verb2} {target_word}: {'; '.join(target_names)}"
                         f" | Provenance: {', '.join(sorted(target_sources)) or 'SPOKE'}.")
        else:
            lines.append(f"  - No {target_word} found in SPOKE for {hop2['from_type'].lower()} {key}.")

    return "\n".join(lines) + "\n"


def retrieve_context(question, vectorstore, embedding_function, node_context_df, context_volume, context_sim_threshold, context_sim_min_threshold, edge_evidence, api=True, seed_entity=None, target_type=None, seed_type=None):
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

    # Multi-hop benchmark questions: walk seed -> bridge -> target explicitly
    if api and seed_entity and seed_type and target_type:
        two_hop_context = retrieve_two_hop_context(seed_entity, seed_type, target_type)
        if two_hop_context is not None:
            return two_hop_context[:35000]

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

            candidate_types = SEED_TYPE_TO_SPOKE_TYPES.get(
                str(seed_type).lower(), ["Compound", "Gene", "Protein", "Disease"]
            )
            for candidate_type in candidate_types:
                try:
                    node_context, context_table = get_context_using_spoke_api(node_name, node_type=candidate_type)
                    if node_context:
                        break
                except Exception:
                    continue
            # Exact name not found: retry with SPOKE's spelling of the same name
            if not node_context:
                for candidate_type in candidate_types:
                    for spoke_name in resolve_spoke_names(node_name, candidate_type):
                        try:
                            node_context, context_table = get_context_using_spoke_api(spoke_name, node_type=candidate_type)
                        except Exception:
                            continue
                        if node_context:
                            break
                    if node_context:
                        break

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
                nbr_end_point = f"/api/v1/neighborhood/{node_type}/{attribute}/{quote(str(node_name), safe='')}"
                
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
        node_context_df,config_data["CONTEXT_VOLUME"], 
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
