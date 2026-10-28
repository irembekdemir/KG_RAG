from kg_rag.utility import *
import argparse
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-g', type=str, default='gpt-5.4-mini', help='GPT model selection')
parser.add_argument('-b', '--benchmark_file', type=str, default=None, help='Path to benchmark JSON file')
parser.add_argument('-o', '--output_file', type=str, default='benchmark_results_with_tokens.csv', help='Path to output CSV file')
args = parser.parse_args()

CHAT_MODEL_ID = args.g
SYSTEM_PROMPT = system_prompts["PROMPT_BASED_TEXT_GENERATION"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

def process_benchmark(json_path, output_csv):
    print(f"\n--- Benchmark Starting: {json_path} ---")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in tqdm(data, desc="Questions are processing..."):
        question = item.get('question') or item.get('prompt') or item.get('text')
        options = item.get('options') or item.get('choices') or ""
        correct_answer = item.get('answer') or item.get('correct_answer') or ""

        # Prompt-based baseline: no KG context, the question only
        if options:
            full_prompt = f"Question: {question}\nOptions: {options}"
        else:
            full_prompt = f"Question: {question}"

        # 3. OpenAI Çağrısı ve Token Takibi
        try:
            output, p_tokens, c_tokens, r_tokens, t_tokens = get_GPT_response(
                full_prompt, 
                SYSTEM_PROMPT, 
                CHAT_MODEL_ID, 
                CHAT_DEPLOYMENT_ID, 
                temperature=TEMPERATURE
            )
        except Exception as e:
            output, p_tokens, c_tokens, r_tokens, t_tokens = f"Error: {e}", 0, 0, 0, 0

        results.append({
            "Question": question,
            "Options": str(options),
            "Correct_Answer": str(correct_answer),
            "Model_Output": output,
            "Prompt_Tokens": p_tokens,
            "Completion_Tokens": c_tokens,
            "Reasoning_Tokens": r_tokens,
            "Total_Tokens": t_tokens,
            "Timestamp": datetime.now().isoformat()
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n Results and token usage are recorded in  '{output_csv}'\n")

def main():
    if args.benchmark_file:
        # Otomatik JSON toplu koşum modu
        process_benchmark(args.benchmark_file, args.output_file)
    else:
        # Orijinal manuel soru modu
        print(" ")
        question = input("Enter your question : ")    
        print("Here is the prompt-based answer:")
        print("")
        output, p_tokens, c_tokens, _, t_tokens = get_GPT_response(
            question, SYSTEM_PROMPT, CHAT_MODEL_ID, CHAT_DEPLOYMENT_ID, temperature=TEMPERATURE
        )
        stream_out(output)
        print(f"\n[Token Usage] Prompt: {p_tokens} | Completion: {c_tokens} | Total: {t_tokens}")

if __name__ == "__main__":
    main()
