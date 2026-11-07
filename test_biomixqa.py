import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from kg_rag.utility import (
    config_data,
    load_sentence_transformer,
    load_chroma,
    retrieve_context
)

def run_benchmark():
    print("BiomixQA MCQ Veri Seti Yükleniyor...")
    dataset = load_dataset("kg-rag/BiomixQA", "mcq", split="train")
    total_questions = len(dataset)
    print(f"Toplam Soru Sayısı: {total_questions}")
    
    print("Vector store ve embedding modelleri hazırlanıyor...")
    embedding_model_name = config_data.get('SENTENCE_EMBEDDING_MODEL_NAME') or config_data.get('EMBEDDING_MODEL_NAME') or "all-MiniLM-L6-v2"
    
    embedding_function = load_sentence_transformer(embedding_model_name)
    vectorstore = load_chroma(config_data['VECTOR_DB_PATH'], embedding_model_name)
    node_context_df = pd.read_csv(config_data['NODE_CONTEXT_PATH'])

    results = []
    context_found_count = 0

    print("\nBenchmark Testi Başlatılıyor (Tüm Veri Seti)...\n")
    for row in tqdm(dataset):
        question = row.get('question') or row.get('Question') or row.get('prompt') or row.get('text') or list(row.values())[0]
        options = row.get('options') or row.get('Options') or row.get('choices') or row.get('distractors') or ""
        correct_answer = row.get('answer') or row.get('Answer') or row.get('label') or row.get('correct_answer') or ""

        try:
            context = retrieve_context(
                question,
                vectorstore,
                embedding_function,
                node_context_df,
                config_data['CONTEXT_VOLUME'],
                config_data['QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD'],
                config_data['QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY'],
                config_data['EDGE_EVIDENCE'],
                api=False
            )
        except Exception:
            context = ""

        if context and str(context).strip() != "":
            context_found_count += 1

        results.append({
            "Question": question,
            "Options": str(options),
            "Correct_Answer": str(correct_answer),
            "Retrieved_Context": context
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv("biomixqa_benchmark_results.csv", index=False)
    
    retrieval_rate = (context_found_count / total_questions) * 100
    print("\n--- BENCHMARK TEST SONUÇLARI ---")
    print(f"Toplam İşlenen Soru : {total_questions}")
    print(f"Bağlam Bulunan Soru : {context_found_count}")
    print(f"Bağlam Çekme Başarısı: %{retrieval_rate:.2f}")
    print("Tüm sonuçlar 'biomixqa_benchmark_results.csv' dosyasına yazıldı.")

if __name__ == "__main__":
    run_benchmark()