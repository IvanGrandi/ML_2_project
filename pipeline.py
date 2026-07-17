# pipeline.py
import os
import numpy as np
import pandas as pd
import preprocessing as prep
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def apply_sampling(df, run_on_sample, sample_size):
    """Randomly reduces the DataFrame size if the option is enabled."""
    if run_on_sample and len(df) > sample_size:
        print(f"⚡ Sample Mode Enabled: Randomly selecting {sample_size} rows out of {len(df)}...")
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    elif run_on_sample:
        print(f"ℹ️ DataFrame contains {len(df)} rows, which is less than the target sample size ({sample_size}). All rows will be processed.")
    return df


def _process_single_chunk(df_chunk):
    """Internal function to clean reviews from a single chunk (used in parallel processing)."""
    review_column = "text_content"
    df_chunk[review_column] = df_chunk[review_column].apply(
        lambda x: " ".join(prep.get_cleaning_steps(x)["No StopWords"]) if pd.notna(x) else ""
    )
    return df_chunk

def run_parallel_nlp(df, max_cores):
    """Splits the DataFrame and applies heavy NLP preprocessing using multi-processing."""
    review_column = "text_content"
    if review_column not in df.columns:
        print("⚠️ Review column is missing, heavy NLP step skipped.")
        return df

    print("Applying heavy NLP preprocessing on reviews...")
    available_cores = os.cpu_count()
    cores_to_use = min(max_cores, available_cores)
    
    print(f"💻 CPU detected: {available_cores} logical cores available.")
    print(f"💻 CPU utilized: {cores_to_use} logical cores assigned.")
    
    # Split the DataFrame into chunks
    num_chunks = cores_to_use * 4
    chunks = np.array_split(df, num_chunks)
    
    processed_chunks = []
    with ProcessPoolExecutor(max_workers=cores_to_use) as executor:
        futures = [executor.submit(_process_single_chunk, chunk) for chunk in chunks]
        
        for future in tqdm(futures, desc="Processing NLP batches", total=len(futures)):
            processed_chunks.append(future.result())
            
    df_final = pd.concat(processed_chunks, ignore_index=True)
    print("✅ NLP preprocessing completed!")
    return df_final