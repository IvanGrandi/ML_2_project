import json
import os
import pandas as pd
import preprocessing as prep
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import file_manager as fm
import logger
import pipeline as pipe
import cleaning as clean

RUN_ON_SAMPLE = False  # Set to False to run on ALL data
SAMPLE_SIZE = 25000
MAX_CORES_USED = 6  
SAVE_FILE = "data_processed.pkl"

if __name__ == "__main__":

    # Look for saved data file to load data faster.
    df = fm.load_saved_data(SAVE_FILE)

    if df is not None:
        print("🚀 Ready for the next step (Visualization or Machine Learning)!")
    else:
        print(f"🔍 No save file '{SAVE_FILE}' detected. Launching full processing...")
        
        # Load json files and get a dataframe containing all the data
        start_step1 = time.time()
        df = fm.load_json_files(data_folder="data")
        
        if df is None:
            exit()        
        
        end_step1 = time.time()
        step1_time = end_step1 - start_step1

        # Get a sample of the dataframe for testing purpose
        df = pipe.apply_sampling(df, RUN_ON_SAMPLE, SAMPLE_SIZE)
        
        # Applying author and synopsis cleaning
        start_step2 = time.time()
        logger.show_missing_values_report(df) 
        clean.show_duplicates_report(df)
        df = clean.remove_duplicates(df)
        print(f"📊 DataFrame size is now: {len(df)} rows")
        df = clean.run_column_check(df)
        print(f"📊 DataFrame size is now: {len(df)} rows")

        end_step2 = time.time()
        step2_time = end_step2 - start_step2

        start_step3 = time.time()
        df = pipe.run_parallel_nlp(df, MAX_CORES_USED)       
        end_step3 = time.time()
        step3_time = end_step3 - start_step3

        # Display final columns
        print("\nFinal columns of your DataFrame:")
        print(df.columns.tolist())
            
        # Save the dataframe to load it faster next time
        fm.save_dataframe(df, SAVE_FILE)
       
        # Generate and display the timing report
        logger.show_time_report(step1_time, step2_time, step3_time, len(df))

    # Display tables  
    logger.show_visualizations(df)