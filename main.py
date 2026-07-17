# main.py
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

RUN_ON_SAMPLE = True  # Set to False to run on ALL data
SAMPLE_SIZE = 25000
MAX_CORES_USED = 6  
SAVE_FILE = "data_processed.pkl"

if __name__ == "__main__":

    df = fm.load_saved_data(SAVE_FILE)

    if df is not None:
        print("🚀 Ready for the next step (Visualization or Machine Learning)!")
        
    else:
        print(f"🔍 No save file '{SAVE_FILE}' detected. Launching full processing...")
        
        start_step1 = time.time()
        df = fm.load_json_files(data_folder="data")
        if df is None:
            exit()
        
        end_step1 = time.time()
        step1_time = end_step1 - start_step1

        # ==========================================
        # APPLYING SAMPLING
        # ==========================================
        df = pipe.apply_sampling(df, RUN_ON_SAMPLE, SAMPLE_SIZE)
        
        # --- METADATA CLEANING ---
        start_step2 = time.time()
        df = pipe.clean_metadata(df)

        end_step2 = time.time()
        step2_time = end_step2 - start_step2

        start_step3 = time.time()
        df = pipe.run_parallel_nlp(df, MAX_CORES_USED)       
        end_step3 = time.time()
        step3_time = end_step3 - start_step3

        # Display final columns
        print("\nFinal columns of your DataFrame:")
        print(df.columns.tolist())
            

        # ==========================================
        #        ONE-LINE SAVE (PKL FORMAT)
        # ==========================================
        fm.save_dataframe(df, SAVE_FILE)


        # ==========================================
        #         PERFORMANCE REPORT DISPLAY
        # ==========================================
       
        # Generate and display the timing report
        logger.show_time_report(step1_time, step2_time, step3_time, len(df))

    # ==========================================
    #         DISPLAY TABLE CONTENT HERE
    # ==========================================
    
    logger.show_visualizations(df)