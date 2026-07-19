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
import visualizations as viz
import subprocess

RUN_ON_SAMPLE = False  # Set to False to run on ALL data
SAMPLE_SIZE = 25000
MAX_CORES_USED = 6  
SAVE_FILE = "data_processed.pkl"

def load_and_process_pipeline():
    # Look for saved data file to load data faster.
    df = fm.load_saved_data(SAVE_FILE)

    if df is not None:
        print("🚀 Ready for the next step (Visualization or Machine Learning)!")
        return df
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

        return df

def run_visualization_menu(df):
    """Sub-menu for exploratory data analysis."""
    if df is None:
        print("❌ Error: You must load or process data (Option 1) before viewing charts!")
        return
    
    while True:
        print("\n==================================================")
        print("📊 EXPLORATORY DATA ANALYSIS MENU")
        print("==================================================")
        print("1. Plot Rating Distribution")
        print("2. Plot Likes & Dislikes Distribution (Boxplot)")
        print("3. Plot Correlation Matrix")
        print("4. Plot Temporal Trends (Reviews & Ratings over time)")
        print("5. Plot Text Length Distribution")
        print("6. Plot Global Word Cloud (User Reviews)")
        print("7. Plot Comparative Word Clouds (Tops vs Flops)")
        print("8. Run All Visualizations sequentially")
        print("9. Exit and close")
        print("==================================================")
        
        choice = input("👉 Enter your choice (1-9): ").strip()
        print("\n")

        if choice == "1":
            viz.plot_rating_distribution(df)
        elif choice == "2":
            viz.plot_engagement_distribution(df)
        elif choice == "3":
            viz.plot_correlation_matrix(df)
        elif choice == "4":
            viz.plot_temporal_trends(df)
        elif choice == "5":
            viz.plot_text_length_distribution(df, text_column="text_content")
        elif choice == "6":
            viz.plot_global_wordcloud(df, text_column="text_content")
        elif choice == "7":
            viz.plot_comparative_wordclouds(df, text_column="text_content")
        elif choice == "8":
            print("🚀 Launching all visualizations...")
            viz.plot_rating_distribution(df)
            viz.plot_engagement_distribution(df)
            viz.plot_correlation_matrix(df)
            viz.plot_temporal_trends(df)
            viz.plot_text_length_distribution(df, text_column="text_content")
            viz.plot_global_wordcloud(df, text_column="text_content")
            viz.plot_comparative_wordclouds(df, text_column="text_content")
        elif choice == "9":
            print("👋 Exiting visualization menu. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 9.")

if __name__ == "__main__":

    df = None  
    while True:
        print("\n" + "="*50)
        print("🚀 MAIN PROJECT MENU - ML & TEXTUAL ANALYSIS")
        print("="*50)
        print("1. 📂 Load / Process Dataset (Run Pipeline)")
        print("2. 📊 Open Exploratory Data Analysis & Charts")
        print("3. 🤖 Launch Jupyter Notebook for Predictive Models")
        print("4. ❌ Exit")
        print("="*50)
        
        main_choice = input("👉 Select an option (1-4): ").strip()
        if main_choice == "1":
            df = load_and_process_pipeline()
            if df is not None:
                print("✅ Data is ready in memory for analysis!")
                
        elif main_choice == "2":
            if df is not None:
                run_visualization_menu(df)
            else: 
                print("❌ Error: You must load or process data (Option 1) before viewing charts!")
            
        elif main_choice == "3":
            notebook_path = "training.ipynb"
            if df is not None:
                if os.path.exists(notebook_path):
                    print(f"💻 Opening {notebook_path} in VS Code...")
                    # La commande 'code' appelle VS Code directement
                    subprocess.Popen(["code", notebook_path], shell=True)
                else:
                    print(f"❌ Error: The file '{notebook_path}' was not found in this directory.")
            else: 
                print("❌ Error: You must load or process data (Option 1) before launching the notebook!")

        elif main_choice == "4":
            print("👋 Exiting the application. Goodbye!")
            break
        else:
            print("❌ Invalid option. Please enter a number between 1 and 4.")
