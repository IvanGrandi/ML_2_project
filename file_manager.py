# file_manager.py
import json
import os
import time
import pandas as pd

def load_json_files(data_folder="data"):
    """Loads and aggregates all JSON files from the specified folder into a single DataFrame."""
    if not os.path.exists(data_folder):
        print(f"❌ The folder '{data_folder}' does not exist.")
        return None
        
    json_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]

    if not json_files:
        print(f"Warning: No JSON files found in '{data_folder}'.")
        return None
    else: 
        print(f"Found {len(json_files)} JSON files in '{data_folder}'.")
        
    df_list = []
    for file in json_files:
        full_path = os.path.join(data_folder, file)
        print(f"  -> Processing {file}...")
        
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Resolve naming conflict of the root ID
        if "id" in data:
            data["movie_id"] = data.pop("id")

        all_metadata = [key for key in data.keys() if key != "reviews"]

        temp_df = pd.json_normalize(
            data, 
            record_path=["reviews"], 
            meta=all_metadata
        )
        
        if "id" in temp_df.columns:
            temp_df = temp_df.rename(columns={"id": "review_id"})

        df_list.append(temp_df)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print(f"\n✅ Merge successful! Total accumulated rows: {len(df)}")
        return df
    
    print("❌ No DataFrame could be created.")   
    return None

def load_saved_data(pickle_path):
    """Loads a DataFrame from a Pickle file if it exists."""
    if os.path.exists(pickle_path):
        print(f" Save file found! Fast-loading '{pickle_path}'...")
        start_load = time.time()
        df = pd.read_pickle(pickle_path)
        print(f"✅ Data successfully loaded in {time.time() - start_load:.2f} seconds ({len(df)} rows).")
        return df
    return None

def save_dataframe(df, pickle_path):
    """Saves the DataFrame to a Pickle file."""
    print(f"\nSaving processed DataFrame to '{pickle_path}'...")
    df.to_pickle(pickle_path)
    print("✅ Save completed successfully!")