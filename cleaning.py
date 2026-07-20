import re
import pandas as pd
import numpy as np

def remove_duplicates(df):
    """
    Removes exact duplicates and enforces unique review_id.
    """
    initial_count = len(df)
    print("\nStarting duplicate removal process...")

    # List columns not of type list.
    hashable_cols = [col for col in df.columns if not df[col].apply(lambda x: isinstance(x, list)).any()]

    # 1. Strict duplicates removal
    df = df.drop_duplicates(subset=hashable_cols).reset_index(drop=True)
    
    # 2. Remove duplicates on review_id basis
    if "review_id" in df.columns:
        df = df.drop_duplicates(subset=["review_id"], keep="first").reset_index(drop=True)

    final_count = len(df)
    removed = initial_count - final_count
    
    print(f"✅ Duplicate removal complete. Dropped {removed} rows. Remaining rows: {final_count}")
    return df


def show_duplicates_report(df):
    """
    Analyzes and displays global and key-based duplicates within the DataFrame.
    """
    print("\n" + "="*50)
    print("DUPLICATES DETECTION REPORT")
    print("="*50)
    
    total_rows = len(df)
    print(f"Total rows in DataFrame: {total_rows}\n")

    # 1. Doublons absolus (toutes les colonnes identiques)
    hashable_cols = [col for col in df.columns if not df[col].apply(lambda x: isinstance(x, list)).any()]
    global_dups = df.duplicated(subset=hashable_cols).sum()
    print(f"• Strict Global Duplicates (Exact row copies):")
    print(f"  ↳ Found: {global_dups} rows ({ (global_dups/total_rows)*100:.2f}% )")

    # 2. Doublons sur l'identifiant unique de la review
    if "review_id" in df.columns:
        review_dups = df.duplicated(subset=["review_id"]).sum()
        print(f"• Duplicate 'review_id' entries:")
        print(f"  ↳ Found: {review_dups} occurrences")

    print("="*50 + "\n")

def run_column_check(df):
    """Quickly cleans the synopsis and author columns."""
    print("\n" + "="*50)
    print("RUNNING GLOBAL DATA SANITY CHECK & CLEANING")
    print("="*50)
    total_rows = len(df)


    # --- 1. CLEANING STRINGS & METADATA ---
    string_cols = [col for col in ["name","synopsis", "review_id", "author"] if col in df.columns]
    for col in string_cols:
        df[col], valid_mask = process_string_column(df[col])
        valid_count = valid_mask.sum()
        print(f"• Column '{col}' (String):")
        print(f"  ↳ Valid (populated): {valid_count} | Invalid (empty/missing): {total_rows - valid_count}")
    
    print("Checking numeric columns...")

    # --- 1. Identifiers (Int, >= 0) -> Drop Strategy ---
    
    if "movie_id" in df.columns:
        df["movie_id"] = process_numeric_column(
            df["movie_id"], min_val=0, is_integer=True, fallback_strategy="drop"
        )
    
    initial_rows = len(df)
    df = df.dropna(subset=["movie_id"]).reset_index(drop=True)


    # --- 2. Ratings (Float, entre 0.5 et 5.0) -> Mean Strategy ---
    if "rating" in df.columns:
        df["rating"] = process_numeric_column(
            df["rating"], min_val=0.0, max_val=5.0, is_integer=False, fallback_strategy="mean"
        )

    # --- 3. Likes / Dislikes (Int, >= 0) -> Zero Strategy ---
    for col in [c for c in ["likes", "dislikes", "runtime_in_minutes"] if c in df.columns]:
        df[col] = process_numeric_column(
            df[col], min_val=0, is_integer=True, fallback_strategy="zero"
        )

    print("Checking date columns...")

    # We list all columns of date type to check
    date_cols = [c for c in ["date", "release_date"] if c in df.columns]

    for col in date_cols:
        # 1. Idenntify and replace invalid strings by NaN
        df[col] = process_date_column(df[col])
        
        # 2. Remove invalid lines
        df = df.dropna(subset=[col]).reset_index(drop=True)
        
        # 3. Display result
        print(f"  ❌ Invalid rows in '{col}' dropped. DataFrame size is now: {len(df)} rows.")
    print("✅ Metadata cleaning completed.")
    return df

# Clean input: replace newlines with spaces and strip outer whitespaces
def clean_input(text):
    """Cleans a string by removing newlines and collapsing multiple spaces."""
    if not text or pd.isna(text):
        return ""
    text = str(text).replace("\n", "")
    return re.sub(r"\s+", " ", text).strip()

def process_string_column(series):
    """
    Cleans a string column, enforces string type, and tracks validity.
    Returns the cleaned series and a boolean mask of valid rows.
    """
    # 1. Force formatting and clean text
    cleaned_series = series.fillna("").astype(str).apply(clean_input)
    
    # 2. A row is valid if it's not just an empty string after cleaning
    valid_mask = cleaned_series != ""
    
    return cleaned_series, valid_mask


def process_numeric_column(series, min_val=None, max_val=None, is_integer=False, fallback_strategy="zero"):
    """
    Converts a column to numeric, coerces errors, and validates ranges.
    Prints the number of invalid rows before applying the fallback strategy.
    """
    # 1. Forced conversion in numeric ( errors become NaN )
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # 2. Validity mask creation
    valid_mask = numeric_series.notna()
    if min_val is not None:
        valid_mask &= (numeric_series >= min_val)
    if max_val is not None:
        valid_mask &= (numeric_series <= max_val)
        
    invalid_count = len(series) - valid_mask.sum()
    if invalid_count > 0:
        print(f" Warning: Column '{series.name}': Found {invalid_count} invalid/missing values (strategy: '{fallback_strategy}')")
    # ----------------------------------------------------------
        
    # 3. Test fallback strategy
    if fallback_strategy == "drop":
        fallback_val = np.nan
    elif fallback_strategy == "mean":
        fallback_val = numeric_series[valid_mask].mean()
        if pd.isna(fallback_val):
            fallback_val = 0
    else:
        fallback_val = 0
        
    # 4. Imputation
    clean_array = np.where(valid_mask, numeric_series, fallback_val)
    clean_series = pd.Series(clean_array, index=series.index)
    
    # 5. Forçage du type final
    if fallback_strategy != "drop":
        if is_integer:
            return clean_series.round().astype(int)
        else:
            return clean_series.astype(float)
            
    return clean_series

def process_date_column(series):
    """
    Validates if string dates are real, valid dates.
    Keeps the original string if valid, returns np.nan if invalid.
    """
    # 1. On teste la validité en arrière-plan (sans toucher à la série d'origine)
    test_datetime = pd.to_datetime(series, errors='coerce')
    valid_mask = test_datetime.notna()
    
    # 2. On compte et on affiche les lignes invalides
    invalid_count = len(series) - valid_mask.sum()
    if invalid_count > 0:
        print(f"  ⚠️ Column '{series.name}': Found {invalid_count} invalid/missing dates.")
    
    # 3. On garde TA string d'origine si c'est valide, sinon on met NaN pour le drop
    clean_series = np.where(valid_mask, series, np.nan)
    
    return pd.Series(clean_series, index=series.index)
