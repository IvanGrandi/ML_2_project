# logger.py
import pandas as pd

def show_time_report(step1_time, step2_time, step3_time, total_rows):
    """Displays a clean execution performance and timing report."""
    print("\n" + "="*50)
    print("⏱️  COMPUTATION TIME REPORT (METRICS)")
    print("="*50)
    print(f"• Step 1 (Read + Merge JSONs)       : {step1_time:.2f} seconds")
    print(f"• Step 2 (Metadata Cleaning)        : {step2_time:.2f} seconds")
    if step3_time > 0:
        print(f"• Step 3 (Heavy NLP Preprocessing)  : {step3_time/60:.2f} minutes ({step3_time:.1f} seconds)")
        print(f"  ↳ Average NLP Processing Speed    : {total_rows/step3_time:.1f} reviews/second")
    print("="*50)

def show_visualizations(df):
    """Displays DataFrame previews in the console."""
    print("\n" + "="*80)
    print("                 FULL VIEW OF THE CLEANED DATAFRAME")
    print("="*80)
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 80):
        print(df)
    print("="*80)

    print("\n" + "="*80)
    print("         REDUCED VIEW (REVIEWS INFORMATION ONLY)")
    print("="*80)

    review_columns = ["movie_id", "review_id", "author", "date", "rating", "likes", "dislikes", "text_content"]
    existing_columns = [col for col in review_columns if col in df.columns]
    
    # Create the subset DataFrame
    df_reviews_only = df[existing_columns]

    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
        print(df_reviews_only)
    print("="*80)