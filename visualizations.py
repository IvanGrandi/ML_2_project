import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Global plot style configuration
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_rating_distribution(df):
    """
    Plots a histogram showing the distribution of ratings.
    """
    if "rating" not in df.columns:
        print("⚠️ 'rating' column not found for visualization.")
        return
        
    plt.figure()
    # Using histplot with a KDE (Kernel Density Estimate) to clearly see the trend
    sns.histplot(data=df, x="rating", bins=10, kde=True, color="skyblue", edgecolor="black")
    
    plt.title("Rating Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Rating", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    
    plt.tight_layout()
    plt.show()  # This command opens the window with the plot!

def plot_engagement_distribution(df):
    """
    Plots the distribution of Likes and Dislikes using Boxplots.
    Use a logarithmic scale if the data is highly skewed.
    """
    engagement_cols = [c for c in ["likes", "dislikes"] if c in df.columns]
    if not engagement_cols:
        print("⚠️ No engagement columns ('likes'/'dislikes') found for visualization.")
        return

    plt.figure()
    # A side-by-side boxplot to compare distributions
    sns.boxplot(data=df[engagement_cols], palette="Set2")
    
    plt.title("Likes and Dislikes Distribution (Boxplot)", fontsize=14, fontweight='bold')
    plt.ylabel("Number of Interactions (Linear Scale)", fontsize=12)
    
    # If you have massive outliers (e.g., 0 likes for 95% of rows and 10,000 for 1%), 
    # uncomment the line below to switch to a log scale to make it readable:
    # plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """
    Computes and displays the correlation matrix (Heatmap) for all numerical columns.
    """
    # Keep only relevant numerical columns (exclude identifiers like movie_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "movie_id" in numeric_cols:
        numeric_cols.remove("movie_id")
        
    if len(numeric_cols) < 2:
        print("⚠️ Not enough numeric columns to compute a correlation matrix.")
        return

    plt.figure(figsize=(8, 6))
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    # Generating the Heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,          # Displays the numerical values inside the cells
        cmap="coolwarm",     # Color palette from blue (negative) to red (positive)
        fmt=".2f",           # Rounds to 2 decimal places
        linewidths=0.5, 
        vmin=-1, vmax=1      # Correlation boundary limits
    )
    
    plt.title("Correlation Matrix of Numerical Variables", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_temporal_trends(df):
    """
    Displays the timeline trends for review volumes and average ratings.
    """
    # Check if at least one valid date column is present
    date_col = next((c for c in ["date", "release_date"] if c in df.columns), None)
    if not date_col or "rating" not in df.columns:
        print("⚠️ Missing date or rating column for temporal visualization.")
        return

    # Temporarily convert to datetime just to cleanly extract the year
    temp_years = pd.to_datetime(df[date_col], errors='coerce').dt.year
    
    # Create a temporary DataFrame grouped by year
    yearly_data = df.groupby(temp_years).agg(
        volume=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # Rename the indexed column
    yearly_data.rename(columns={date_col: 'year'}, inplace=True)
    # Filter out anomalous/unrealistic years if the scraper caught a bug
    yearly_data = yearly_data[yearly_data['year'] > 1900]

    # Create a dual-axis chart (Volume vs. Average Rating)
    fig, ax1 = plt.subplots()

    # Axis 1: Review Volume (Bar Chart)
    sns.barplot(data=yearly_data, x="year", y="volume", color="gainsboro", ax=ax1, edgecolor="black")
    ax1.set_title("Temporal Trends of Reviews and Average Ratings", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Number of Reviews", color="gray", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="gray")
    plt.xticks(rotation=45)

    # Axis 2: Average Rating (Line Chart)
    ax2 = ax1.twinx()
    sns.lineplot(data=yearly_data, x=ax1.get_xticks(), y="avg_rating", color="crimson", marker="o", linewidth=2, ax=ax2)
    ax2.set_ylabel("Average Rating", color="crimson", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="crimson")
    ax2.set_ylim(0, 5)  # Locks the rating axis strictly between 0 and 5

    plt.tight_layout()
    plt.show()

def plot_text_length_distribution(df, text_column="synopsis"):
    """
    Plots the distribution of word counts in the cleaned text column.
    """
    if text_column not in df.columns:
        print(f"⚠️ '{text_column}' column not found for text length visualization.")
        return

    # Calculate the word count for each row (handling empty strings safely)
    word_counts = df[text_column].fillna("").apply(lambda x: len(str(x).split()))

    plt.figure()
    sns.histplot(word_counts, bins=30, kde=True, color="purple", edgecolor="black")
    plt.title(f"Text Length Distribution ({text_column})", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_global_wordcloud(df, text_column="text_content"):
    """
    Generates a global word cloud exclusively from user reviews.
    """
    if text_column not in df.columns:
        print(f"⚠️ '{text_column}' column not found for global Word Cloud.")
        return

    print(f"☁️ Generating global Word Cloud for '{text_column}'...")
    
    # Combine all valid reviews into one massive string
    all_text = " ".join(df[text_column].fillna("").astype(str))

    if len(all_text.strip()) == 0:
        print("⚠️ No text available to generate the Word Cloud.")
        return

    # Global Word Cloud configuration
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color="white", 
        max_words=150,
        collocations=False,
        colormap="viridis"
    ).generate(all_text)

    # Display the plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Global Word Cloud of User Reviews", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_comparative_wordclouds(df, text_column="lemmatized"):
    """
    Generates two comparative Word Clouds: highly rated vs. poorly rated movies.
    """
    if text_column not in df.columns or "rating" not in df.columns:
        print("⚠️ Missing text or rating column for Word Clouds.")
        return

    # Separate texts based on user ratings
    good_reviews = " ".join(df[df["rating"] >= 4][text_column].fillna("").astype(str))
    bad_reviews = " ".join(df[df["rating"] <= 2][text_column].fillna("").astype(str))

    # Base configuration for the WordCloud generator
    wc_config = {
        "width": 800, 
        "height": 400, 
        "background_color": "white", 
        "max_words": 100,
        "collocations": False  # Prevents duplicating phrases glued together
    }

    plt.figure(figsize=(12, 6))
    
    # 1. WordCloud for Top Rated
    if len(good_reviews.strip()) > 0:
        wordcloud_good = WordCloud(**wc_config, colormap="Greens").generate(good_reviews)
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_good, interpolation="bilinear")
        plt.title("Keywords from HIGHLY Rated Movies (★ >= 4)", fontsize=14, fontweight='bold', color="green")
        plt.axis("off")

    # 2. WordCloud for Low Rated
    if len(bad_reviews.strip()) > 0:
        wordcloud_bad = WordCloud(**wc_config, colormap="Reds").generate(bad_reviews)
        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_bad, interpolation="bilinear")
        plt.title("Keywords from POORLY Rated Movies (★ <= 2)", fontsize=14, fontweight='bold', color="red")
        plt.axis("off")

    plt.tight_layout()
    plt.show()