import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Configuration globale du style des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def plot_rating_distribution(df):
    """
    Affiche l'histogramme de la distribution des notes (ratings).
    """
    if "rating" not in df.columns:
        print("⚠️ 'rating' column not found for visualization.")
        return
        
    plt.figure()
    # On utilise un histplot avec un KDE (courbe de densité) pour bien voir la tendance
    sns.histplot(data=df, x="rating", bins=10, kde=True, color="skyblue", edgecolor="black")
    
    plt.title("Distribution des Notes (Ratings)", fontsize=14, fontweight='bold')
    plt.xlabel("Note", fontsize=12)
    plt.ylabel("Nombre d'avis", fontsize=12)
    
    plt.tight_layout()
    plt.show() # C'est cette commande qui ouvre la fenêtre avec le graphique !

def plot_engagement_distribution(df):
    """
    Affiche la distribution des Likes et Dislikes via des Boxplots.
    Utilise une échelle logarithmique si les données sont très étalées.
    """
    engagement_cols = [c for c in ["likes", "dislikes"] if c in df.columns]
    if not engagement_cols:
        print("⚠️ No engagement columns ('likes'/'dislikes') found for visualization.")
        return

    plt.figure()
    # Un boxplot côte à côte pour comparer les distributions
    sns.boxplot(data=df[engagement_cols], palette="Set2")
    
    plt.title("Distribution des Likes et Dislikes (Boxplot)", fontsize=14, fontweight='bold')
    plt.ylabel("Nombre d'interactions (Échelle linéaire)", fontsize=12)
    
    # Si tu as d'immenses valeurs (ex: 0 like pour 95% des films et 10 000 pour 1%), 
    # décommente la ligne ci-dessous pour passer en échelle log et y voir quelque chose :
    # plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """
    Calcule et affiche la matrice de corrélation (Heatmap) pour toutes les colonnes numériques.
    """
    # On ne garde que les colonnes numériques qui ont du sens (on exclut movie_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "movie_id" in numeric_cols:
        numeric_cols.remove("movie_id")
        
    if len(numeric_cols) < 2:
        print("⚠️ Not enough numeric columns to compute a correlation matrix.")
        return

    plt.figure(figsize=(8, 6))
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    # Génération de la Heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,          # Affiche les valeurs numériques dans les cases
        cmap="coolwarm",     # Palette de couleur du bleu (négatif) au rouge (positif)
        fmt=".2f",           # Arrondi à 2 décimales
        linewidths=0.5, 
        vmin=-1, vmax=1      # Les limites de la corrélation
    )
    
    plt.title("Matrice de Corrélation des variables numériques", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_temporal_trends(df):
    """
    Affiche l'évolution du volume d'avis et de la note moyenne dans le temps.
    """
    # On vérifie si on a au moins une colonne date valide
    date_col = next((c for c in ["date", "release_date"] if c in df.columns), None)
    if not date_col or "rating" not in df.columns:
        print("⚠️ Missing date or rating column for temporal visualization.")
        return

    # On convertit temporairement en datetime juste pour extraire l'année proprement
    temp_years = pd.to_datetime(df[date_col], errors='coerce').dt.year
    
    # Création d'un DataFrame temporaire groupé par année
    yearly_data = df.groupby(temp_years).agg(
        volume=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # On renomme la colonne indexée
    yearly_data.rename(columns={date_col: 'year'}, inplace=True)
    # On filtre les années aberrantes si le scraper a eu un bug
    yearly_data = yearly_data[yearly_data['year'] > 1900]

    # Création d'un graphique à double axe (Volume vs Note Moyenne)
    fig, ax1 = plt.subplots()

    # Axe 1 : Volume d'avis (Barres)
    sns.barplot(data=yearly_data, x="year", y="volume", color="gainsboro", ax=ax1, edgecolor="black")
    ax1.set_title("Évolution temporelle des avis et des notes moyennes", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Année", fontsize=12)
    ax1.set_ylabel("Nombre d'avis", color="gray", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="gray")
    plt.xticks(rotation=45)

    # Axe 2 : Note moyenne (Ligne)
    ax2 = ax1.twinx()
    sns.lineplot(data=yearly_data, x=ax1.get_xticks(), y="avg_rating", color="crimson", marker="o", linewidth=2, ax=ax2)
    ax2.set_ylabel("Note Moyenne", color="crimson", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="crimson")
    ax2.set_ylim(0, 5) # Bloque l'axe des notes entre 0 et 5

    plt.tight_layout()
    plt.show()

def plot_text_length_distribution(df, text_column="synopsis"):
    """
    Affiche la distribution du nombre de mots dans les textes nettoyés.
    """
    if text_column not in df.columns:
        print(f"⚠️ '{text_column}' column not found for text length visualization.")
        return

    # Calcul du nombre de mots pour chaque ligne (on gère les chaînes vides)
    word_counts = df[text_column].fillna("").apply(lambda x: len(str(x).split()))

    plt.figure()
    sns.histplot(word_counts, bins=30, kde=True, color="purple", edgecolor="black")
    plt.title(f"Distribution de la longueur des textes ({text_column})", fontsize=14, fontweight='bold')
    plt.xlabel("Nombre de mots", fontsize=12)
    plt.ylabel("Nombre de documents", fontsize=12)
    
    plt.tight_layout()
    plt.show()

def plot_global_wordcloud(df, text_column="text_content"):
    """
    Génère un nuage de mots global uniquement sur les avis des spectateurs.
    """
    if text_column not in df.columns:
        print(f"⚠️ '{text_column}' column not found for global Word Cloud.")
        return

    print(f"☁️ Generating global Word Cloud for '{text_column}'...")
    
    # On rassemble tous les avis valides en un seul grand texte
    all_text = " ".join(df[text_column].fillna("").astype(str))

    if len(all_text.strip()) == 0:
        print("⚠️ No text available to generate the Word Cloud.")
        return

    # Configuration du nuage de mots global
    wordcloud = WordCloud(
        width=1000, 
        height=500, 
        background_color="white", 
        max_words=150,
        collocations=False,
        colormap="viridis"
    ).generate(all_text)

    # Affichage
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"Nuage de mots global des avis spectateurs", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_comparative_wordclouds(df, text_column="text_content"):
    """
    Génère deux Nuages de Mots comparatifs : les films bien notés vs mal notés.
    """
    if text_column not in df.columns or "rating" not in df.columns:
        print("⚠️ Missing text or rating column for Word Clouds.")
        return

    # Séparation des textes selon la note
    good_reviews = " ".join(df[df["rating"] >= 4][text_column].fillna("").astype(str))
    bad_reviews = " ".join(df[df["rating"] <= 2][text_column].fillna("").astype(str))

    # Configuration de base du générateur de WordCloud
    wc_config = {
        "width": 800, 
        "height": 400, 
        "background_color": "white", 
        "max_words": 100,
        "collocations": False # Évite de dupliquer les mots collés
    }

    # 1. WordCloud des Tops
    plt.figure(figsize=(12, 6))
    if len(good_reviews.strip()) > 0:
        wordcloud_good = WordCloud(**wc_config, colormap="Greens").generate(good_reviews)
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_good, interpolation="bilinear")
        plt.title("Mots-clés des films BIEN notés (★ >= 4)", fontsize=14, fontweight='bold', color="green")
        plt.axis("off")

    # 2. WordCloud des Flops
    if len(bad_reviews.strip()) > 0:
        wordcloud_bad = WordCloud(**wc_config, colormap="Reds").generate(bad_reviews)
        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_bad, interpolation="bilinear")
        plt.title("Mots-clés des films MAL notés (★ <= 2)", fontsize=14, fontweight='bold', color="red")
        plt.axis("off")

    plt.tight_layout()
    plt.show()