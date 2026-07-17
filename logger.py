# logger.py
import pandas as pd

def afficher_rapport_temps(temps_etape1, temps_etape2, temps_etape3, total_lignes):
    """Affiche un joli rapport de performance."""
    print("\n" + "="*50)
    print("⏱️  RAPPORT DE TEMPS DE CALCUL (METRICS)")
    print("="*50)
    print(f"• Étape 1 (Lecture + Assemblage JSONs) : {temps_etape1:.2f} secondes")
    print(f"• Étape 2 (Nettoyage Métadonnées)      : {temps_etape2:.2f} secondes")
    if temps_etape3 > 0:
        print(f"• Étape 3 (Preprocessing NLP Lourd)    : {temps_etape3/60:.2f} minutes ({temps_etape3:.1f} secondes)")
        print(f"  ↳ Vitesse moyenne du NLP             : {total_lignes/temps_etape3:.1f} avis/seconde")
    print("="*50)

def afficher_visualisations(df):
    """Affiche les aperçus de tableaux en console."""
    print("\n" + "="*80)
    print("                 AFFICHAGE GLOBAL DU TABLEAU NETTOYÉ")
    print("="*80)
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 80):
        print(df)
    print("="*80)

    print("\n" + "="*80)
    print("         AFFICHAGE DU TABLEAU RÉDUIT (INFOS DES REVIEWS UNIQUEMENT)")
    print("="*80)

    colonnes_reviews = ["movie_id", "review_id", "author", "date", "rating", "likes", "dislikes", "text_content"]
    colonnes_existantes = [col for col in colonnes_reviews if col in df.columns]
    
    # 2. Création du DataFrame réduit
    df_reviews_only = df[colonnes_existantes]

    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
        print(df_reviews_only)
    print("="*80)