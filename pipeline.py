# pipeline.py
import os
import numpy as np
import pandas as pd
import preprocessing as prep
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def appliquer_echantillonnage(df, executer_sur_echantillon, taille_echantillon):
    """Réduit aléatoirement la taille du DataFrame si l'option est activée."""
    if executer_sur_echantillon and len(df) > taille_echantillon:
        print(f"⚡ Mode Échantillon Activé : Sélection aléatoire de {taille_echantillon} lignes sur {len(df)}...")
        return df.sample(n=taille_echantillon, random_state=42).reset_index(drop=True)
    elif executer_sur_echantillon:
        print(f"ℹ️ Le DataFrame contient {len(df)} lignes, inférieur à l'échantillon ({taille_echantillon}). Tout sera traité.")
    return df

def nettoyer_metadonnees(df):
    """Nettoie rapidement le synopsis et l'auteur."""
    print("\nNettoyage des métadonnées (synopsis et auteurs)...")
    if "synopsis" in df.columns:
        df["synopsis"] = df["synopsis"].apply(prep.clean_input)
    if "author" in df.columns:
        df["author"] = df["author"].apply(prep.clean_input)
    print("✅ Nettoyage des métadonnées terminé.")
    return df

def _traiter_un_chunk(df_chunk):
    """Fonction interne pour nettoyer les avis d'un seul chunk (utilisée en parallèle)."""
    colonne_avis = "text_content"
    df_chunk[colonne_avis] = df_chunk[colonne_avis].apply(
        lambda x: " ".join(prep.get_cleaning_steps(x)["No StopWords"]) if pd.notna(x) else ""
    )
    return df_chunk

def executer_nlp_parallele(df, max_coeurs):
    """Découpe le DataFrame et applique le NLP lourd en utilisant le multi-processing."""
    colonne_avis = "text_content"
    if colonne_avis not in df.columns:
        print("⚠️ La colonne d'avis est absente, étape de NLP lourd ignorée.")
        return df

    print("Application du preprocessing lourd sur les avis...")
    nb_coeurs_dispos = os.cpu_count()
    nb_coeurs_utilises = min(max_coeurs, nb_coeurs_dispos)
    
    print(f"💻 Processeur détecté : {nb_coeurs_dispos} cœurs logiques disponibles.")
    print(f"💻 Processeur utilisé  : {nb_coeurs_utilises} cœurs logiques affectés.")
    
    # Découpage en chunks
    nb_chunks = nb_coeurs_utilises * 4
    chunks = np.array_split(df, nb_chunks)
    
    chunks_traites = []
    with ProcessPoolExecutor(max_workers=nb_coeurs_utilises) as executor:
        futures = [executor.submit(_traiter_un_chunk, chunk) for chunk in chunks]
        
        for future in tqdm(futures, desc="Traitement des paquets NLP", total=len(futures)):
            chunks_traites.append(future.result())
            
    df_final = pd.concat(chunks_traites, ignore_index=True)
    print("✅ Preprocessing NLP terminé !")
    return df_final