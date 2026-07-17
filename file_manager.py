# file_manager.py
import json
import os
import time
import pandas as pd

def charger_fichiers_json(dossier_data="data"):
    """Charge et assemble tous les fichiers JSON du dossier spécifié."""
    if not os.path.exists(dossier_data):
        print(f"❌ Le dossier '{dossier_data}' n'existe pas.")
        return None
        
    fichiers_json = [f for f in os.listdir(dossier_data) if f.endswith(".json")]

    if not fichiers_json:
        print(f"⚠️ Aucun fichier JSON trouvé dans '{dossier_data}'.")
        return None
    else: 
        print(f"📂 {len(fichiers_json)} fichiers JSON trouvés dans le dossier '{dossier_data}'.")
        
    liste_dfs = []
    for fichier in fichiers_json:
        chemin_complet = os.path.join(dossier_data, fichier)
        print(f"  -> Traitement de {fichier}...")
        
        with open(chemin_complet, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Résolution du conflit de nommage de l'ID racine
        if "id" in data:
            data["movie_id"] = data.pop("id")

        toutes_les_metadonnees = [cle for cle in data.keys() if cle != "reviews"]

        df_temp = pd.json_normalize(
            data, 
            record_path=["reviews"], 
            meta=toutes_les_metadonnees
        )
        
        if "id" in df_temp.columns:
            df_temp = df_temp.rename(columns={"id": "review_id"})

        liste_dfs.append(df_temp)

    if liste_dfs:
        df = pd.concat(liste_dfs, ignore_index=True)
        print(f"\n✅ Assemblage réussi ! Total de lignes cumulées : {len(df)}")
        return df
    
    print("❌ Aucun DataFrame n'a pu être créé.")   
    return None

def charger_sauvegarde(chemin_pickle):
    """Charge un DataFrame depuis un fichier Pickle s'il existe."""
    if os.path.exists(chemin_pickle):
        print(f"♻️ Sauvegarde trouvée ! Chargement rapide de '{chemin_pickle}'...")
        start_load = time.time()
        df = pd.read_pickle(chemin_pickle)
        print(f"✅ Données chargées avec succès en {time.time() - start_load:.2f} secondes ({len(df)} lignes).")
        return df
    return None

def sauvegarder_dataframe(df, chemin_pickle):
    """Sauvegarde le DataFrame au format Pickle."""
    print(f"\n💾 Sauvegarde du DataFrame traité dans '{chemin_pickle}'...")
    df.to_pickle(chemin_pickle)
    print("✅ Sauvegarde terminée avec succès !")