import json
import os
import pandas as pd
import preprocessing as prep
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


#with open("allocine_reviews.json", "r", encoding="utf-8") as f:
#    data = json.load(f)

#df = pd.json_normalize(data)

#print(df.columns)
EXECUTER_SUR_ECHANTILLON = True  # Modifie par False pour lancer sur TOUTES les données
TAILLE_ECHANTILLON = 25000
MAX_COEURS_UTILISES = 8  
FICHIER_SAUVEGARDE = "data_processed.pkl"

def traiter_un_chunk(df_chunk):
    # On applique le traitement NLP lourd sur ce petit morceau
    colonne_avis = "text_content"
    df_chunk[colonne_avis] = df_chunk[colonne_avis].apply(
        lambda x: " ".join(prep.get_cleaning_steps(x)["No StopWords"]) if pd.notna(x) else ""
    )
    return df_chunk

if __name__ == "__main__":

    if os.path.exists(FICHIER_SAUVEGARDE):
        print(f"♻️ Sauvegarde trouvée ! Chargement rapide de '{FICHIER_SAUVEGARDE}'...")
        start_load = time.time()
        
        # Chargement ultra-rapide
        df = pd.read_pickle(FICHIER_SAUVEGARDE)
        
        print(f"✅ Données chargées avec succès en {time.time() - start_load:.2f} secondes ({len(df)} lignes).")
        print("🚀 Prêt pour l'étape suivante (Visualisation ou Machine Learning) !")
        
    else:
        print(f"🔍 Aucune sauvegarde '{FICHIER_SAUVEGARDE}' détectée. Lancement complet du traitement...")
        dossier_data = "data"

        liste_dfs = []
        temps_etape3 = 0
        if not os.path.exists(dossier_data):
            print(f"❌ Le dossier '{dossier_data}' n'existe pas. Crée-le et places-y tes fichiers JSON.")
            exit()
        
        fichiers_json = [f for f in os.listdir(dossier_data) if f.endswith(".json")]
        print(f"📂 {len(fichiers_json)} fichiers JSON trouvés dans le dossier '{dossier_data}'.")
        
        start_etape1 = time.time()
        
        for fichier in fichiers_json:
            chemin_complet = os.path.join(dossier_data, fichier)
            print(f"  -> Traitement de {fichier}...")
            # 1. Chargement du fichier JSON d'AlloCiné
            with open(chemin_complet, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 2. RÉSOLUTION DU CONFLIT : 
            # On renomme la clé "id" racine (le film) en "movie_id" directement dans le dictionnaire.
            # Ainsi, il n'y a plus aucun conflit avec le "id" de la review.
            if "id" in data:
                data["movie_id"] = data.pop("id")

            # Récupération dynamique de toutes les métadonnées (qui contient maintenant 'movie_id' et plus 'id')
            toutes_les_metadonnees = [cle for cle in data.keys() if cle != "reviews"]

            # 3. Normalisation (plus aucun conflit !)
            df = pd.json_normalize(
                data, 
                record_path=["reviews"], 
                meta=toutes_les_metadonnees
            )
            
            # 4. Renommage optionnel pour que l'ID de la review soit plus explicite
            if "id" in df.columns:
                df = df.rename(columns={"id": "review_id"})

            liste_dfs.append(df)

        if liste_dfs:
            df = pd.concat(liste_dfs, ignore_index=True)
            end_etape1 = time.time()
            temps_etape1 = end_etape1 - start_etape1
            print(f"\n✅ Assemblage réussi ! Total de lignes cumulées : {len(df)}")
        else:
            print("❌ Aucun DataFrame n'a pu être créé.")
            exit()

        # ==========================================
        # APPLICATION DE L'ÉCHANTILLONNAGE
        # ==========================================
      

        if EXECUTER_SUR_ECHANTILLON:
            if len(df) > TAILLE_ECHANTILLON:
                print(f"⚡ Mode Échantillon Activé : Sélection aléatoire de {TAILLE_ECHANTILLON} lignes sur {len(df)}...")
                # Le random_state=42 permet d'avoir toujours le même échantillon à chaque exécution
                df = df.sample(n=TAILLE_ECHANTILLON, random_state=42).reset_index(drop=True)
            else:
                print(f"ℹ️ Le DataFrame contient {len(df)} lignes, ce qui est inférieur à l'échantillon demandé ({TAILLE_ECHANTILLON}). Tout sera traité.")

        # --- NETTOYAGE DES MÉTADONNÉES AVEC TA FONCTION UNIQUE ---
        print("\nNettoyage des métadonnées (synopsis et auteurs)...")
        start_etape2 = time.time()
        if "synopsis" in df.columns:
            df["synopsis"] = df["synopsis"].apply(prep.clean_input)
            
        if "author" in df.columns:
            df["author"] = df["author"].apply(prep.clean_input)

        end_etape2 = time.time()
        temps_etape2 = end_etape2 - start_etape2
        print("✅ Nettoyage des métadonnées terminé.")

        # --- PRÉTRAITEMENT NLP LOURD (Uniquement sur l'avis) ---
        colonne_avis = "text_content" 

        if colonne_avis in df.columns:
            print("Application du preprocessing lourd sur les avis...")
            start_etape3 = time.time()

            nb_coeurs = os.cpu_count()
            print(f"💻 Processeur détecté : {nb_coeurs} cœurs logiques disponibles.")
            nb_coeurs = min(8, nb_coeurs)
            print(f"💻 Processeur détecté : {nb_coeurs} cœurs logiques utilisés.")
            
        
            # 2. Découper le DataFrame en chunks (un chunk par cœur)
            # On crée par exemple 4 fois plus de chunks que de cœurs pour équilibrer la charge
            nb_chunks = nb_coeurs * 4
            chunks = np.array_split(df, nb_chunks)
            
            # 3. Lancer le traitement en parallèle avec une barre de progression
            chunks_traites = []
            with ProcessPoolExecutor(max_workers=nb_coeurs) as executor:
                # On soumet toutes les tâches aux différents cœurs CPU
                futures = [executor.submit(traiter_un_chunk, chunk) for chunk in chunks]
                
                # tqdm nous affiche la progression au fur et à mesure que les cœurs terminent leur chunk
                for future in tqdm(futures, desc="Traitement des paquets NLP", total=len(futures)):
                    chunks_traites.append(future.result())
                    
            # 4. Re-fusionner tous les morceaux traités dans le bon ordre
            df = pd.concat(chunks_traites, ignore_index=True)
            
            end_etape3 = time.time()
            temps_etape3 = end_etape3 - start_etape3
            print("✅ Preprocessing NLP terminé !")

            # Affichage du résultat final
            print("\nColonnes finales de ton DataFrame :")
            print(df.columns.tolist())
            

        # ==========================================
        #    SAUVEGARDE EN UNE LIGNE (FORMAT PKL)
        # ==========================================
        print("\n💾 Sauvegarde du DataFrame traité en cours...")
        
        # Sauvegarde rapide en 1 ligne
        df.to_pickle("data_processed.pkl")
        
        print("✅ Sauvegarde terminée avec succès sous le nom 'data_processed.pkl' !")
        
        # ==========================================
        # AFFICHAGE DU RAPPORT DE PERFORMANCES
        # ==========================================
        print("\n" + "="*50)
        print("⏱️  RAPPORT DE TEMPS DE CALCUL (METRICS)")
        print("="*50)
        print(f"• Étape 1 (Lecture + Assemblage JSONs) : {temps_etape1:.2f} secondes")
        print(f"• Étape 2 (Nettoyage Métadonnées)      : {temps_etape2:.2f} secondes")
        if temps_etape3 > 0:
            print(f"• Étape 3 (Preprocessing NLP Lourd)    : {temps_etape3/60:.2f} minutes ({temps_etape3:.1f} secondes)")
            print(f"  ↳ Vitesse moyenne du NLP             : {len(df)/temps_etape3:.1f} avis/seconde")
        print("="*50)

    # ==========================================
    #    AFFICHAGE DU CONTENU DU TABLEAU ICI
    # ==========================================
    print("\n" + "="*80)
    print("                 AFFICHAGE GLOBAL DU TABLEAU NETTOYÉ")
    print("="*80)
    
    # On utilise une largeur de colonne maximale de 80 caractères pour que le texte 
    # des avis ne casse pas la mise en forme du tableau en console.
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 80):
        print(df)
        
    print("="*80)

    # ==========================================
    #    AFFICHAGE DE LA VERSION RÉDUITE (REVIEWS UNIQUEMENT)
    # ==========================================
    print("\n" + "="*80)
    print("         AFFICHAGE DU TABLEAU RÉDUIT (INFOS DES REVIEWS UNIQUEMENT)")
    print("="*80)

    # 1. On définit la liste des colonnes uniques à chaque avis
    colonnes_reviews = [
        "movie_id",
        "review_id", 
        "author", 
        "date", 
        "rating", 
        "likes", 
        "dislikes",
        "text_content", 
    ]

    # On filtre pour ne garder que les colonnes qui existent réellement dans notre df
    # (sécurité au cas où une colonne manquerait)
    colonnes_existantes = [col for col in colonnes_reviews if col in df.columns]

    # 2. Création du DataFrame réduit
    df_reviews_only = df[colonnes_existantes]

    # 3. Affichage propre en console
    with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.max_colwidth', 50):
        print(df_reviews_only)

    print("="*80)

       

"""
if __name__ == "__main__":
    exemple_review = "Les acteurs étaient parfaits... &amp; j'ai adoré la fin ! http://allocine.fr"
    res = prep.get_cleaning_steps(exemple_review)

    for step, result in res.items():
        print(f"{step} :\n   {result}\n")

        """