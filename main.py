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

EXECUTER_SUR_ECHANTILLON = True  # Modifie par False pour lancer sur TOUTES les données
TAILLE_ECHANTILLON = 25000
MAX_COEURS_UTILISES = 6  
FICHIER_SAUVEGARDE = "data_processed.pkl"

if __name__ == "__main__":

    df = fm.charger_sauvegarde(FICHIER_SAUVEGARDE)

    if df is not None:
        print("🚀 Prêt pour l'étape suivante (Visualisation ou Machine Learning) !")
        
    else:
        print(f"🔍 Aucune sauvegarde '{FICHIER_SAUVEGARDE}' détectée. Lancement complet du traitement...")
        
        start_etape1 = time.time()
        df = fm.charger_fichiers_json(dossier_data="data")
        if df is None:
            exit()
        
        end_etape1 = time.time()
        temps_etape1 = end_etape1 - start_etape1

        # ==========================================
        # APPLICATION DE L'ÉCHANTILLONNAGE
        # ==========================================
        df = pipe.appliquer_echantillonnage(df, EXECUTER_SUR_ECHANTILLON, TAILLE_ECHANTILLON)
        
        # --- NETTOYAGE DES MÉTADONNÉES AVEC TA FONCTION UNIQUE ---
        start_etape2 = time.time()
        df = pipe.nettoyer_metadonnees(df)

        end_etape2 = time.time()
        temps_etape2 = end_etape2 - start_etape2

        start_etape3 = time.time()
        df = pipe.executer_nlp_parallele(df, MAX_COEURS_UTILISES)       
        end_etape3 = time.time()
        temps_etape3 = end_etape3 - start_etape3

        # Affichage du résultat final
        print("\nColonnes finales de ton DataFrame :")
        print(df.columns.tolist())
            

        # ==========================================
        #    SAUVEGARDE EN UNE LIGNE (FORMAT PKL)
        # ==========================================
        
        fm.sauvegarder_dataframe(df, FICHIER_SAUVEGARDE)


        # ==========================================
        # AFFICHAGE DU RAPPORT DE PERFORMANCES
        # ==========================================
       
        # 6. Génération et affichage du rapport de temps
        logger.afficher_rapport_temps(temps_etape1, temps_etape2, temps_etape3, len(df))

    # ==========================================
    #    AFFICHAGE DU CONTENU DU TABLEAU ICI
    # ==========================================
    
    logger.afficher_visualisations(df)