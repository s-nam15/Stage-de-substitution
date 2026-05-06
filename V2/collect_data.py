import cv2
import mediapipe as mp
import csv
import os

# ===== CONFIGURATION DU MAPPING (Trié par nom de fichier) =====
mapping = {
    "BRAVO": "applaudissements",
    "JOINTS": "bout_des_doigts_joints",
    "CORNES": "cornes_avec_les_doigts",
    "HONNEUR": "doigt_dhonneur",
    "DOIGTS_CROISES": "doigts_croises",
    "DOS_MAIN": "dos_de_main_levee",
    "INDEX_HAUT_V2": "index_pointant_vers_le_haut",
    "INDEX_USER": "index_pointant_vers_lutilisateur",
    "I_LOVE_YOU": "signe_je_taime",
    "INDEX_DROITE": "main_avec_index_pointant_a_droite",
    "INDEX_GAUCHE": "main_avec_index_pointant_a_gauche",
    "INDEX_BAS": "main_avec_index_pointant_vers_le_bas",
    "INDEX_HAUT_V1": "main_avec_index_pointant_vers_le_haut",
    "INDEX_POUCE_CROISE": "main_avec_index_et_pouce_croises",
    "PAUME_BAS": "main_paume_vers_le_bas",
    "PAUME_HAUT": "main_paume_vers_le_haut",
    "POUSSE_DROITE": "main_qui_pousse_vers_la_droite",
    "POUSSE_GAUCHE": "main_qui_pousse_vers_la_gauche",
    "ECRIT": "main_qui_ecrit",
    "MAIN_LEVEE": "main_levee",
    "MAIN_ECARTEE": "main_levee_doigts_ecartes",
    "VERS_GAUCHE": "main_vers_la_gauche",
    "VERS_DROITE": "main_vers_la_droite",
    "MAINS_EN_PRIERE": "mains_en_priere",
    "MAINS_LEVEES": "mains_levees",
    "OUVERTES": "mains_ouvertes",
    "COEUR": "mains_qui_forment_un_coeur",
    "OK": "ok",
    "PRIERE_V1": "paume_contre_paume_doigts_vers_le_haut",
    "POING_DROITE": "poing_a_droite",
    "POING_GAUCHE": "poing_a_gauche",
    "POING_FACE": "poing_de_face",
    "POING_LEVE": "poing_leve",
    "POIGNEE": "poignee_de_main",
    "RAPPROCHES": "pouce_et_index_rapproches",
    "DISLIKE": "pouce_vers_le_bas",
    "LIKE": "pouce_vers_le_haut",
    "VULCAIN": "salut_vulcain",
    "SELFIE": "selfie",
    "TELEPHONE": "signe_appel_telephonique_avec_les_doigts",
    "SIGNE_MAIN": "signe_de_la_main",
    "PEACE": "v_de_la_victoire",
    "VERNIS": "vernis_a_ongles"
}

# ===== CONFIGURATION DES CHEMINS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# ===== INTERFACE DE CHOIX DU LABEL =====
valid_labels = sorted(list(mapping.keys()))
print("\n" + "="*50)
print("LISTE DES GESTES DISPONIBLES :")
# Affichage par colonnes pour plus de clarté
for i in range(0, len(valid_labels), 3):
    row = valid_labels[i:i+3]
    print("  ".join(f"{label:<20}" for label in row))
print("="*50 + "\n")

while True:
    label = input("Entrez le NOM DU GESTE à enregistrer (ex: PEACE, OK...) : ").upper()
    if label in valid_labels:
        print(f"✅ Prêt à enregistrer pour : {label} ({mapping[label]})")
        break
    else:
        print(f"❌ '{label}' n'est pas dans la liste. Réessayez.")

# ===== INITIALISATION MEDIAPIPE & CAMERA =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ===== BOUCLE DE COLLECTE =====
with open(DATASET_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    print("\n[INFO] Appuyez sur 'S' pour sauvegarder une pose.")
    print("[INFO] Appuyez sur 'ECHAP' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Dessin du squelette
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Capture des données
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                cv2.putText(frame, f"Geste: {label}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, "Appuyez sur 'S' pour ENREGISTRER", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    row = landmarks + [label]
                    writer.writerow(row)
                    print(f"Enregistré : {label}")

        cv2.imshow("Collecte de donnees - Robot TIAGO", frame)

        # Quitter avec Echap (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print(f"\n[FIN] Les données ont été ajoutées dans : {DATASET_PATH}")