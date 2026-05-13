import cv2
import mediapipe as mp
import csv
import os

# ===== CONFIGURATION =====
mapping = {
    "FINGERS_JOINED": "bout_des_doigts_joints",
    "HORNS": "cornes_avec_les_doigts",
    "MIDDLE_FINGER": "doigt_dhonneur",
    "CROSSED_FINGERS": "doigts_croises",
    "POINT_UP": "index_pointant_vers_le_haut",
    "POINT_AT_USER": "index_pointant_vers_lutilisateur",
    "LOVE_YOU": "signe_je_taime",
    "POINT_RIGHT": "main_avec_index_pointant_a_droite",
    "POINT_LEFT": "main_avec_index_pointant_a_gauche",
    "POINT_DOWN": "main_avec_index_pointant_vers_le_bas",
    "POINT_UP_HAND": "main_avec_index_pointant_vers_le_haut",
    "CROSSED_THUMB_INDEX": "main_avec_index_et_pouce_croises",
    "PALM_DOWN": "main_paume_vers_le_bas",
    "PALM_UP": "main_paume_vers_le_haut",
    "RAISED_HAND": "main_levee",
    "SPREAD_HAND": "main_levee_doigts_ecartes",
    "PRAY_HANDS": "mains_en_priere",
    "RAISED_HANDS": "mains_levees",
    "OPEN_HANDS": "mains_ouvertes",
    "HEART_HANDS": "mains_qui_forment_un_coeur",
    "OK": "ok",
    "PALMS_TOGETHER": "paume_contre_paume_doigts_vers_le_haut",
    "FIST_RIGHT": "poing_a_droite",
    "FIST_LEFT": "poing_a_gauche",
    "FRONT_FIST": "poing_de_face",
    "RAISED_FIST": "poing_leve",
    "HANDSHAKE": "poignee_de_main",
    "PINCHED_FINGERS": "pouce_et_index_rapproches",
    "THUMBS_DOWN": "pouce_vers_le_bas",
    "THUMBS_UP": "pouce_vers_le_haut",
    "VULCAN": "salut_vulcain",
    "CALL_ME": "signe_appel_telephonique_avec_les_doigts",
    "PEACE": "v_de_la_victoire",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Sélection du label
valid_labels = sorted(list(mapping.keys()))
print("\n--- GESTES DISPONIBLES ---")
print(", ".join(valid_labels))
label = input("\nEntrez le nom du geste à collecter : ").upper()

if label not in mapping:
    print("Erreur: Label non présent dans le mapping.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with open(DATASET_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    print(f"🚀 Collecte lancée pour : {label}. Appuyez sur 'S' pour sauvegarder.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks_to_save = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks_to_save.extend([lm.x, lm.y, lm.z])
            
            # Compléter avec des zéros si une seule main est détectée
            if len(result.multi_hand_landmarks) == 1:
                landmarks_to_save.extend([0.0] * 63)

        # Affichage status
        nb_mains = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        cv2.putText(frame, f"Label: {label} | Mains: {nb_mains}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Collecte Multi-Mains", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(landmarks_to_save) == 126:
            writer.writerow(landmarks_to_save + [label])
            print(f"✅ Enregistré (126 points)")
        elif key == 27: # ESC
            break

cap.release()
cv2.destroyAllWindows()

#test1