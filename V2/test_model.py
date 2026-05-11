import cv2
import mediapipe as mp
import joblib
import os
import numpy as np

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
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "img")

# Chargement Modèle
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé.")
else:
    print(f"❌ Modèle introuvable à : {MODEL_PATH}")
    exit()

# Chargement Images
gesture_images = {}
for ml_label, file_name in mapping.items():
    path = os.path.join(IMG_DIR, f"{file_name}.png")
    img = cv2.imread(path)
    if img is not None:
        gesture_images[ml_label] = img

def overlay_emoji(frame, img, x, y, size=120):
    if img is None: return
    try:
        img_res = cv2.resize(img, (size, size))
        h, w, _ = img_res.shape
        y1, y2 = max(0, y), min(frame.shape[0], y + h)
        x1, x2 = max(0, x), min(frame.shape[1], x + w)
        frame[y1:y2, x1:x2] = img_res[0:y2-y1, 0:x2-x1]
    except: pass

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        test_features = []
        for hl in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            for lm in hl.landmark:
                test_features.extend([lm.x, lm.y, lm.z])
        
        if len(result.multi_hand_landmarks) == 1:
            test_features.extend([0.0] * 63)

        if len(test_features) == 126:
            try:
                pred = model.predict([test_features])[0]
                
                # On prend la position du bout de l'index (point 8) pour placer l'affichage
                idx_x = int(result.multi_hand_landmarks[0].landmark[8].x * w_f)
                idx_y = int(result.multi_hand_landmarks[0].landmark[8].y * h_f)

                # --- REGLAGE POSITION ---
                # On met l'emoji bien au dessus (idx_y - 180)
                # On met le texte un peu plus bas que l'emoji pour qu'ils ne se touchent plus
                text_y = idx_y - 40 
                emoji_y = idx_y - 200

                # Dessiner un petit rectangle noir derrière le texte pour la lisibilité
                cv2.rectangle(frame, (idx_x - 10, text_y - 30), (idx_x + 250, text_y + 10), (0,0,0), -1)
                
                # Affichage du texte en VERT
                cv2.putText(frame, pred, (idx_x, text_y), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                
                # Affichage Emoji
                if pred in gesture_images:
                    overlay_emoji(frame, gesture_images[pred], idx_x - 60, emoji_y)
            except: pass

    cv2.imshow("TIAGO Robot Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()