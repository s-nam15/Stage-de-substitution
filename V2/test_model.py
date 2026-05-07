import cv2
import mediapipe as mp
import joblib
import os

# ===== CONFIGURATION DES CHEMINS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "img")

# ===== MAPPING (Identique à ton dictionnaire précédent) =====
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

# ===== CHARGEMENT DU MODÈLE =====
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
else:
    print(f"❌ Erreur : {MODEL_PATH} introuvable.")
    exit()

# ===== CHARGEMENT DES EMOJIS =====
gesture_images = {}
if os.path.exists(IMG_DIR):
    for ml_name, file_name in mapping.items():
        img_path = os.path.join(IMG_DIR, f"{file_name}.png")
        img = cv2.imread(img_path)
        if img is not None:
            gesture_images[ml_name] = img
    print(f"🖼️ {len(gesture_images)} images d'emojis chargées.")
else:
    print(f"❌ Dossier {IMG_DIR} introuvable.")
    exit()

# ===== INITIALISATION MEDIAPIPE (PASSAGE À 2 MAINS) =====
mp_hands = mp.solutions.hands
# On change max_num_hands à 2
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def overlay_image(frame, img, x, y, size=120):
    if img is None: return
    img_resized = cv2.resize(img, (size, size))
    h, w, _ = img_resized.shape
    # Sécurité pour ne pas sortir du cadre
    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)
    img_portion = img_resized[0:y2-y1, 0:x2-x1]
    frame[y1:y2, x1:x2] = img_portion

# ===== BOUCLE PRINCIPALE =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        # On boucle sur chaque main détectée (jusqu'à 2)
        for hand_landmarks in result.multi_hand_landmarks:
            # 1. Dessin du squelette
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Extraction des caractéristiques
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            # 3. Prédiction
            try:
                prediction = model.predict([features])[0]
                gesture_name = str(prediction).upper()
            except:
                gesture_name = "ERREUR"

            # 4. Calcul de la position pour l'emoji (au-dessus de la main)
            # On prend le point 0 (poignet) pour positionner l'emoji
            x_pos = int(hand_landmarks.landmark[0].x * w_frame)
            y_pos = int(hand_landmarks.landmark[0].y * h_frame) - 150 # On l'affiche au-dessus

            # 5. Affichage du texte et de l'emoji
            cv2.putText(frame, gesture_name, (x_pos, y_pos + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if gesture_name in gesture_images:
                overlay_image(frame, gesture_images[gesture_name], x_pos - 60, y_pos)

    cv2.imshow("Detection TIAGO Multi-Mains", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()