import cv2
import mediapipe as mp
import joblib
import os

# ===== CONFIGURATION DES CHEMINS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
IMG_DIR = os.path.join(BASE_DIR, "img")

# ===== DICTIONNAIRE DE CORRESPONDANCE (MAPPING) =====
# À gauche : Le nom que tu tapes dans le terminal lors de la collecte (labels)
# À droite : Le nom exact du fichier .png dans ton dossier /img (sans le .png)
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
    print(f"❌ Erreur : Le fichier {MODEL_PATH} est introuvable !")
    exit()

# ===== CHARGEMENT DES EMOJIS =====
gesture_images = {}
if os.path.exists(IMG_DIR):
    for ml_name, file_name in mapping.items():
        img_path = os.path.join(IMG_DIR, f"{file_name}.png")
        img = cv2.imread(img_path)
        if img is not None:
            gesture_images[ml_name] = img
        else:
            print(f"⚠️ Image introuvable : {file_name}.png")
    print(f"🖼️ {len(gesture_images)} images d'emojis chargées.")
else:
    print(f"❌ Erreur : Le dossier {IMG_DIR} est introuvable !")
    exit()

# ===== INITIALISATION MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===== FONCTION OVERLAY (Affiche l'emoji) =====
def overlay_image(frame, img, x, y, size=150):
    if img is None:
        return
    img_resized = cv2.resize(img, (size, size))
    h, w, _ = img_resized.shape
    
    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        frame[y:y+h, x:x+w] = img_resized

# ===== BOUCLE PRINCIPALE =====
cap = cv2.VideoCapture(0)

print("🚀 Lancement de la reconnaissance... Appuyez sur ECHAP pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            try:
                # Prédiction du label (ex: "PEACE")
                prediction = model.predict([features])[0]
                gesture_name = str(prediction).upper()
            except Exception as e:
                gesture_name = "ERREUR"

            # Texte à l'écran
            cv2.putText(frame, f"Geste: {gesture_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Affichage de l'emoji via le mapping
            if gesture_name in gesture_images:
                overlay_image(frame, gesture_images[gesture_name], 20, 80)
            else:
                cv2.putText(frame, "Emoji non trouve", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imshow("Detection TIAGO - Labo CSIGS", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#test1