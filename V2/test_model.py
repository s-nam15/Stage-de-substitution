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
    "BRAVO": "Applaudissements",
    "JOINTS": "Bout Des Doigts Joints",
    "CORNES": "Cornes Avec Les Doigts",
    "HONNEUR": "Doigt D’honneur",
    "DOS_MAIN": "Dos De Main Levée",
    "DOIGT_HAUT_V2": "Index Pointant Vers Le Haut",
    "INDEX_USER": "Index Pointant Vers L’utilisateur",
    "I_LOVE_YOU": "Signe Je T’aime",
    "INDEX_DROITE": "Main Avec Index Pointant À Droite",
    "INDEX_GAUCHE": "Main Avec Index Pointant À Gauche",
    "INDEX_BAS": "Main Avec Index Pointant Vers Le Bas",
    "INDEX_HAUT_V1": "Main Avec Index Pointant Vers Le Haut",
    "INDEX_POUCE_CROISE": "Main Avec Index Et Pouce Croisés",
    "PAUME_BAS": "Main Paume Vers Le Bas",
    "PAUME_HAUT": "Main Paume Vers Le Haut",
    "POUSSE_DROITE": "Main Qui Pousse Vers La Droite",
    "POUSSE_GAUCHE": "Main Qui Pousse Vers La Gauche",
    "ECRIT": "Main Qui Écrit",
    "MAIN_LEVEE": "Main Levée",
    "MAIN_ECARTEE": "Main Levée Doigts Écartés",
    "VERS_GAUCHE": "Main Vers La Gauche",
    "VERS_DROITE": "Main Vers La Droite",
    "MAINS_EN_PRIERE": "Mains En Prière",
    "MAINS_LEVEES": "Mains Levées",
    "OUVERTES": "Mains Ouvertes",
    "COEUR": "Mains Qui Forment Un Cœur",
    "OK": "Ok",
    "PRIERE_V1": "Paume Contre Paume Doigts Vers Le Haut",
    "POING_DROITE": "Poing À Droite",
    "POING_GAUCHE": "Poing À Gauche",
    "POING_FACE": "Poing De Face",
    "POING_LEVE": "Poing Levé",
    "POIGNEE": "Poignée De Main",
    "RAPPROCHES": "Pouce Et Index Rapprochés",
    "DISLIKE": "Pouce Vers Le Bas",
    "LIKE": "Pouce Vers Le Haut",
    "VULCAIN": "Salut Vulcain",
    "SELFIE": "Selfie",
    "TELEPHONE": "Signe Appel Téléphonique Avec Les Doigts",
    "SIGNE_MAIN": "Signe De La Main",
    "PEACE": "V De La Victoire",
    "VERNIS": "Vernis À Ongles"
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