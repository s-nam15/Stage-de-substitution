import cv2
import mediapipe as mp
import csv
import os

# ===== CONFIGURATION DU MAPPING (Trié par nom de fichier) =====
mapping = {
    "BRAVO": "Applaudissements",
    "JOINTS": "Bout Des Doigts Joints",
    "CORNES": "Cornes Avec Les Doigts",
    "HONNEUR": "Doigt D’honneur",
    "DOS_MAIN": "Dos De Main Levée",
    "INDEX_HAUT_V2": "Index Pointant Vers Le Haut",
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