import cv2
import mediapipe as mp

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ===== CHARGEMENT DES IMAGES =====
gesture_images = {
    "POING": cv2.imread("poing.jpg"),
    "main ouverte": cv2.imread("hand.jpg"),
    "INDEX": cv2.imread("index.jpg"),
    "PEACE": cv2.imread("peace.jpg"),
    "UNKNOWN": cv2.imread("unknown.jpg"),
}

# ===== CHARGER EMOJIS (CHEMIN ABSOLU RECOMMANDÉ) =====
fist_img = cv2.imread("fist.png")
hand_img = cv2.imread("hand.png")
peace_img = cv2.imread("peace.png")

# ===== VERIFICATION =====
print("fist:", fist_img is None)
print("hand:", hand_img is None)
print("peace:", peace_img is None)


# ===== FONCTION OVERLAY (ANTI-CRASH) =====
def overlay_image(frame, img, x, y, size=120):
    if img is None:
        return  # évite crash

    img = cv2.resize(img, (size, size))
    h, w, _ = img.shape

    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        frame[y : y + h, x : x + w] = img


# ===== FONCTION DOIGTS =====
def fingers_up(lm, hand_label):
    fingers = []

    fingers.append(lm[8].y < lm[6].y)  # index
    fingers.append(lm[12].y < lm[10].y)  # middle
    fingers.append(lm[16].y < lm[14].y)  # ring
    fingers.append(lm[20].y < lm[18].y)  # pinky

    if hand_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    return fingers


# ===== BOUCLE =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            hand_label = hand_info.classification[0].label

            fingers = fingers_up(lm, hand_label)

            # ===== GESTES + EMOJI =====

            if fingers[:4] == [False, False, False, False]:
                overlay_image(frame, fist_img, 50, 150)

            elif fingers.count(True) >= 4:
                overlay_image(frame, hand_img, 50, 150)

            elif fingers[0] == True and fingers[1:4] == [False, False, False]:
                gesture_text = "INDEX"

            elif fingers[:2] == [True, True] and fingers[2:4] == [False, False]:
                overlay_image(frame, peace_img, 50, 150)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
