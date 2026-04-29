import cv2
import mediapipe as mp

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


# ===== FONCTION : doigts levés =====
def fingers_up(lm, hand_label):

    fingers = []

    # Index (8 vs 6)
    fingers.append(lm[8].y < lm[6].y)

    # Middle (12 vs 10)
    fingers.append(lm[12].y < lm[10].y)

    # Ring (16 vs 14)
    fingers.append(lm[16].y < lm[14].y)

    # Pinky (20 vs 18)
    fingers.append(lm[20].y < lm[18].y)

    # Pouce : sens inversé selon la main (avec effet miroir)
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

    frame = cv2.flip(frame, 1)  # effet miroir
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture_text = ""

    if result.multi_hand_landmarks:

        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            hand_label = hand_info.classification[0].label  # "Right" ou "Left"
            fingers = fingers_up(lm, hand_label)

            # ===== LOGIQUE GESTES =====

            if fingers[:4] == [False, False, False, False]:
                gesture_text = "POING"

            elif fingers == [True, True, True, True, True]:
                gesture_text = "MAIN OUVERTE"

            elif fingers[0] == True and fingers[1:4] == [False, False, False]:
                gesture_text = "INDEX"

            elif fingers[:2] == [True, True] and fingers[2:4] == [False, False]:
                gesture_text = "PEACE"

            else:
                gesture_text = "UNKNOWN"

    # ===== AFFICHAGE =====
    cv2.putText(
        frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
    )

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# fsdgkjk
