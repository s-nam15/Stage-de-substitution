import cv2
import mediapipe as mp

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ===== FONCTION : doigts levés =====
def fingers_up(lm):

    fingers = []

    # Index (8 vs 6)
    fingers.append(lm[8].y < lm[6].y)

    # Middle (12 vs 10)
    fingers.append(lm[12].y < lm[10].y)

    # Ring (16 vs 14)
    fingers.append(lm[16].y < lm[14].y)

    # Pinky (20 vs 18)
    fingers.append(lm[20].y < lm[18].y)

    # Thumb (horizontal)
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

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            fingers = fingers_up(lm)

            # ===== LOGIQUE GESTES =====

            if fingers == [False, False, False, False, False]:
                gesture_text = "POING ✊"

            elif fingers == [True, True, True, True, True]:
                gesture_text = "MAIN OUVERTE ✋"

            elif fingers == [True, False, False, False, False]:
                gesture_text = "INDEX ☝️"

            elif fingers == [True, True, False, False, False]:
                gesture_text = "PEACE ✌️"

            else:
                gesture_text = "UNKNOWN"

    # ===== AFFICHAGE =====
    cv2.putText(frame, gesture_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 255, 0), 3)

    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()