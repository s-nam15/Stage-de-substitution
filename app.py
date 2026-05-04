from flask import Flask, Response, render_template
import cv2
import mediapipe as mp

app = Flask(__name__)

# ===== INIT MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ===== CHARGER IMAGES =====
fist_img = cv2.imread("fist.png")
hand_img = cv2.imread("hand.png")
peace_img = cv2.imread("peace.png")

# ===== OVERLAY =====
def overlay_image(frame, img, x, y, size=120):
    if img is None:
        return
    img = cv2.resize(img, (size, size))
    h, w, _ = img.shape
    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        frame[y:y+h, x:x+w] = img

# ===== DOIGTS =====
def fingers_up(lm, hand_label):
    fingers = []
    fingers.append(lm[8].y < lm[6].y)
    fingers.append(lm[12].y < lm[10].y)
    fingers.append(lm[16].y < lm[14].y)
    fingers.append(lm[20].y < lm[18].y)

    if hand_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    return fingers

# ===== GENERATE VIDEO =====
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(
                result.multi_hand_landmarks, result.multi_handedness):

                mp_draw.draw_landmarks(frame, hand_landmarks,
                                       mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                hand_label = hand_info.classification[0].label
                fingers = fingers_up(lm, hand_label)

                # ===== GESTES =====
                if fingers.count(True) == 0:
                    overlay_image(frame, fist_img, 50, 150)

                elif fingers.count(True) >= 4:
                    overlay_image(frame, hand_img, 50, 150)

                elif fingers[:2] == [True, True] and fingers[2:4] == [False, False]:
                    overlay_image(frame, peace_img, 50, 150)

        # encode image
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)