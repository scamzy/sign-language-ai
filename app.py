import cv2
import mediapipe as mp
import time
import pyttsx3

# ---------------- TTS INIT ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face = mp_face.FaceDetection()
hands = mp_hands.Hands(max_num_hands=1)

# Gesture dictionary
gesture_dict = {
    (0,0,0,0,0): "STOP",
    (0,1,0,0,0): "I",
    (0,1,1,0,0): "YOU",
    (1,0,0,0,0): "GOOD",
    (1,1,0,0,0): "BAD",
    (1,1,1,1,1): "HELLO"
}

detected_words = []
final_sentence = ""
last_spoken = ""
last_word = ""
last_gesture_time = time.time()
pause_threshold = 3

cap = cv2.VideoCapture(0)

def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    states = []

    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        states.append(1)
    else:
        states.append(0)

    for i in range(1,5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            states.append(1)
        else:
            states.append(0)

    return tuple(states)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------- FACE ----------------
    face_result = face.process(rgb)
    face_count = 0

    if face_result.detections:
        face_count = len(face_result.detections)
        for det in face_result.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin*w), int(bbox.ymin*h)
            w_box, h_box = int(bbox.width*w), int(bbox.height*h)
            cv2.rectangle(frame, (x,y), (x+w_box, y+h_box), (0,255,0), 2)

    # Attention + Health
    attention = "Active" if face_count > 0 else "Not Present"
    health = "Stable"

    # ---------------- HAND ----------------
    hand_result = hands.process(rgb)

    if hand_result.multi_hand_landmarks:
        for handLms in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_states(handLms)

            if fingers in gesture_dict:
                word = gesture_dict[fingers]
                now = time.time()

                if word != last_word:
                    detected_words.append(word)
                    last_word = word
                    last_gesture_time = now

                cv2.putText(frame, "Detected: " + word, (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # ---------------- SENTENCE LOGIC ----------------
    if time.time() - last_gesture_time > pause_threshold and detected_words:

        unique = []
        for w in detected_words:
            if w not in unique:
                unique.append(w)

        # Smart grammar
        if "I" in unique and "GOOD" in unique:
            final_sentence = "HELLO, I AM GOOD"
        elif "I" in unique and "BAD" in unique:
            final_sentence = "HELLO, I AM BAD"
        elif "I" in unique and "YOU" in unique:
            final_sentence = "I LOVE YOU"
        else:
            final_sentence = " ".join(unique)

        # ---------------- SPEAK ----------------
        if final_sentence != last_spoken:
            speak(final_sentence)
            last_spoken = final_sentence

        # ---------------- SAVE HISTORY ----------------
        with open("history.txt", "a") as f:
            f.write(final_sentence + "\n")

        detected_words = []

    # ---------------- UI ----------------
    cv2.rectangle(frame, (0,0), (640,40), (0,0,0), -1)
    cv2.putText(frame, "AI Sign Language Interpreter + Face Monitoring", (80,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.rectangle(frame, (20,320), (620,460), (30,30,30), -1)
    cv2.rectangle(frame, (20,320), (620,460), (200,200,200), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"Users: {face_count}", (40,350), font, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Health: {health}", (40,370), font, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Attention: {attention}", (40,390), font, 0.5, (255,255,255), 1)

    cv2.putText(frame, "Detected: " + (last_word if last_word else "-"),
                (40,420), font, 0.6, (0,255,255), 2)

    cv2.putText(frame, "Sentence:", (40,440), font, 0.5, (255,255,255), 1)
    cv2.putText(frame, final_sentence, (150,440),
                font, 0.6, (0,255,0), 2)

    cv2.imshow("AI Translator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()