from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import webbrowser
from threading import Timer
import face_recognition
import os
from gtts import gTTS

app = Flask(__name__)

# ============ DETECTION LOGIC START ============

# --- Sound Setup ---
pygame.mixer.init()
alarm_playing = False
try:
    alarm_sound = pygame.mixer.Sound('warning-sound-6686.mp3')
except pygame.error as e:
    print(f"Error: 'warning-sound-6686.mp3' sound file not found: {e}")
    alarm_sound = None

# --- Load Known Faces and Welcome Sounds ---
known_face_encodings = []
known_face_names = []
welcome_sounds = {}
FACES_DIR = "known_faces"

if not os.path.exists(FACES_DIR):
    print(f"ERROR: Folder '{FACES_DIR}' not found. Please create it and add known faces.")
else:
    print(f"Loading faces from '{FACES_DIR}' folder...")
    for file_name in os.listdir(FACES_DIR):
        if file_name.endswith((".jpg", ".png", ".jpeg")):
            try:
                driver_name = os.path.splitext(file_name)[0].title()
                image_path = os.path.join(FACES_DIR, file_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(driver_name)
                    print(f"-> Loaded {file_name} as Driver: {driver_name}")

                    welcome_file = f"welcome_{driver_name.lower()}.mp3"

                    if not os.path.exists(welcome_file):
                        try:
                            print(f"'{welcome_file}' not found, creating it...")
                            tts = gTTS(text=f"Welcome, {driver_name}! System is active.", lang='en')
                            tts.save(welcome_file)
                            print(f"'{welcome_file}' created successfully.")
                        except Exception as e:
                            print(f"gTTS Error (check Internet connection): {e}")

                    if os.path.exists(welcome_file):
                        welcome_sounds[driver_name] = pygame.mixer.Sound(welcome_file)
                        print(f"-> Welcome sound loaded for '{driver_name}'.")
                else:
                    print(f"WARNING: No face detected in {file_name}. Skipping.")
            except Exception as e:
                print(f"ERROR: Error loading {file_name}: {e}")

    if not known_face_names:
        print("WARNING: No faces loaded from the 'known_faces' folder.")
    else:
        print(f"Successfully loaded {len(known_face_names)} known faces.")

# --- Webcam Setup ---
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
except Exception as e:
    print(f"Webcam Error: {e}")
    predictor = None

# --- Dlib Detector and Predictor ---
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("Error: 'shape_predictor_68_face_landmarks.dat' not found!")
    print("Please make sure the file is in the same folder as app.py")
    predictor = None


# --- Helper Functions ---
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0


# --- Frame Generator Function ---
def generate_frames():
    global alarm_playing, face_rect, frame_counter, sleep, drowsy, active, status, color

    sleep, drowsy, active = 0, 0, 0
    status, color = "", (0, 0, 0)
    FRAME_SKIP_RATE = 5
    RECOG_SKIP_RATE = 30
    frame_counter = 0
    face_rect = None
    current_driver_name = "Scanning..."
    drivers_welcomed = {}

    if predictor is None or not cap.isOpened():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        error_text = "Error: Webcam not found."
        if predictor is None:
            error_text = "Error: 'shape_predictor_68_face_landmarks.dat' not found."
        cv2.putText(frame, error_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            cv2.waitKey(30)
            continue

        frame_counter += 1

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
        except Exception:
            continue

        face_detected = False

        if frame_counter % FRAME_SKIP_RATE == 0 or face_rect is None:
            try:
                faces = detector(gray)
                if len(faces) > 0:
                    face_rect = faces[0]
                    face_detected = True
                else:
                    face_rect = None
            except RuntimeError:
                face_rect = None
                continue
        else:
            if face_rect is not None:
                face_detected = True

        if face_detected and face_rect is not None:
            try:
                landmarks = predictor(gray, face_rect)
                landmarks_np = face_utils.shape_to_np(landmarks)
                x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                print(f"Landmark prediction error: {e}")
                face_rect = None
                continue

            # Face Recognition (every 30 frames)
            if frame_counter % RECOG_SKIP_RATE == 1:
                try:
                    face_location_dlib = (face_rect.top(), face_rect.right(), face_rect.bottom(), face_rect.left())
                    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8)
                    current_face_encodings = face_recognition.face_encodings(rgb_frame, [face_location_dlib])

                    if len(current_face_encodings) > 0:
                        current_face_encoding = current_face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
                        face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)

                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                current_driver_name = known_face_names[best_match_index]
                                if not drivers_welcomed.get(current_driver_name):
                                    if current_driver_name in welcome_sounds and not pygame.mixer.get_busy():
                                        welcome_sounds[current_driver_name].play()
                                    drivers_welcomed[current_driver_name] = True
                            else:
                                current_driver_name = "Unknown Driver"
                        else:
                            current_driver_name = "Unknown Driver"
                except Exception as e:
                    print(f"Face recognition error: {e}")

            # Drowsiness Detection Logic
            left_blink = blinked(landmarks_np[36], landmarks_np[37], landmarks_np[38],
                                 landmarks_np[41], landmarks_np[40], landmarks_np[39])
            right_blink = blinked(landmarks_np[42], landmarks_np[43], landmarks_np[44],
                                  landmarks_np[47], landmarks_np[46], landmarks_np[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    if alarm_sound and not alarm_playing:
                        alarm_sound.play(-1)
                        alarm_playing = True
            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)
                    if alarm_sound and not alarm_playing:
                        alarm_sound.play(-1)
                        alarm_playing = True
            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)
                    if alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False

            for n in range(0, 68):
                (x, y) = landmarks_np[n]
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        else:
            status = "No Face Detected"
            color = (200, 200, 200)
            sleep, drowsy, active = 0, 0, 0
            if alarm_playing:
                alarm_sound.stop()
                alarm_playing = False
            current_driver_name = "Scanning..."

        cv2.putText(frame, f"Driver: {current_driver_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')


# ============ DETECTION LOGIC END ============

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Auto Open Browser ---
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


# --- Run the App ---
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=False)
