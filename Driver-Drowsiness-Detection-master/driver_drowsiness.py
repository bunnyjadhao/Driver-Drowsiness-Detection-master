import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame  # Import pygame for the alarm

# ============ OPTIMIZATION 1: SOUND OPTIMIZATION ============
# Sound file ko pehle se hi load kar lein.
pygame.mixer.init()
try:
    # 'music' ki jagah 'Sound' ka istemal karein, yeh chote effects ke liye behtar hai
    alarm_sound = pygame.mixer.Sound('warning-sound-6686.mp3')
except pygame.error as e:
    print(f"Error: 'warning-sound-6686.mp3' sound file nahi mila ya load nahi hua: {e}")
    print("Alarm ke bina jaari... (Continuing without alarm)")
    alarm_sound = None

alarm_playing = False
# ==========================================================

# Camera (index 0) aur DSHOW driver (jo v8 mein fix hua tha)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the shape predictor model
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("Error: shape_predictor_68_face_landmarks.dat not found!")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Status tracking
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# ============ OPTIMIZATION 2: FRAME SKIPPING ============
# Hum har 5 mein se 1 frame par hi face detection chalayenge
FRAME_SKIP_RATE = 5
frame_counter = 0
face_rect = None  # Aakhri (last) face ki location store karega
# =======================================================

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to detect blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Open eyes
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Closed eyes

# Main loop
while True:
    ret, frame = cap.read()
    frame_counter += 1 # Frame counter ko badhayein

    if not ret:
        print("Error: Failed to grab a frame. Is webcam (index 0) connected?")
        cv2.waitKey(1000)
        continue
        
    if frame is None or frame.size == 0:
        print("Error: Received an empty frame from webcam (index 0).")
        cv2.waitKey(30)
        continue

    # Grayscale conversion (jaisa v8 mein fix kiya tha)
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        try:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            continue
    else:
        continue

    # dlib ke liye force conversion
    try:
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
    except Exception as e:
        continue

    face_detected = False

    # ============ OPTIMIZATION 2: FRAME SKIPPING LOGIC ============
    # Har FRAME_SKIP_RATE frames par detector chalayein
    if frame_counter % FRAME_SKIP_RATE == 0 or face_rect is None:
        try:
            faces = detector(gray)
            if len(faces) > 0:
                face_rect = faces[0]  # Pehla chehra store karein
                face_detected = True
            else:
                face_rect = None  # Koi chehra nahi mila
        except RuntimeError:
            # Agar dlib fail ho (jo pehle ho raha tha), toh skip karein
            face_rect = None
            continue
    else:
        # Beech ke (intermediate) frames ke liye, check karein ki purana face_rect hai ya nahi
        if face_rect is not None:
            face_detected = True
    # =============================================================

    
    # Agar chehra mila (naya ya purana), toh landmarks dhoondhein
    if face_detected and face_rect is not None:
        try:
            # Expensive detector ki jagah, hum sirf predictor chalayenge
            landmarks = predictor(gray, face_rect)
            landmarks = face_utils.shape_to_np(landmarks)
            
            # Chehre par box banayein
            x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            face_detected = True
        except Exception as e:
            # Agar predictor fail ho (chehra frame se bahar chala jaye)
            face_rect = None  # Agle frame par detector ko force karein
            face_detected = False
            continue

        # Blinking detect karein
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41],
                             landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47],
                              landmarks[46], landmarks[45])

        # Status Update
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                # ============ OPTIMIZATION 1: SOUND PLAY ============
                if alarm_sound and not alarm_playing:
                    alarm_sound.play(-1)  # -1 ka matlab hai loop mein bajao
                    alarm_playing = True
                # ====================================================
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                # ============ OPTIMIZATION 1: SOUND PLAY ============
                if alarm_sound and not alarm_playing:
                    alarm_sound.play(-1)
                    alarm_playing = True
                # ====================================================
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                # ============ OPTIMIZATION 1: SOUND STOP ============
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False
                # ====================================================
        
        # =========== YAHAN BADLAAV KIYA GAYA HAI (v10) ===========
        # Aapke request par 68 dots ko waapas enable kar diya hai.
        # Yeh thoda lag paida kar sakta hai.
        for n in range(0, 68):
            (x, y) = landmarks[n]
            # Dots ko thoda chhota (radius 1) aur laal (red) kar diya hai
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        # =======================================================

    else:
        # Agar chehra nahi mila
        status = "No Face Detected"
        color = (200, 200, 200)
        # Counters reset karein
        sleep = 0
        drowsy = 0
        active = 0
        # Alarm band karein
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

    # Status text hamesha frame par dikhayein
    cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Sirf ek hi window dikhayein
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release resources
if alarm_sound:
    alarm_sound.stop()
cap.release()
cv2.destroyAllWindows()