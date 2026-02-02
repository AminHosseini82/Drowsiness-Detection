import cv2
import dlib
from scipy.spatial import distance
import pygame
import time

# --- Constants ---
EYE_AR_THRESH = 0.26        
EYE_AR_PRE_THRESH = 0.30    
CONSEC_FRAMES_HIGH = 60     
CONSEC_FRAMES_LOW = 200     
ALARM_COOLDOWN = 3  # Seconds between alarm replays

# 🌟 فریم‌های لازم برای تشخیص بیداری (کم‌تر = سریع‌تر قطع میشه)
AWAKE_FRAMES_NEEDED = 5  # 5 فریم متوالی چشم باز = قطع فوری آلارم

# ⚠️ Update these paths
ALARM_HIGH_SOUND = r"F:\\University\\7th term\\Computer vision\\project\\Drowsiness-Detection\\alarms\\alarm_high.wav"
ALARM_LOW_SOUND = r"F:\\University\\7th term\\Computer vision\\project\\Drowsiness-Detection\\alarms\\alarm_low.wav"

# --- Counters ---
DROWSY_COUNTER = 0
PRE_DROWSY_COUNTER = 0
AWAKE_COUNTER = 0  # 🌟 شمارنده جدید برای تشخیص بیداری

# --- Alarm Management ---
alarm_high_on = False
alarm_low_on = False
last_alarm_time_high = 0
last_alarm_time_low = 0

# -------------------------------------------------------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# -------------------------------------------------------------------------
# Initialize pygame mixer for audio
print("Initializing audio system...")
try:
    pygame.mixer.init()
    alarm_high = pygame.mixer.Sound(ALARM_HIGH_SOUND)
    alarm_low = pygame.mixer.Sound(ALARM_LOW_SOUND)
    print("✓ Sound files loaded successfully.")
except Exception as e:
    print(f"ERROR loading sound files: {e}")
    print("Program will continue without sound.")
    alarm_high = None
    alarm_low = None

# -------------------------------------------------------------------------
# Initialize Dlib and OpenCV
print("Initializing camera and face detector...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

hog_face_detector = dlib.get_frontal_face_detector()

try:
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("✓ Face landmark detector loaded.")
except RuntimeError:
    print("ERROR: 'shape_predictor_68_face_landmarks.dat' not found.")
    cap.release()
    exit()

print("✓ Starting monitoring... (Press ESC to quit)")
print("-" * 60)

# -------------------------------------------------------------------------
# Main Loop
frame_count = 0
faces = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame")
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces every frame
        faces = hog_face_detector(gray)
        
        current_time = time.time()

        # Process if faces detected
        if len(faces) > 0:
            face = faces[0]
            
            try:
                face_landmarks = dlib_facelandmark(gray, face)
            except Exception as e:
                print(f"Warning: Could not extract landmarks: {e}")
                continue

            leftEye = []
            rightEye = []

            # Extract Left Eye (Points 36-41)
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                
                next_point = n + 1
                if n == 41: next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # Extract Right Eye (Points 42-47)
            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                
                next_point = n + 1
                if n == 47: next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # Calculate EAR
            try:
                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)
                EAR = (left_ear + right_ear) / 2
                EAR = round(EAR, 2)
            except Exception as e:
                print(f"Warning: EAR calculation error: {e}")
                continue

            # --- Drowsiness Detection Logic ---
            
            # Severe Drowsiness (EAR < 0.26)
            if EAR < EYE_AR_THRESH:
                DROWSY_COUNTER += 1
                PRE_DROWSY_COUNTER = 0
                AWAKE_COUNTER = 0  # 🌟 ریست counter بیداری
                
                if DROWSY_COUNTER >= CONSEC_FRAMES_HIGH:
                    cv2.putText(frame, "!!! DROWSY ALERT !!!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    # Play high alarm with cooldown
                    if (current_time - last_alarm_time_high) > ALARM_COOLDOWN:
                        if alarm_high is not None:
                            try:
                                pygame.mixer.stop()
                                alarm_high.play()
                                print(f"[{time.strftime('%H:%M:%S')}] 🚨 HIGH ALARM - EAR: {EAR}")
                            except Exception as e:
                                print(f"Error playing high alarm: {e}")
                        
                        alarm_high_on = True
                        alarm_low_on = False
                        last_alarm_time_high = current_time

            # Fatigue (0.26 <= EAR < 0.30)
            elif EAR < EYE_AR_PRE_THRESH:
                PRE_DROWSY_COUNTER += 1
                DROWSY_COUNTER = 0
                AWAKE_COUNTER = 0  # 🌟 ریست counter بیداری
                
                if PRE_DROWSY_COUNTER >= CONSEC_FRAMES_LOW:
                    cv2.putText(frame, "Tired - Take a Break!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                    
                    # Play low alarm with cooldown
                    if (current_time - last_alarm_time_low) > ALARM_COOLDOWN:
                        if alarm_low is not None:
                            try:
                                pygame.mixer.stop()
                                alarm_low.play()
                                print(f"[{time.strftime('%H:%M:%S')}] ⚠️ LOW ALARM - EAR: {EAR}")
                            except Exception as e:
                                print(f"Error playing low alarm: {e}")
                        
                        alarm_low_on = True
                        alarm_high_on = False
                        last_alarm_time_low = current_time
            
            # Normal State (Awake - EAR >= 0.30)
            else:
                AWAKE_COUNTER += 1  # 🌟 افزایش counter بیداری
                
                # 🌟 اگر چند فریم متوالی چشم باز بود، فوراً آلارم را قطع کن
                if AWAKE_COUNTER >= AWAKE_FRAMES_NEEDED:
                    # قطع فوری آلارم
                    if alarm_high_on or alarm_low_on:
                        try:
                            pygame.mixer.stop()
                            print(f"[{time.strftime('%H:%M:%S')}] ✅ Driver is AWAKE - Alarm STOPPED - EAR: {EAR}")
                        except Exception as e:
                            print(f"Error stopping alarm: {e}")
                    
                    # Reset all counters
                    DROWSY_COUNTER = 0
                    PRE_DROWSY_COUNTER = 0
                    alarm_high_on = False
                    alarm_low_on = False
                
                cv2.putText(frame, "Active - Monitoring", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display information
            cv2.putText(frame, f"EAR: {EAR}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status bar at bottom
            status_text = f"Drowsy: {DROWSY_COUNTER}/{CONSEC_FRAMES_HIGH} | Tired: {PRE_DROWSY_COUNTER}/{CONSEC_FRAMES_LOW} | Awake: {AWAKE_COUNTER}"
            cv2.putText(frame, status_text, (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Gradually reset counters
            DROWSY_COUNTER = max(0, DROWSY_COUNTER - 1)
            PRE_DROWSY_COUNTER = max(0, PRE_DROWSY_COUNTER - 1)
            AWAKE_COUNTER = 0

        # Show frame
        cv2.imshow("Driver Drowsiness Detection System", frame)

        # Check for ESC key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n[EXIT] User pressed ESC - Closing program...")
            break
        elif key == ord('r'):  # Press 'r' to reset counters
            DROWSY_COUNTER = 0
            PRE_DROWSY_COUNTER = 0
            AWAKE_COUNTER = 0
            pygame.mixer.stop()
            print("[RESET] Counters reset manually & alarm stopped")

except KeyboardInterrupt:
    print("\n[EXIT] Program interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n[ERROR] Unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("\n[CLEANUP] Shutting down...")
    
    try:
        pygame.mixer.stop()
        pygame.mixer.quit()
    except:
        pass
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("[DONE] Program closed successfully.")
    print("=" * 60)
