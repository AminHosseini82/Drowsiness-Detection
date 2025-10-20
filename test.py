import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound
import threading # 🌟 کتابخانه جدید برای اجرای همزمان

# --- ثابت‌ها (Constants) ---
EYE_AR_THRESH = 0.26        
EYE_AR_PRE_THRESH = 0.30    
CONSEC_FRAMES_HIGH = 60     
CONSEC_FRAMES_LOW = 200     
ALARM_HIGH_SOUND = "alarm_high.wav"  
ALARM_LOW_SOUND = "alarm_low.wav"    

# --- شمارنده‌ها (Counters) ---
DROWSY_COUNTER = 0          
PRE_DROWSY_COUNTER = 0

# --- مدیریت آلارم ---
# 🌟 وضعیت آلارم را مدیریت می‌کنیم تا فقط یک بار پخش شود و تکرار نشود
alarm_high_on = False 
alarm_low_on = False 

# -------------------------------------------------------------------------
# 🌟 تابع جدید: اجرای playsound در یک نَخ جداگانه
def play_alarm_thread(sound_file):
    """
    پخش فایل صوتی در یک نَخ جدید.
    توجه: این تابع فقط برای پخش یکباره در هر نوبت هشدار استفاده می‌شود.
    """
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")

# ... (تابع calculate_EAR بدون تغییر)
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# ... (مقداردهی اولیه Dlib و OpenCV بدون تغییر)
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
# فرض بر این است که مسیر فایل shape_predictor_68_face_landmarks.dat درست است

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    
    # فرض می‌کنیم چهره‌ای در فریم هست، در غیر این صورت شمارنده‌ها ریست می‌شوند
    if len(faces) == 0:
        DROWSY_COUNTER = 0
        PRE_DROWSY_COUNTER = 0
        alarm_high_on = False
        alarm_low_on = False
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        
        # ... (کدهای استخراج نقاط و رسم خطوط چشم، که برای سادگی در اینجا خلاصه شده) ...
        leftEye = []
        rightEye = []
        # (کدهای استخراج نقاط و رسم خطوط) ... 
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41: next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47: next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
        # ...
        
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        
        # ------------------------------------------------------------------
        # --- منطق هشدار با مدیریت نَخ ---
        # ------------------------------------------------------------------

        if EAR < EYE_AR_THRESH: # تشخیص خواب‌آلودگی شدید (EAR < 0.26)
            DROWSY_COUNTER += 1
            PRE_DROWSY_COUNTER = 0
            alarm_low_on = False # ریست کردن آلارم مرزی

            if DROWSY_COUNTER >= CONSEC_FRAMES_HIGH:
                cv2.putText(frame, "!!! DROWSY ALERT !!!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # 📢 اگر آلارم شدید فعال نیست، آن را فعال کرده و در نَخ جدید پخش می‌کنیم
                if not alarm_high_on:
                    alarm_high_on = True
                    # ایجاد و شروع نَخ جدید برای پخش صدا
                    t = threading.Thread(target=play_alarm_thread, args=(ALARM_HIGH_SOUND,))
                    t.start()
                    print("Drowsy! HIGH Alarm initiated.")
        
        elif EAR < EYE_AR_PRE_THRESH: # تشخیص خواب‌آلودگی مرزی (0.26 <= EAR < 0.30)
            PRE_DROWSY_COUNTER += 1
            DROWSY_COUNTER = 0
            alarm_high_on = False # ریست کردن آلارم شدید

            if PRE_DROWSY_COUNTER >= CONSEC_FRAMES_LOW:
                cv2.putText(frame, "Tired - Take a Break!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                
                # 📢 اگر آلارم مرزی فعال نیست، آن را فعال کرده و در نَخ جدید پخش می‌کنیم
                if not alarm_low_on:
                    alarm_low_on = True
                    # ایجاد و شروع نَخ جدید برای پخش صدا
                    t = threading.Thread(target=play_alarm_thread, args=(ALARM_LOW_SOUND,))
                    t.start()
                    print("Pre-Drowsy. LOW Alarm initiated.")
            
        else: # بیدار (EAR >= 0.30)
            DROWSY_COUNTER = 0
            PRE_DROWSY_COUNTER = 0
            alarm_high_on = False # ریست کردن هر دو وضعیت آلارم
            alarm_low_on = False

        print(f"EAR: {EAR}, Drowsy: {DROWSY_COUNTER}/{CONSEC_FRAMES_HIGH}, Pre-Drowsy: {PRE_DROWSY_COUNTER}/{CONSEC_FRAMES_LOW}")


    cv2.imshow("Driver Monitoring", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()