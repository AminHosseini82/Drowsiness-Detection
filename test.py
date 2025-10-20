import cv2
import dlib
from scipy.spatial import distance
import threading
import simpleaudio as sa # 🌟 کتابخانه جدید

# --- ثابت‌ها (Constants) ---
EYE_AR_THRESH = 0.26        
EYE_AR_PRE_THRESH = 0.30    
CONSEC_FRAMES_HIGH = 60     
CONSEC_FRAMES_LOW = 200     
# 👈 مطمئن شوید مسیر و نام فایل‌های WAV درست هستند
ALARM_HIGH_SOUND = "F:\Amin_Projects\Learning\Drowsiness-Detection_3\alarms\alarm_high.wav"  
ALARM_LOW_SOUND = "F:\Amin_Projects\Learning\Drowsiness-Detection_3\alarms\alarm_low.wav"

# --- شمارنده‌ها (Counters) ---
DROWSY_COUNTER = 0          
PRE_DROWSY_COUNTER = 0

# --- مدیریت آلارم ---
alarm_high_on = False 
alarm_low_on = False 

# 🌟 متغیرهای ذخیره WaveObject برای پخش سریع‌تر
wave_high = None
wave_low = None

# -------------------------------------------------------------------------
# 🌟 تابع جدید: اجرای simpleaudio در یک نَخ جداگانه
def play_alarm_thread(wave_obj):
    """
    پخش فایل صوتی با استفاده از simpleaudio.
    """
    if wave_obj is not None:
        try:
            # play_obj = wave_obj.play() # 👈 اگر بخواهید آلارم تا انتها پخش شود
            wave_obj.play()
        except Exception as e:
            # این خطا معمولاً در صورت مشکل در زیرسیستم صوتی رخ می‌دهد
            print(f"Error playing sound via simpleaudio: {e}")
            
# -------------------------------------------------------------------------
# ... (تابع calculate_EAR بدون تغییر)
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# --- بارگذاری اولیه فایل‌های صوتی قبل از حلقه اصلی ---
try:
    wave_high = sa.WaveObject.from_wave_file(ALARM_HIGH_SOUND)
    wave_low = sa.WaveObject.from_wave_file(ALARM_LOW_SOUND)
    print("Sound files loaded successfully.")
except FileNotFoundError:
    print("ERROR: Could not find sound files. Check the paths and file names.")
    # می‌توانیم در اینجا برنامه را متوقف کنیم یا با آلارم‌های غیر فعال ادامه دهیم
except Exception as e:
    print(f"ERROR loading sound files: {e}")
    
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
    
    # ... (کدهای منطق تشخیص چهره و استخراج نقاط چشم مانند قبل) ...
    # (کدهای استخراج نقاط و رسم خطوط) ... 
    
    # 🌟 فرض می‌کنیم استخراج نقاط leftEye و rightEye انجام شده است
    leftEye = []
    rightEye = []

    # کدهای استخراج نقاط و رسم خطوط چشم (36 تا 47) باید اینجا قرار گیرند
    if len(faces) > 0:
        face = faces[0] # پردازش اولین چهره
        face_landmarks = dlib_facelandmark(gray, face)
        
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
        # ------------------------------------------------------------------
        
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
            alarm_low_on = False 

            if DROWSY_COUNTER >= CONSEC_FRAMES_HIGH:
                cv2.putText(frame, "!!! DROWSY ALERT !!!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                if not alarm_high_on:
                    alarm_high_on = True
                    # 🚀 استفاده از WaveObject بارگذاری شده 
                    t = threading.Thread(target=play_alarm_thread, args=(wave_high,))
                    t.start()
                    print("Drowsy! HIGH Alarm initiated.")
        
        elif EAR < EYE_AR_PRE_THRESH: # تشخیص خواب‌آلودگی مرزی (0.26 <= EAR < 0.30)
            PRE_DROWSY_COUNTER += 1
            DROWSY_COUNTER = 0
            alarm_high_on = False 

            if PRE_DROWSY_COUNTER >= CONSEC_FRAMES_LOW:
                cv2.putText(frame, "Tired - Take a Break!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                
                if not alarm_low_on:
                    alarm_low_on = True
                    # 🚀 استفاده از WaveObject بارگذاری شده 
                    t = threading.Thread(target=play_alarm_thread, args=(wave_low,))
                    t.start()
                    print("Pre-Drowsy. LOW Alarm initiated.")
            
        else: # بیدار (EAR >= 0.30)
            DROWSY_COUNTER = 0
            PRE_DROWSY_COUNTER = 0
            alarm_high_on = False
            alarm_low_on = False
            
        print(f"EAR: {EAR}, Drowsy: {DROWSY_COUNTER}/{CONSEC_FRAMES_HIGH}, Pre-Drowsy: {PRE_DROWSY_COUNTER}/{CONSEC_FRAMES_LOW}")
    else:
        # اگر چهره‌ای نبود، شمارنده‌ها ریست می‌شوند
        DROWSY_COUNTER = 0
        PRE_DROWSY_COUNTER = 0
        alarm_high_on = False
        alarm_low_on = False


    cv2.imshow("Driver Monitoring", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()