import urllib.request              # fixed: need the submodule explicitly
import cv2
import numpy as np
from keras.models import load_model

# ----------------------------------------------------
# 1) LOAD FACE DETECTOR AND TRAINED MODEL
# ----------------------------------------------------
CASCADE_PATH = r'haarcascade_frontalface_default (1).xml'
MODEL_PATH = r"final_model.h5"
URL = 'http://10.114.75.94:8080/shot.jpg'
CONFIDENCE_THRESHOLD = 0.80   # upgrade: below this, treat as "Unknown"

classifier = cv2.CascadeClassifier(CASCADE_PATH)
if classifier.empty():
    raise IOError(f"Could not load cascade file: {CASCADE_PATH}")

model = load_model(MODEL_PATH)

labels = ["SK"]   # add more names here if you retrain with more classes

# ----------------------------------------------------
# 2) LABEL DECODER FUNCTION (upgraded with confidence check)
# ----------------------------------------------------
def get_pred_label(pred_probs):
    pred_idx = np.argmax(pred_probs)
    confidence = pred_probs[pred_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return "Unknown", confidence
    return labels[pred_idx], confidence

# ----------------------------------------------------
# 3) FACE PREPROCESSING BEFORE PREDICTION
# ----------------------------------------------------
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

# ----------------------------------------------------
# 4) START REAL-TIME RECOGNITION LOOP
# ----------------------------------------------------
while True:
    try:
        img_url = urllib.request.urlopen(URL, timeout=5)   # fixed: timeout added
    except Exception as e:
        print("Failed to fetch frame:", e)
        continue

    image = np.array(bytearray(img_url.read()), dtype=np.uint8)
    frame = cv2.imdecode(image, -1)

    if frame is None:
        print("Bad frame received, skipping...")
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # fixed: detect on grayscale
    faces = classifier.detectMultiScale(gray_frame, 1.3, 5)

    for x, y, w, h in faces:
        # fixed: guard against zero-size / out-of-bounds crops
        if w <= 0 or h <= 0:
            continue
        x, y = max(x, 0), max(y, 0)
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        pred_probs = model.predict(preprocess(face), verbose=0)[0]  # fixed: silence spam
        name, confidence = get_pred_label(pred_probs)

        label_text = f"{name} ({confidence*100:.1f}%)"   # upgrade: show confidence
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)

        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

    cv2.imshow("capture", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()