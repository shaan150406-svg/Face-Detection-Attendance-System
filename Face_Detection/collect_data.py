import cv2
import urllib.request       # fixed: need the submodule explicitly
import numpy as np
import os

# --- Setup ---
CASCADE_PATH = r'haarcascade_frontalface_default (1).xml'
URL = "http://10.114.75.94:8080/shot.jpg"
SAVE_DIR = "images"
TARGET_COUNT = 100

classifier = cv2.CascadeClassifier(CASCADE_PATH)
if classifier.empty():
    raise IOError(f"Could not load cascade file: {CASCADE_PATH}")

os.makedirs(SAVE_DIR, exist_ok=True)  # fixed: ensure save folder exists

data = []

while len(data) < TARGET_COUNT:
    try:
        image_from_url = urllib.request.urlopen(URL, timeout=5)  # fixed: added timeout
    except Exception as e:
        print("Failed to fetch frame:", e)
        continue

    frame = np.array(bytearray(image_from_url.read()), dtype=np.uint8)
    frame = cv2.imdecode(frame, -1)

    if frame is None:
        print("Bad frame received, skipping...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # fixed: detect on grayscale
    face_points = classifier.detectMultiScale(gray, 1.3, 5)

    if len(face_points) > 0:
        # Optional upgrade: pick the largest face instead of the first one found
        x, y, w, h = max(face_points, key=lambda f: f[2] * f[3])

        face_frame = frame[y:y+h, x:x+w]          # fixed: standard crop
        cv2.imshow("Only face", face_frame)

        if len(data) < TARGET_COUNT:              # fixed: correct condition
            data.append(face_frame)
            print(f"{len(data)}/{TARGET_COUNT}")

        # Draw a rectangle + count on the full frame for visual feedback
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(frame, str(len(data)), (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)  # fixed: sane scale/position

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) == ord("q"):
        break

cv2.destroyAllWindows()

if len(data) == TARGET_COUNT:
    name = input("Enter Face holder name: ").strip()
    for i in range(TARGET_COUNT):
        path = os.path.join(SAVE_DIR, f"{name}_{i}.jpg")
        ok = cv2.imwrite(path, data[i])
        if not ok:
            print(f"Warning: failed to save {path}")
    print("Done")
else:
    print("Need more data")