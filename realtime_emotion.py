import cv2
import numpy as np
import tensorflow as tf

# =========================
# CONFIG
# =========================
IMG_SIZE = 48
EMOTIONS = ["Angry", "Happy", "Sad"]

PROTO = "deploy.prototxt"
MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("emotion_model.h5")

# =========================
# LOAD FACE DETECTOR (DNN)
# =========================
face_net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("🎥 Webcam started. Press Q to quit.")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            preds = model.predict(face, verbose=0)
            emotion = EMOTIONS[np.argmax(preds)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            break  # only first face

    cv2.imshow("Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed.")
