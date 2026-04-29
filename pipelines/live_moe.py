import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# LOAD MODELS
# -----------------------------
def load_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

kitchen = load_model("models/experts/kitchen/kitchen.tflite")
display = load_model("models/experts/display/display.tflite")
climate = load_model("models/experts/climate/climate.tflite")

input_details = kitchen.get_input_details()
output_details = kitchen.get_output_details()

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame):
    img = cv2.resize(frame, (320, 320))
    img = np.expand_dims(img, axis=0)

    if input_details[0]['dtype'] == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)

    return img

# -----------------------------
# ROUTER (MINIMAL MoE)
# -----------------------------
def route(frame_count):
    if frame_count % 60 < 20:
        return "kitchen"
    elif frame_count % 60 < 40:
        return "display"
    else:
        return "climate"

# -----------------------------
# INFERENCE
# -----------------------------
def infer(model, input_data):
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()

    boxes = model.get_tensor(output_details[0]['index'])[0]
    classes = model.get_tensor(output_details[1]['index'])[0]
    scores = model.get_tensor(output_details[2]['index'])[0]

    return boxes, classes, scores

# -----------------------------
# DRAW
# -----------------------------
def draw(frame, boxes, scores, label, threshold=0.4):
    h, w, _ = frame.shape

    for i in range(len(scores)):
        if scores[i] > threshold:
            y1, x1, y2, x2 = boxes[i]

            x1, x2 = int(x1*w), int(x2*w)
            y1, y2 = int(y1*h), int(y2*h)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {scores[i]:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)

    return frame

# -----------------------------
# MAIN LOOP
# -----------------------------
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_data = preprocess(frame)

    expert = route(frame_count)

    if expert == "kitchen":
        boxes, classes, scores = infer(kitchen, input_data)
    elif expert == "display":
        boxes, classes, scores = infer(display, input_data)
    else:
        boxes, classes, scores = infer(climate, input_data)

    frame = draw(frame, boxes, scores, expert)

    cv2.imshow("MoE System", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()