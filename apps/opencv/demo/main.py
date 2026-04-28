
import cv2
from pipelines.inference.runtime import InferenceEngine

engine = InferenceEngine()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = engine.run(frame)

    for r in results:
        x1, y1, x2, y2 = r.get("bbox", [0,0,0,0])
        label = f'{r["object"]} ({r["room"]})'

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("MoE Vision System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
