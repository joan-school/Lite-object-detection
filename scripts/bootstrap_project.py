import os

# ==============================
# FOLDER STRUCTURE
# ==============================

folders = [
    "apps/mediapipe/graph",
    "apps/mediapipe/demo",
    "apps/opencv/demo",

    "data/raw",
    "data/processed",
    "data/splits",
    "data/labels",
    "data/feedback",

    "models/backbone",
    "models/router",
    "models/scene_classifier",
    "models/experts/display",
    "models/experts/kitchen",
    "models/experts/climate",
    "models/experts/utility",

    "pipelines/training/backbone",
    "pipelines/training/experts",
    "pipelines/training/router",
    "pipelines/training/scene",
    "pipelines/inference",

    "postprocess",
    "utils",
    "scripts",
    "outputs",
    "docs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ==============================
# CREATE FILE HELPER
# ==============================

def create_file(path, content=""):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# ==============================
# CONFIG FILE
# ==============================

create_file("utils/config.py", """
CLASSES = [
    "tv",
    "refrigerator",
    "air_conditioner",
    "washing_machine",
    "microwave",
    "dishwasher",
    "robot_vacuum",
    "air_purifier"
]

EXPERT_MAP = {
    "display": ["tv"],
    "kitchen": ["refrigerator", "microwave", "dishwasher"],
    "climate": ["air_conditioner", "air_purifier"],
    "utility": ["washing_machine", "robot_vacuum"]
}

EXPERT_IDS = {
    "display": 0,
    "kitchen": 1,
    "climate": 2,
    "utility": 3
}
""")

# ==============================
# PREPROCESS
# ==============================

create_file("utils/preprocess.py", """
import cv2

def preprocess(frame):
    img = cv2.resize(frame, (320, 320))
    img = img / 255.0
    return img
""")

# ==============================
# NMS
# ==============================

create_file("utils/nms.py", """
def nms(detections):
    # TODO: implement proper NMS
    return detections
""")

# ==============================
# ROUTER
# ==============================

create_file("pipelines/inference/router.py", """
class Router:
    def __init__(self):
        pass

    def select(self, gap_vector):
        # TODO: load router TFLite model
        return 0
""")

# ==============================
# EXPERTS
# ==============================

create_file("pipelines/inference/experts.py", """
class ExpertEnsemble:
    def __init__(self):
        self.models = {
            0: "models/experts/display/model.tflite",
            1: "models/experts/kitchen/model.tflite",
            2: "models/experts/climate/model.tflite",
            3: "models/experts/utility/model.tflite",
        }

    def run(self, expert_id, features):
        model_path = self.models[expert_id]
        # TODO: load + run TFLite model
        return []
""")

# ==============================
# SCENE CLASSIFIER
# ==============================

create_file("pipelines/inference/scene.py", """
class SceneClassifier:
    def __init__(self):
        pass

    def run(self, gap):
        return "unknown", 0.0
""")

# ==============================
# FUSION
# ==============================

create_file("pipelines/inference/fusion.py", """
def fuse(detections, room, room_conf):
    results = []

    for det in detections:
        results.append({
            "object": det.get("label", "unknown"),
            "room": room,
            "confidence": det.get("confidence", 0) * room_conf,
            "bbox": det.get("bbox", [])
        })

    return results
""")

# ==============================
# POSTPROCESS
# ==============================

create_file("postprocess/postprocess.py", """
def confidence_gate(results, threshold=0.5):
    return [r for r in results if r["confidence"] > threshold]
""")

# ==============================
# RUNTIME PIPELINE
# ==============================

create_file("pipelines/inference/runtime.py", """
from pipelines.inference.router import Router
from pipelines.inference.experts import ExpertEnsemble
from pipelines.inference.scene import SceneClassifier
from pipelines.inference.fusion import fuse

class InferenceEngine:
    def __init__(self):
        self.router = Router()
        self.experts = ExpertEnsemble()
        self.scene = SceneClassifier()

    def run(self, frame):
        feat_maps, gap = self.extract_features(frame)

        expert_id = self.router.select(gap)
        detections = self.experts.run(expert_id, feat_maps)

        room, room_conf = self.scene.run(gap)

        results = fuse(detections, room, room_conf)
        return results

    def extract_features(self, frame):
        # TODO: integrate backbone TFLite model
        return None, None
""")

# ==============================
# OPENCV DEMO
# ==============================

create_file("apps/opencv/demo/main.py", """
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
""")

# ==============================
# README
# ==============================

create_file("README.md", """
# Samsung MoE Vision System

Modular on-device computer vision system using:

- Mixture of Experts (MoE)
- Scene Context Awareness
- TFLite + OpenCV / MediaPipe

Supports:
TV, Refrigerator, AC, Washing Machine, Microwave,
Dishwasher, Robot Vacuum, Air Purifier
""")

# ==============================
# DONE
# ==============================

print("🚀 Full project skeleton + starter files created!")