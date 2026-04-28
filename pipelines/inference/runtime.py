
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
