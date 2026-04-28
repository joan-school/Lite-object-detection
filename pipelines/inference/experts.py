
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
