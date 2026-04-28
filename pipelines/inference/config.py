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
