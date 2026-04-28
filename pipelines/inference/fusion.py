
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
