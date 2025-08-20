
# TODO: we should really centralize this somewhere
def feature_to_label(feature: str) -> str:
    if feature == "asr":
        return "Speech to Text"
    if feature == "caption":
        return "Object Detection"
    if feature == "celeb":
        return "Celebrity Detection"
    if feature == "logo":
        return "Logo Detection"
    if feature == "music":
        return "Music Detection"
    if feature == "ocr":
        return "Optical Character Recognition"
    if feature == "shot":
        return "Shot Detection"
    if feature == "llava":
        return "LLAVA Caption"
    return feature.replace("_", " ").title()

# e.g. "Shot Tags" -> "shot_tags"
def label_to_track(label: str) -> str:
    return label.lower().replace(" ", "_")