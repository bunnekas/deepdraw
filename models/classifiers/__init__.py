from .dino_classifier import DinoClassifier

_CLASSIFIERS = {"dinov2_vitl14_reg": DinoClassifier}

def get_classifier(backbone_name, backbone, feature_dim, num_classes):
    if backbone_name not in _CLASSIFIERS:
        raise ValueError(f"No classifier registered for backbone '{backbone_name}'")
    return _CLASSIFIERS[backbone_name](backbone, feature_dim, num_classes)