from .dinov2 import load_dinov2

_BACKBONE_LOADERS = {"dinov2_vitl14_reg": load_dinov2}

def get_backbone(name: str, **kwargs):
    if name not in _BACKBONE_LOADERS:
        raise ValueError(f"Unknown backbone: {name}")
    return _BACKBONE_LOADERS[name](**kwargs)