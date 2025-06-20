import torch
import open_clip

def load_clip(freeze: bool = False):
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    backbone = model.visual
    feature_dim = 1024  # oder ggf. 768, je nach Clip-Modell

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, feature_dim