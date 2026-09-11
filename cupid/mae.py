"""ViT Masked Autoencoder used by CUPID, plus the checkpoint loader.

Inference-only port of the research code: the encoder produces the CLS-token
identity descriptor; the decoder produces video-level identity analogy maps
and the demo's framewise overlays.
"""

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=12,
        num_head=3,
    ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward_no_masking(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )

        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        return features


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        num_layer=4,
        num_head=3,
        use_conv_layers=False,
    ) -> None:
        super().__init__()

        # Unused in inference; retain for strict released-checkpoint compatibility.
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim)
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=image_size // patch_size,
        )

        # Optional: Add convolutional layers to reduce grid artifacts
        self.use_conv_layers = use_conv_layers
        if use_conv_layers:
            self.conv_refine = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
            )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward_no_masking(self, features):
        features = features + self.pos_embedding

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = self.head(features)
        img = self.patch2img(patches)

        # Apply convolutional refinement if enabled
        if self.use_conv_layers:
            img = self.conv_refine(img)

        return img


class MAE_ViT(torch.nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=2,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
        use_conv_decoder=False,
    ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(
            image_size, patch_size, emb_dim, encoder_layer, encoder_head
        )
        self.decoder = MAE_Decoder(
            image_size, patch_size, emb_dim, decoder_layer, decoder_head, use_conv_layers=use_conv_decoder
        )


def load_trained_model(checkpoint_path, device="cuda:0"):
    """
    Load a trained MAE-ViT model from an inference checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        model: Loaded MAE_ViT model (eval mode)
        args_dict: Training arguments stored in the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "args" in checkpoint:
        args_dict = checkpoint["args"]
    else:
        raise ValueError("No args found in checkpoint. Cannot reconstruct model.")

    # The fallback defaults below are load-bearing: released checkpoints omit
    # keys that were never changed from the research-code defaults.
    model = MAE_ViT(
        image_size=224,
        patch_size=args_dict.get("patch_size", 16),
        emb_dim=args_dict.get("emb_dim", 192),
        encoder_layer=args_dict.get("num_layers", 12),
        encoder_head=args_dict.get("encoder_head", 3),
        decoder_layer=args_dict.get("decoder_layer", 4),
        decoder_head=args_dict.get("decoder_head", 3),
        use_conv_decoder=args_dict.get("use_conv_decoder", False),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, args_dict
