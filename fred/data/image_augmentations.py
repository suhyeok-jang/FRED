"""Per-view resize and video frame augmentation for WAMModel (one param set per clip, shared across views and frames).

Resize (``build_view_resize``): every view to the processor's (H, W) cell — cv2 INTER_LINEAR, or torch bilinear
(bit-matches the DiT4DiT-official LIBERO pipeline); optionally center-cropping the source to the target aspect
first (``data.video_crop_aspect``).

Train (``data.image_aug``), applied per view AFTER the resize:
    resized crop (area 0.9, aspect-preserving) → [rotate U(−deg, deg) when ``rotate_deg`` > 0] →
    brightness/contrast/saturation U(1−x, 1+x) → hue U(−h, h)  [data.image_aug_{brightness,contrast,saturation,hue}]
  • rotate_deg=0: no rotation (LIBERO); rotate_deg=5: RoboCasa kitchen / GR-1 / DROID / OpenArm.
  Geometry and photometry are applied to EVERY view (wrist included).
Eval: deterministic center crop (area 0.9) → resize back; no rotation, no jitter.
"""

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision import transforms as TVT
from torchvision.transforms import functional as TVF


class TorchBilinearResize(ImageOnlyTransform):
    """Resize with ``F.interpolate(mode='bilinear', align_corners=False)`` (DiT4DiT-official LIBERO resize)."""

    def __init__(self, height: int, width: int, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.height = int(height)
        self.width = int(width)

    def apply(self, img, **params):
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        if img.dtype == np.uint8:
            t = t / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.height, self.width), mode="bilinear", align_corners=False).squeeze(0)
        return t.permute(1, 2, 0).contiguous().numpy().astype(np.float32)

    def get_transform_init_args_names(self):
        return ("height", "width")


class CenterCropToAspect(ImageOnlyTransform):
    """Center-crop a source frame to the target H:W aspect (``data.video_crop_aspect``)."""

    def __init__(self, height: int, width: int, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.height, self.width = int(height), int(width)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        tgt = self.width / self.height
        if w / h > tgt:  # source wider than target -> trim width
            nw = max(1, int(round(h * tgt)))
            x0 = (w - nw) // 2
            return img[:, x0:x0 + nw]
        nh = max(1, int(round(w / tgt)))
        y0 = (h - nh) // 2
        return img[y0:y0 + nh]

    def get_transform_init_args_names(self):
        return ("height", "width")


def build_view_resize(height: int, width: int, resize_backend: str = "cv2_linear", crop_aspect: bool = False):
    """Per-view resize to (height, width) as an albumentations Compose; crop_aspect center-crops the source to the
    target aspect first (zero distortion)."""
    tfs = [CenterCropToAspect(height, width)] if crop_aspect else []
    if resize_backend == "torch_bilinear":
        tfs.append(TorchBilinearResize(height, width, p=1.0))
    else:
        assert resize_backend == "cv2_linear", f"resize_backend must be 'cv2_linear' or 'torch_bilinear', got {resize_backend!r}"
        tfs.append(A.Resize(int(height), int(width), interpolation=cv2.INTER_LINEAR, p=1.0))
    return A.Compose(tfs)


def apply_resize(transform, frames) -> torch.Tensor:
    """Apply the resize Compose to every (H, W, C) frame of a clip -> (T, C, H, W) uint8 tensor (a float32
    [0, 1] output is scaled back to uint8)."""
    out = []
    for img in frames:
        img_array = transform(image=np.array(img))["image"]
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        assert img_array.dtype == np.uint8, f"Unexpected data type: {img_array.dtype}"
        out.append(torch.from_numpy(img_array).permute(2, 0, 1))
    return torch.stack(out)


def sample_aug_params(height: int, width: int, rotate_deg: float, jitter: tuple) -> dict:
    """ONE param set for a clip. Draw order (torch global RNG): crop → angle → brightness → contrast →
    saturation → hue. Crop: torchvision RandomResizedCrop params with scale=(0.9, 0.9) and ratio fixed to
    the input aspect (W/H) so non-square inputs never fall back to a stretched square crop.
    ``jitter`` = (b, c, s, h): factors U(1−b, 1+b) / U(1−c, 1+c) / U(1−s, 1+s), hue U(−h, h) (torchvision ColorJitter semantics)."""
    ar = float(width) / float(height)
    i, j, h, w = TVT.RandomResizedCrop.get_params(img=torch.zeros(height, width), scale=(0.9, 0.9), ratio=(ar, ar))
    angle = torch.FloatTensor(1).uniform_(-rotate_deg, rotate_deg).item() if rotate_deg > 0 else 0.0
    jb, jc, js, jh = jitter
    brightness = torch.FloatTensor(1).uniform_(1.0 - jb, 1.0 + jb).item()
    contrast = torch.FloatTensor(1).uniform_(1.0 - jc, 1.0 + jc).item()
    saturation = torch.FloatTensor(1).uniform_(1.0 - js, 1.0 + js).item()
    hue = torch.FloatTensor(1).uniform_(-jh, jh).item()
    return {"i": i, "j": j, "h": h, "w": w, "angle": angle,
            "brightness": brightness, "contrast": contrast, "saturation": saturation, "hue": hue}


def apply_aug_train(frames_tchw_uint8: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply a sampled param set to (T, C, H, W) uint8 frames: resized crop → rotate → photometry."""
    assert frames_tchw_uint8.dtype == torch.uint8, f"expected uint8, got {frames_tchw_uint8.dtype}"
    _, _, H, W = frames_tchw_uint8.shape
    out = []
    for img in frames_tchw_uint8:  # (C, H, W)
        img = TVF.resized_crop(img, params["i"], params["j"], params["h"], params["w"], size=[H, W], antialias=True)
        if params["angle"] != 0.0:
            img = TVF.rotate(img, params["angle"], expand=False)
        img = TVF.adjust_brightness(img, params["brightness"])
        img = TVF.adjust_contrast(img, params["contrast"])
        img = TVF.adjust_saturation(img, params["saturation"])
        img = TVF.adjust_hue(img, params["hue"])
        out.append(img)
    return torch.stack(out)


def apply_aug_eval(frames_tchw_uint8: torch.Tensor) -> torch.Tensor:
    """Deterministic eval counterpart: center crop of area 0.9 (side sqrt(0.9)) → resize back to (H, W)."""
    assert frames_tchw_uint8.dtype == torch.uint8, f"expected uint8, got {frames_tchw_uint8.dtype}"
    _, _, H, W = frames_tchw_uint8.shape
    side_frac = 0.9 ** 0.5
    ch, cw = int(round(H * side_frac)), int(round(W * side_frac))
    top, left = (H - ch) // 2, (W - cw) // 2
    return torch.stack([TVF.resized_crop(img, top, left, ch, cw, size=[H, W], antialias=True) for img in frames_tchw_uint8])
