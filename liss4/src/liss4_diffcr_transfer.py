"""Adapter for fine-tuning the DiffCR baseline used by the linked LISS-IV repo."""

import importlib.util
import os

import torch
import torch.nn as nn


NETWORK_RELATIVE_PATH = os.path.join(
    "models",
    "ours",
    "nafnet_double_encoder_splitcaCond_splitcaUnet.py",
)


def resolve_diffcr_root(path=None):
    """Find a DiffCR source checkout, including the AllClear submodule layout."""
    candidates = []
    if path:
        candidates.append(os.path.abspath(path))
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.extend(
            os.path.join(project_root, relative)
            for relative in (
                "external/DiffCR",
                "external/allclear/baselines/DiffCR",
                "allclear/baselines/DiffCR",
                "baselines/DiffCR",
            )
        )

    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, NETWORK_RELATIVE_PATH)):
            return candidate
    attempted = "\n".join(candidates) if candidates else "(none)"
    raise FileNotFoundError(
        "Could not find the DiffCR model source. Clone the DiffCR repository or "
        "pass --diffcr-root pointing to its repository. "
        "Checked:\n" + attempted
    )


def import_diffcr_unet(diffcr_root):
    model_file = os.path.join(diffcr_root, NETWORK_RELATIVE_PATH)
    spec = importlib.util.spec_from_file_location(
        "clearsky_diffcr_unet", model_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import DiffCR network at {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNet


def extract_denoiser_state(checkpoint):
    """Extract compatible tensor weights from common DiffCR checkpoint layouts."""
    state = checkpoint
    if isinstance(state, dict):
        for name in (
            "state_dict",
            "model_state_dict",
            "params_ema",
            "params",
            "denoise_fn",
            "ema_model",
        ):
            candidate = state.get(name)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise ValueError("The DiffCR checkpoint does not contain a state dictionary")

    cleaned = {}
    for original_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "denoise_fn."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    if not cleaned:
        raise ValueError("No tensor weights were found in the DiffCR checkpoint")
    return cleaned


class LISS4DiffCRTransfer(nn.Module):
    """Three-band LISS-IV adapter around a pretrained DiffCR denoiser.

    The pretrained network consumes three condition frames plus a noisy image.
    With single-date LISS-IV inputs, this adapter repeats the same image for
    those four inputs, as the linked LISS-IV transfer repository does. Its
    Sentinel-style [0, 1] inputs are converted to the DiffCR [-1, 1] range.
    This is the DiffCR route used by the linked LISS-IV repository. The common
    ``diffcr_new.pth`` checkpoint is configured for Sen2_MTC_New in its source
    repository; it is not the AllClear project's separate UnCRtainTS model.
    """

    def __init__(self, diffcr_root=None, pretrained_checkpoint=None):
        super().__init__()
        self.diffcr_root = resolve_diffcr_root(diffcr_root)
        unet_class = import_diffcr_unet(self.diffcr_root)
        self.backbone = unet_class(
            img_channel=3,
            width=64,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 1],
            dec_blk_nums=[1, 1, 1, 1],
        )
        self.pretrained_report = None
        if pretrained_checkpoint:
            self.pretrained_report = self.load_pretrained(pretrained_checkpoint)

    def load_pretrained(self, checkpoint_path):
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        source = extract_denoiser_state(checkpoint)
        target = self.backbone.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        required = ("intro.weight", "cond_intro.weight", "ending.weight")
        missing_required = [key for key in required if key not in compatible]
        matched_values = sum(value.numel() for value in compatible.values())
        total_values = sum(value.numel() for value in target.values())
        coverage = matched_values / max(1, total_values)
        if missing_required or coverage < 0.90:
            raise ValueError(
                "This checkpoint does not match the DiffCR network: "
                f"matched {len(compatible)}/{len(target)} tensors "
                f"({coverage:.1%} of parameters); missing required keys: "
                f"{missing_required}"
            )

        self.backbone.load_state_dict(compatible, strict=False)
        return {
            "checkpoint": os.path.abspath(checkpoint_path),
            "matched_tensors": len(compatible),
            "total_tensors": len(target),
            "parameter_coverage": coverage,
        }

    def forward(self, cloudy):
        if cloudy.ndim != 4 or cloudy.shape[1] != 3:
            raise ValueError(
                f"cloudy must have shape [B,3,H,W], got {tuple(cloudy.shape)}"
            )
        normalized = cloudy * 2.0 - 1.0
        denoiser_input = torch.cat(
            (normalized, normalized, normalized, normalized), dim=1
        )
        timestep = torch.zeros(cloudy.shape[0], device=cloudy.device)
        reconstructed = self.backbone(denoiser_input, timestep)
        return (reconstructed + 1.0) * 0.5
