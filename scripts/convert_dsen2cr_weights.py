import argparse
import re

import h5py
import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsen2cr import DSen2CR


def dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def natural_key(x):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", x)]


def find_model_weights(h5):
    if "model_weights" in h5:
        return h5["model_weights"]
    return h5


def collect_conv_weights(h5):
    root = find_model_weights(h5)
    pairs = []

    # Preferred path: Keras HDF5 layer metadata preserves layer order.
    if "layer_names" in root.attrs:
        names = [dec(v) for v in root.attrs["layer_names"]]
        for name in names:
            if name not in root:
                continue
            g = root[name]
            if "weight_names" not in g.attrs:
                continue

            cur = {}
            for wn in g.attrs["weight_names"]:
                wn = dec(wn)
                short = wn.split("/")[-1]

                ds = None
                if wn in g:
                    ds = g[wn]
                elif wn in root:
                    ds = root[wn]
                elif short in g:
                    ds = g[short]

                if ds is None:
                    continue

                if short.startswith("kernel"):
                    cur["k"] = np.asarray(ds)
                elif short.startswith("bias"):
                    cur["b"] = np.asarray(ds)

            if "k" in cur and "b" in cur:
                pairs.append((name, cur["k"], cur["b"]))

    # Fallback for other Keras HDF5 layouts.
    if not pairs:
        def visitor(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if not name.endswith("/kernel:0") and not name.endswith("/bias:0"):
                return

        groups = {}

        def walk(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.endswith("/kernel:0"):
                base = name.rsplit("/", 1)[0]
                groups.setdefault(base, {})["k"] = np.asarray(obj)
            elif name.endswith("/bias:0"):
                base = name.rsplit("/", 1)[0]
                groups.setdefault(base, {})["b"] = np.asarray(obj)

        root.visititems(walk)
        for name in sorted(groups, key=natural_key):
            cur = groups[name]
            if "k" in cur and "b" in cur:
                pairs.append((name, cur["k"], cur["b"]))

    # The architecture has exactly 34 convolution layers:
    # 1 input + 16*2 residual + 1 output.
    pairs = [
        p for p in pairs
        if isinstance(p[1], np.ndarray)
        and p[1].ndim == 4
        and p[1].shape[0] == 3
        and p[1].shape[1] == 3
    ]

    if len(pairs) != 34:
        raise RuntimeError(
            f"Expected 34 convolution layers in DSen2-CR checkpoint, found {len(pairs)}. "
            "The HDF5 checkpoint layout may differ from the original Keras format."
        )

    return pairs


def convert(src, dst):
    model = DSen2CR()
    convs = [model.inp[0]]
    for b in model.blocks:
        convs.extend([b.c1, b.c2])
    convs.append(model.out)

    with h5py.File(src, "r") as h5:
        pairs = collect_conv_weights(h5)

    if len(convs) != len(pairs):
        raise RuntimeError("Model/checkpoint convolution count mismatch.")

    for i, (_, kernel, bias) in enumerate(pairs):
        expected = tuple(convs[i].weight.shape)

        # Keras Conv2D stores kernels as [H,W,in,out].
        kernel = np.transpose(kernel, (3, 2, 0, 1))

        if tuple(kernel.shape) != expected:
            raise RuntimeError(
                f"Layer {i}: checkpoint kernel shape {tuple(kernel.shape)} "
                f"does not match PyTorch shape {expected}."
            )

        if tuple(bias.shape) != tuple(convs[i].bias.shape):
            raise RuntimeError(
                f"Layer {i}: checkpoint bias shape {tuple(bias.shape)} "
                f"does not match PyTorch shape {tuple(convs[i].bias.shape)}."
            )

        convs[i].weight.data.copy_(torch.from_numpy(kernel).float())
        convs[i].bias.data.copy_(torch.from_numpy(bias).float())

    torch.save(model.state_dict(), dst)

    print(f"✅ Converted DSen2-CR checkpoint")
    print(f"   Input : {src}")
    print(f"   Output: {dst}")
    print("   Layers: 34 convolution layers")


def main():
    p = argparse.ArgumentParser(
        description="Convert the public DSen2-CR Keras HDF5 checkpoint to PyTorch."
    )
    p.add_argument("--src", required=True, help="Path to model_SARcarl.hdf5")
    p.add_argument(
        "--dst",
        default=str(PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth"),
        help="Output PyTorch state-dict path.",
    )
    args = p.parse_args()

    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
