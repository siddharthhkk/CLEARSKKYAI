import argparse
import base64
import io
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser(
        description="Embed the Bhoonidhi pair preview as a compact UTF-8 SVG."
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="previews/bhoonidhi_pair_preview.png",
    )
    ap.add_argument(
        "output",
        nargs="?",
        default="previews/bhoonidhi_pair_preview_embed.svg",
    )
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--quality", type=int, default=65)
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    out = os.path.abspath(args.output)

    if not os.path.isfile(inp):
        raise SystemExit(f"FAIL: input image not found: {inp}")

    img = Image.open(inp).convert("RGB")
    scale = args.width / img.width
    h = max(1, round(img.height * scale))
    img = img.resize((args.width, h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=args.quality, optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}">
<image width="{img.width}" height="{img.height}" href="data:image/jpeg;base64,{data}" />
</svg>
'''

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Input  : {inp}")
    print(f"Output : {out}")
    print(f"Size   : {img.width}x{img.height}")
    print(f"Bytes  : {os.path.getsize(out):,}")


if __name__ == "__main__":
    main()
