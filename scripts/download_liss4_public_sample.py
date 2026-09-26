import argparse
import os
import urllib.request


URL = (
    "https://huggingface.co/spaces/kk947/LISS-IV-Cloud-Removal/"
    "resolve/main/guwahati_cloudy_test.tif?download=true"
)


def main():
    ap = argparse.ArgumentParser(
        description="Download a public LISS-IV sample for Phase-1 pipeline inspection."
    )
    ap.add_argument(
        "--output",
        default="data/raw/cloudy/guwahati_cloudy_test.tif",
    )
    args = ap.parse_args()

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("Source : Hugging Face public LISS-IV sample")
    print(f"URL    : {URL}")
    print(f"Output : {out}")

    try:
        urllib.request.urlretrieve(URL, out)
    except Exception as exc:
        raise RuntimeError(
            "Download failed. The source may be temporarily unavailable, "
            "or the network may block Hugging Face. Open the source URL in a "
            "browser and save the file to the requested output path."
        ) from exc

    if not os.path.isfile(out) or os.path.getsize(out) == 0:
        raise RuntimeError("Downloaded file is missing or empty.")

    print(f"PASS downloaded sample ({os.path.getsize(out) / 1024**2:.2f} MiB)")
    print(
        "NOTE: this is a cloudy LISS-IV sample for file-format/inference "
        "testing. It is NOT a verified cloudy/clear training pair."
    )


if __name__ == "__main__":
    main()
