import argparse
import csv
import datetime as dt
import os
import re
import xml.etree.ElementTree as ET

import rasterio


DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
FOLDER_DATE_RE = re.compile(
    r"(?<!\d)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(20\d{2})(?!\d)",
    re.IGNORECASE,
)
PATH_ROW_RE = re.compile(r"(?<!\d)(\d{3})[_-](\d{2})(?!\d)", re.IGNORECASE)
# Bhoonidhi/ISRO product folder names seen in this batch contain:
#   ...<3-digit path><4-digit row>SSANSTUC00GT...
FOLDER_PATH_ROW_RE = re.compile(
    r"(\d{3})(\d{5})SSANSTUC00GT",
    re.IGNORECASE,
)
SCENE_RE = re.compile(r"(\d{6}[_-]\d{3}[_-]\d{2})", re.IGNORECASE)


def clean_text(x):
    return " ".join(str(x).split())


def parse_date(text):
    if not text:
        return ""

    m = DATE_RE.search(text)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return dt.date(y, mo, d).isoformat()
        except ValueError:
            pass

    m = FOLDER_DATE_RE.search(text)
    if m:
        d, mon, y = m.groups()
        try:
            return dt.datetime.strptime(
                f"{d}{mon}{y}", "%d%b%Y"
            ).date().isoformat()
        except ValueError:
            pass

    return ""


def parse_path_row(text):
    if not text:
        return "", ""

    m = PATH_ROW_RE.search(text)
    if m:
        return m.group(1), m.group(2)

    m = FOLDER_PATH_ROW_RE.search(text)
    if m:
        return m.group(1), str(int(m.group(2)))

    return "", ""


def parse_scene_id(text):
    if not text:
        return ""

    m = SCENE_RE.search(text)
    if m:
        return m.group(1).replace("-", "_")

    return ""


def parse_folder_metadata(folder_name):
    # Bhoonidhi product folders in this batch encode the acquisition date
    # and a zero-padded path/row near the end of the product ID. Example:
    # ...011000053SSANSTUC... -> path 110, row 53
    date = parse_date(folder_name)
    path_no, row_no = parse_path_row(folder_name)

    return {
        "date": date,
        "path": path_no,
        "row": row_no,
        "scene_id": folder_name,
        "satellite": "ResourceSat-2" if folder_name.upper().startswith("R2F") else "",
        "sensor": "",
        "product_hint": folder_name[:4] if len(folder_name) >= 4 else "",
    }


def flatten_xml_values(root):
    vals = []

    for elem in root.iter():
        if elem.text and elem.text.strip():
            vals.append(f"{elem.tag}={clean_text(elem.text)}")

        for k, v in elem.attrib.items():
            vals.append(f"{k}={clean_text(v)}")

    return vals


def read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def read_metadata(folder, filenames):
    chunks = []

    for name in filenames:
        low = name.lower()
        path = os.path.join(folder, name)

        if low.endswith(".xml"):
            try:
                root = ET.parse(path).getroot()
                chunks.extend(flatten_xml_values(root))
            except Exception:
                chunks.append(read_text_file(path))
        elif low.endswith((".txt", ".json", ".mtl", ".met")):
            chunks.append(read_text_file(path))

    text = "\n".join(chunks)

    date = parse_date(text)
    path_no, row_no = parse_path_row(text)
    scene_id = parse_scene_id(text)

    sat = ""
    sensor = ""
    product = ""

    lower = text.lower()

    for candidate in [
        "resourcesat-2a",
        "resourcesat-2",
        "resourcessat-2a",
        "rs2a",
        "rs2",
    ]:
        if candidate in lower:
            sat = candidate
            break

    if "liss4" in lower or "lis4" in lower:
        sensor = "LISS4"

    for candidate in ["R2FE", "R2AF", "R2A"]:
        if candidate.lower() in lower:
            product = candidate
            break

    return {
        "date": date,
        "satellite": sat,
        "sensor": sensor,
        "path": path_no,
        "row": row_no,
        "scene_id": scene_id,
        "product_hint": product,
    }


def tif_info(path):
    try:
        with rasterio.open(path) as src:
            return {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0] if src.count else "",
                "crs": str(src.crs) if src.crs else "",
                "xres": abs(float(src.transform.a)),
                "yres": abs(float(src.transform.e)),
                "left": float(src.bounds.left),
                "bottom": float(src.bounds.bottom),
                "right": float(src.bounds.right),
                "top": float(src.bounds.top),
                "driver": src.driver,
            }
    except Exception as exc:
        return {
            "width": "",
            "height": "",
            "count": "",
            "dtype": "",
            "crs": "",
            "xres": "",
            "yres": "",
            "left": "",
            "bottom": "",
            "right": "",
            "top": "",
            "driver": f"ERROR: {exc}",
        }


def classify_band(name):
    low = name.lower()

    if "band2" in low or re.search(
        r"(?:^|[_-])b2(?:nd)?(?:[_-]|\.)", low
    ):
        return "BAND2"
    if "band3" in low or re.search(
        r"(?:^|[_-])b3(?:nd)?(?:[_-]|\.)", low
    ):
        return "BAND3"
    if "band4" in low or re.search(
        r"(?:^|[_-])b4(?:nd)?(?:[_-]|\.)", low
    ):
        return "BAND4"

    return ""


def scan(root):
    groups = {}

    for dirpath, _, filenames in os.walk(root):
        tifs = [
            f for f in filenames
            if f.lower().endswith((".tif", ".tiff"))
        ]

        if not tifs:
            continue

        folder_name = os.path.basename(os.path.normpath(dirpath))
        meta = parse_folder_metadata(folder_name)

        # Real metadata files override folder-name fallbacks when present.
        file_meta = read_metadata(dirpath, filenames)

        for k, v in file_meta.items():
            if v:
                meta[k] = v

        # Always keep the actual extracted product folder as a stable scene ID
        # when no scene ID is present in metadata.
        if not meta["scene_id"] or meta["scene_id"] == "":
            meta["scene_id"] = folder_name

        folder_key = os.path.abspath(dirpath)

        info = groups.setdefault(
            folder_key,
            {
                "folder": folder_key,
                "tifs": [],
                "metadata": meta,
            },
        )

        for name in sorted(tifs):
            path = os.path.join(dirpath, name)
            ti = tif_info(path)

            info["tifs"].append(
                {
                    "name": name,
                    "path": path,
                    "band": classify_band(name),
                    **ti,
                }
            )

    return groups


def list_top_level_dirs_without_tif(root):
    out = []

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)

        if not os.path.isdir(path):
            continue

        found = False

        for dirpath, _, filenames in os.walk(path):
            if any(
                f.lower().endswith((".tif", ".tiff"))
                for f in filenames
            ):
                found = True
                break

        if not found:
            out.append(name)

    return out


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Scan extracted Bhoonidhi LISS-IV downloads, including product "
            "folder-name metadata, GeoTIFF metadata, and scene footprints."
        )
    )
    ap.add_argument(
        "root",
        nargs="?",
        default="data/raw/bhoonidhi_liss4",
    )
    ap.add_argument(
        "--csv",
        default="data/raw/bhoonidhi_liss4/scene_inventory.csv",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)

    if not os.path.isdir(root):
        raise SystemExit(f"FAIL: folder not found: {root}")

    groups = scan(root)

    if not groups:
        raise SystemExit(
            "FAIL: no GeoTIFF files found below the supplied folder."
        )

    rows = []

    for folder, item in sorted(groups.items()):
        meta = item["metadata"]
        tifs = item["tifs"]

        bands = sorted(
            {x["band"] for x in tifs if x["band"]},
            key=lambda x: {
                "BAND2": 0,
                "BAND3": 1,
                "BAND4": 2,
            }.get(x, 99),
        )

        # Recover from TIFF filenames as a final fallback.
        if not meta["date"]:
            for x in tifs:
                meta["date"] = parse_date(x["name"])
                if meta["date"]:
                    break

        if not meta["path"] or not meta["row"]:
            for x in tifs:
                p, r = parse_path_row(x["name"])
                if p:
                    meta["path"] = p
                    meta["row"] = r
                    break

        if not meta["sensor"]:
            # These downloads came from the LISS-IV search collection, and
            # BAND2/BAND3/BAND4 are the expected LISS-IV raster bands.
            if {"BAND2", "BAND3", "BAND4"}.issubset(set(bands)):
                meta["sensor"] = "LISS4"
            elif bands:
                meta["sensor"] = "LISS4? (partial bands)"

        ti = tifs[0]

        rows.append(
            {
                "folder": folder,
                "scene_id": meta["scene_id"],
                "date": meta["date"],
                "satellite": meta["satellite"],
                "sensor": meta["sensor"],
                "path": meta["path"],
                "row": meta["row"],
                "product_hint": meta["product_hint"],
                "band_count_found": len(bands),
                "bands_found": ",".join(bands),
                "tif_count": len(tifs),
                "width": ti["width"],
                "height": ti["height"],
                "count_first_tif": ti["count"],
                "dtype": ti["dtype"],
                "xres": ti["xres"],
                "yres": ti["yres"],
                "crs": ti["crs"],
                "left": ti["left"],
                "bottom": ti["bottom"],
                "right": ti["right"],
                "top": ti["top"],
                "tif_files": ";".join(x["name"] for x in tifs),
                "tif_paths": ";".join(x["path"] for x in tifs),
            }
        )

    out = os.path.abspath(args.csv)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fields = list(rows[0].keys())

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    missing = list_top_level_dirs_without_tif(root)

    print("=== BHOONIDHI LISS-IV SCENE INVENTORY ===")
    print(f"Root scanned       : {os.path.abspath(root)}")
    print(f"Scene groups found : {len(rows)}")
    print(f"Inventory CSV      : {out}")
    print()

    for i, row in enumerate(rows, start=1):
        print(
            f"[{i:02d}] "
            f"scene={row['scene_id'] or '?'} | "
            f"date={row['date'] or '?'} | "
            f"sat={row['satellite'] or '?'} | "
            f"sensor={row['sensor'] or '?'} | "
            f"path/row={row['path'] or '?'}/{row['row'] or '?'} | "
            f"bands={row['bands_found'] or '?'} | "
            f"size={row['width']}x{row['height']} | "
            f"files={row['tif_count']} | "
            f"bounds=({row['left']:.4f},{row['bottom']:.4f},"
            f"{row['right']:.4f},{row['top']:.4f})"
        )

    if missing:
        print()
        print("Top-level folders with NO GeoTIFF found:")
        for name in missing:
            print(f"  - {name}")
        print(
            "\nThis explains why the scene count may be lower than the "
            "number of ZIPs/folders you downloaded."
        )

    print()
    print(
        "Next: use the CSV + footprints to match scenes sharing the same "
        "area, then inspect dates/cloud content for clear/cloudy pairs."
    )


if __name__ == "__main__":
    main()
