# LISS-IV research track

The Sentinel-2 application remains at the repository root. Its model and
scripts stay in the root `src/` and `scripts/` folders. This directory holds
the separate LISS-IV implementation:

```text
liss4/
├── src/       # LISS-IV model, normalization, datasets, and transfer adapter
└── scripts/   # LISS-IV training, preparation, inference, evaluation, checks
```

The Streamlit entry point remains `app_liss4.py` at the repository root. Run
commands from the repository root so data and checkpoint paths resolve as
documented:

```powershell
streamlit run app_liss4.py
python liss4/scripts/inspect_liss4.py path\to\cloudy_liss4.tif
```

LISS-IV imagery uses `[Green, Red, NIR]` bands, typically `BAND2`, `BAND3`,
and `BAND4`. The current checkpoint was trained with synthetic cloud
corruption. Real paired fine-tuning is implemented, but requires co-registered
cloudy/clear LISS-IV GeoTIFF pairs and separate train/validation areas; the
manifest at `data/liss4_pairs/manifest.csv` is currently a blank template.

See the [full training and experiment guide](../PROJECT_GUIDE.md#liss-iv-reproduction-and-checks-legacy)
for data preparation, training, inference, and evaluation details.
