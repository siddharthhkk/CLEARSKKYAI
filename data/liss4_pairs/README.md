# LISS-IV paired training data

`manifest.csv` is the input list for `scripts/train_liss4_paired.py`. It is a
header-only template until real cloudy/clear pairs are added. The trainer
cannot start with an empty manifest.

Add one row per verified pair:

```csv
cloudy,clear,aoi,split,mask
train/cloudy_scene.tif,train/clear_scene.tif,scene_01,train,
val/cloudy_scene.tif,val/clear_scene.tif,scene_02,val,
```

- `cloudy` and `clear` must point to three-band GeoTIFF stacks ordered Green,
  Red, NIR. Paths are relative to this manifest or can be absolute.
- Each cloudy/clear pair must be co-registered and have matching CRS,
  transform, dimensions, and band order.
- `aoi` identifies a source scene or geographic area. Keep each AOI in only
  one split; both `train` and `val` rows are required.
- `mask` is optional. If supplied, it must be a co-registered one-band cloud
  mask where nonzero pixels are cloudy.

The raw imagery currently in this checkout does not include a verified
clear target paired with the cloudy scenes in `data/liss4_demo/`. There is a
much clearer-looking 110/54 acquisition from 2023-11-01 that overlaps the
cloudy 2020-06-01 scene, but the 3.5-year gap makes it a temporal pseudo-pair,
not reliable pixel-level ground truth. The project's cloudiness score is a
heuristic, not a cloud mask or proof of clear conditions. Do not use that pair
as a validated training target without inspecting and explicitly accepting
those limitations.

For separate source bands, stack BAND2 (Green), BAND3 (Red), and BAND4 (NIR)
for each acquisition with `scripts/stack_liss4_bands.py`. Make sure each
cloudy/clear acquisition is already aligned to the same grid before training.
