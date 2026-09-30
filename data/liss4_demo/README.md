# LISS-IV demo sample

Each dropdown option is a 512 × 512 crop from a cloudy Resourcesat-2 LISS-IV scene. The gallery spans these acquisitions:

- 1 June 2020, Path 110 / Row 53 (`R2F01JUN2020047314011000053SSANSTUC00GTDD`)
- 1 June 2020, Path 110 / Row 54 (`R2F01JUN2020047314011000054SSANSTUC00GTDB`)
- 8 June 2020, Path 110 / Row 54 (`RAF08JUN2020018173011000054SSANSTUC00GTDB`)
- 12 August 2020, Path 110 / Row 54 (`R2F12AUG2020048337011000054SSANSTUC00GTDB`)
- 2 August 2023, Path 111 / Row 53 (`R2F02AUG2023063753011100053SSANSTUC00GTDC`)

Every crop contains three stacked bands in the project order: Green (`BAND2`), Red (`BAND3`), and NIR (`BAND4`). Pixel values retain the source product's 10-bit digital-number range.

The 512 × 512 crop windows start at these source pixel coordinates (column, row):

| Acquisition | Column | Row |
|---|---:|---:|
| 1 Jun 2020, Path 110 / Row 53 | 15,336 | 9,192 |
| 1 Jun 2020, Path 110 / Row 54 | 11,000 | 2,000 |
| 8 Jun 2020, Path 110 / Row 54 | 7,144 | 7,144 |
| 12 Aug 2020, Path 110 / Row 54 | 11,240 | 11,240 |
| 2 Aug 2023, Path 111 / Row 53 | 9,192 | 15,336 |

The crops are supplied as convenient inference inputs, not as paired clear references or a validated benchmark. The full source scenes and evaluation rasters remain local and ignored.

The source product metadata identifies IRS-R2 / LISS-IV and the acquisition date. Refer to the terms supplied with the original Bhoonidhi/NRSC product when redistributing this imagery.
