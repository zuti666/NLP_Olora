# Figure B self-review

Status: passed after rendering and inspecting the final PNG with `view_image`; independent root review remains separate.

- The requested four panels are present in a 2×2 layout: directional mean positive CE damage, directional 95th-percentile positive CE damage, raw Fisher mean/Q95, and the Gmean prefix-safe tolerance radius versus epsilon. Train and test are retained in (a), (b), and (d).
- Method display names are EWCLoRA and FS-EWCLoRA. The shared 10-point serif style, blue/orange colors, circle/diamond method markers, and filled/open train/test markers support reading in color and grayscale. The panel letters sit above the axes, clear of the simplified y labels.
- No conclusion, dataset footer, subplot title, or prose annotation is included in the image. The caption defines the ImageNet-A seed0 setup, Task0 anchor, sample counts, directional statistics, scale transforms, and tolerance censoring.
- The figure uses all original 28 neighborhood summary rows. Every signed/positive damage statistic was checked against the 194 raw states, together with raw Fisher means and recomputed linear Q95. All 40 reported Gmean/Q95 tolerance radii and censoring flags were independently recomputed from their sampled prefixes. The method/direction/data/coordinate hashes were checked against raw manifests.
- Exact tolerance ties remain at the original coordinates, shown using a broader blue line beneath a thinner orange line and nested markers. Censored train points retain nested upward triangles at radius 20. No jitter, curve displacement, smoothing, or measurement clipping was used.
- The Fisher sample maximum is preserved in the source CSV but omitted from the panel to reduce clutter; both mean and Q95 remain visible. The caption makes no claim of an enlarged tolerance radius, confidence interval, or worst-case guarantee.
- The output is a 7.6×6.05-inch vector PDF with embedded fonts and no raster images, plus a 2280×1815 PNG at 300 dpi. Text extents passed the shared canvas check. Source CSVs retain full precision and original CSV copies preserve byte hashes.
- The standalone redraw script only reads existing reports/raw JSON and writes new B artifacts. It does not load models, launch experiments, or access a GPU.

The original figures and raw data were not modified.
