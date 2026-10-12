Validation performed on 2026-09-20, using CPU only.

All generators completed successfully. Figure B recomputes signed damage, mean positive damage, and linearly interpolated positive-damage Q95 from the raw JSONL and matches every report row. It also recomputes the mean, Q95, and sampled maximum of raw Fisher trace. Both methods have the same 16 direction hashes at every positive radius; zero is one reused origin. Prefix-safe radii and censoring are recomputed from the complete sampled prefix. Both methods tie at every Gmean threshold on both splits; GAM test Gmean exceeds SGD at every positive radius. Those caption comparisons are explicitly checked.

Figure C verifies tasks 0 through 9 for both methods, the 20-class local test head and 307 examples, the base-relative loss changes, the effective endpoint displacement, and S_path = I / displacement squared. Both source numerical gates pass and both endpoint paths use 9 Simpson nodes. Only the measured t9 sensitivity is drawn. Table A retains both a22 values and the inherited-Fisher caveat. The existing bridge remains an appendix table with its historical-sequence direction identities.

The author reopened both rendered PNGs and inspected legibility, full data inclusion, legends, censor markers, tied curves, and panel labels. An initial B(c) panel-label/ylabel collision was fixed and the new render was inspected again. Tolerance tick crowding was reduced. The C legend now matches the dashed GAM curves. Root independently inspected the PNGs; the separate skill-required reviewer is managed by root.

Programmatic PDF inspection with PyMuPDF: both files are single-page documents with zero embedded image objects, embedded fonts, and vector drawings (B: 177 paths; C: 145 paths). No extracted text span falls outside its page. The PNG metadata reports 299.9994 dpi (300 dpi export), with dimensions B 2300 × 1936 and C 2268 × 1844 pixels. LaTeX snippets are generated with booktabs and graphicx syntax; a TeX compiler is unavailable in this environment, so the snippets have not been compiled here.

Machine-readable source hashes and completed numerical checks:

- `Figure_B_provenance.json`
- `Figure_C_provenance.json`
- `Tables_provenance.json`

Known evidence limits are retained in captions and README: one training seed; direction statistics are not training replicates; finite directions and radius grid do not give worst-case guarantees; Gmean tolerance does not improve; test Gmean supplies counterevidence; A inherits potentially different EWC Fisher matrices and a lower GAM-anchor a22; C historical sequences use different future optimizers and show larger GAM endpoint displacement. No raw report, historical output, or GPU job was altered.
