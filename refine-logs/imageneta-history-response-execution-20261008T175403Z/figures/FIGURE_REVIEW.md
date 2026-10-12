# Codex STRICT visual review — History-response flow

## What I see

The rendered PNG shows a top-to-bottom training flow. The first gradient and virtual SGD prediction lead to a resolvability decision. A zero/unresolvable probe goes directly to U=g. A resolvable probe reads the future gradient on the same realized batch and R0, restores the full base state, and selects either the no-history mirror or a raw-response fusion. All three branches converge on one plain SGD commit. The raw base gradient and measured committed displacement are cached afterward. A left-hand loop returns to the next batch; the dotted right-hand branch observes post-update loss and restores the committed state/R1.

The labels are English, equations are multiline KaTeX, and the Markdown caption is Chinese. The main figure contains no QR, Hessian inverse, extra relocation read, gradient decomposition, or hidden optimization problem.

## Files generated

- history-response-flow.mmd — raw Mermaid source
- history-response-flow.md — identical source plus Chinese caption
- history-response-flow.png — inspected raster rendering
- history-response-flow.svg — scalable vector rendering
- history-response-flow.pdf — single-chart PDF
- flow-theme.css — explicit sans-serif labels and restrained styles
- puppeteer-config.json — isolated browser launch settings
- render-history-response-flow.sh — reproducible rendering command

## Strict verification checklist

### A. File correctness

- [x] Mermaid CLI 12.0.0 accepted the final source and rendered PNG/SVG/PDF with exit code 0.
- [x] The Markdown contains a Mermaid fence.
- [x] Raw source and fenced Markdown source are byte-identical.
- [x] SVG XML, PNG decoding, and PDF signature checks pass.

### B. Arrow correctness

- [x] Task/parameter changes clear history before the first current-batch gradient.
- [x] The future-resolvable Yes branch performs the second gradient; No uses U=g.
- [x] Valid-history Yes fuses responses; No selects 2g−g_future.
- [x] All update branches join exactly one commit.
- [x] Cache writes follow a successful commit and contain raw g plus actual displacement.
- [x] The same-task next-batch loop returns to a new current-batch evaluation.
- [x] The diagnostic arrow is dotted and has no feedback into the update formula.

### C. Block content

- [x] The SGD prediction is theta_f=theta−eta*g; ell_f=eta*||g||.
- [x] Both response signs and denominators match the reviewed first-version algorithm.
- [x] The historical denominator uses ||s||, not the nominal prediction distance.
- [x] The final update is g+ell_f*(h_future+h_history)/2 with w=1/2.
- [x] Probe restoration, same-batch/R0 replay, one commit, cache update, and passive diagnostics are explicit.

### D. Completeness

- [x] Current gradient, virtual future gradient, both decision branches, commit, cache, task reset, and side diagnostics are present.
- [x] No unrequested three-gradient extension is included.
- [x] The caption records the cross-minibatch historical-response limitation and unchanged epoch scheduler cadence.

### E. Visual quality

- [x] Thin borders, white background, restrained green/blue/purple/orange fills.
- [x] Sans-serif labels and mathematical notation are legible; no missing glyphs remain.
- [x] Arrowheads are visible; no crossing arrows or clipped labels.
- [x] Spacing clearly separates probe, branch, update, and cache stages.
- [x] The main training path and passive diagnostic branch can be distinguished immediately.

## Issues found and fixes

1. The first renderer used native MathML without a complete math font, producing missing glyphs. Forced legacy KaTeX rendering resolved this.
2. Mixed HTML/math labels collapsed their line breaks in the installed renderer. Replaced them with explicit multiline gathered math labels and correctly escaped Mermaid/KaTeX row breaks.
3. The default label font rendered as serif. Added a DejaVu Sans CSS override for ordinary labels and math text; the mathematical symbols retain KaTeX fonts.

## Score: 9/10

All semantic and arrow checks pass. The remaining polish is the tall portrait layout, which is appropriate for a full algorithm flow but may need cropping or a more compact redraw for a two-column paper.

## Verdict

- [x] ACCEPT
- [ ] FIX
