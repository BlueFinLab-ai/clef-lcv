# Image Scaling — October 4, 2026

The GUI's former Image fidelity control is now **Image Scaling**. It changes
browser-side preparation only; the Clef API request format is unchanged.

| Preset | Maximum landscape frame |
| --- | --- |
| Original | Source resolution |
| 4K UHD | 3840 × 2160 |
| QHD | 2560 × 1440 |
| Full HD | 1920 × 1080 |
| HD | 1280 × 720 |
| 480p | 640 × 480 |
| MCGA | 320 × 200 |
| Compact (default) | 256 × 256 |

Images fit inside the frame with their aspect ratio preserved. Landscape frames
rotate for portrait sources. Smaller sources are never enlarged. Every variant
is prepared from the decoded original retained in the tab, so switching from
Compact to Original restores source resolution.

JPEG stays JPEG, PNG stays PNG, and WebP stays WebP. When resizing is needed,
the browser encodes once in the source format (JPEG/WebP quality 0.95; PNG is
lossless). Compression settings and metadata are not preserved on resized
images. Original and sources already within the selected frame use the exact
source file bytes; there is no encoding, padding, cropping or format fallback.
PNG and WebP transparency survive browser scaling.

The model still requires 32-pixel patch alignment. The GUI sends native
`media_kwargs.images_kwargs` with `do_resize: true`, `min_pixels: 1024`, and
`max_pixels: 20000000`. This lets the processor round dimensions to patch
boundaries without imposing the checkpoint's usual min/max pixel budget.
The selected display frame is applied entirely in the browser; processed
dimensions can differ slightly for alignment, and very small sources may be
enlarged to a minimum patch. For example, the browser's portrait HD output is
720 × 960; the model processes 704 × 960. Original's 3024 × 4032 source is
uploaded unchanged and processed at 3008 × 4032.

If a browser cannot encode the original format, the UI reports an error instead
of silently converting it. If the resized output exceeds 10 MiB, it asks for
a smaller scaling option. Source limits remain 10 MiB, 20 million pixels and
16 images. GPU memory and the context admission limit still apply.

Direct API clients retain native processor defaults and explicit processor
options; legacy `image_fidelity` remains optional and deprecated. No backend
code or API schema changed for this format-preservation update.

## Current validation: source-format preservation

Browser tests covered JPEG at all eight presets, scaled PNG and WebP,
Original for all three formats, and an already-small JPEG. The four unchanged
source uploads were verified byte-for-byte. PNG/WebP transparency was checked
after scaling. Repeated preparation reused identical upload bytes. Geometry
checks cover 56 source/preset combinations. Additional checks reject encoder
format substitution and oversized outputs without format fallback.

All three live services passed JPEG, PNG and WebP at Compact, with the expected
processed dimensions and finite scene-question answers above 0.5. Full also
passed HD, 4K and the unchanged Original JPEG. Original's processor output was
3008 × 4032, confirming native alignment rather than browser padding. Larger
Flash presets were not inference-tested in this update.

Only static web files were deployed to Full and both Flash services, with
per-file rollback copies. No runtime code, API schema or service restart changed.
Served file hashes matched the saved source at all three addresses. Current
raw metadata and API responses are kept outside this repository; source fixtures are not bundled.

The Compact baseline JPEG payload was about 36 KiB, versus 166 KiB when the
previous implementation converted it to PNG. This comparison concerns upload
size, not GPU inference speed or controlled accuracy.

## Previous validation: initial presets

- CPU geometry checks passed 56 source/preset combinations, covering portrait,
  landscape, exact-size and small sources, no enlargement, patch bounds,
  Original resolution, and invalid dimensions/aspect ratios.
- The actual browser uploaded the same private baseline photo through all eight
  choices. Captured dimensions and encoding matched the helper; a repeated
  Original request reused prepared data with 0.000 s displayed preparation.
- The live Full RTX 3090 service accepted Compact, HD, 4K and Original. Server-reported dimensions
  exactly matched uploaded dimensions and processor resizing was disabled.
  The scene question returned a probability above 0.5 in all four cases.
- Compact passed on Flash 3070 Ti and Flash 2080 Ti with the same dimensional
  and processor checks. Larger Flash presets were not inference-tested here.
- Only three static web files were deployed per service, with rollback copies;
  no model runtime changes or service restarts were needed. Deployed web hashes
  matched the saved source on all three services.

For the 3024 × 4032 baseline source, recorded first-call Full server times were
0.674 s at Compact (192 × 256), 1.150 s at HD (704 × 960), 8.261 s at 4K
(2144 × 2880), and 21.884 s at Original (3040 × 4032 including padding).
These are scene-question smoke checks on a running service, not controlled
performance/quality comparisons. They do not validate identity recognition.

Raw metadata and API responses are kept outside this repository; fixtures remain private and are not bundled.
Run current geometry/format checks with `node scripts/test_image_scaling.js`.
The initial padding and PNG-fallback behavior above is historical.
