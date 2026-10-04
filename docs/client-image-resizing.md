# Client image resizing — October 3, 2026

The original menu, PNG encoding, padding, and `do_resize: false` GUI path below
were superseded on October 4 by [Image Scaling](image-scaling.md). The current
GUI preserves source format and enables native patch alignment. This document
records the initial API compatibility change and validation; direct clients
can still use the options shown here.

The GUI now resizes images before upload. This interface change was deployed on
the test services: Full on an RTX 3090, and Flash on an RTX 3070 Ti and an
RTX 2080 Ti. Models, precision, single-GPU placement and context
admission budgets are unchanged. This deployment preserves Flash's existing
long-request cache policy; the separate long-cache rollout remains Full-only.

## Browser behavior

Standard remains the browser default: a 256² equivalent pixel budget. Low 128²,
medium 512² and high 1024² remain available. The browser keeps the original image
in this tab; each variant is resampled from that original, not a lower-resolution
variant. Prepared data is reused for repeat calls at the same fidelity. Switching
fidelity prepares a fresh variant and replaces the previous prepared variant.
Upload, paste and drag/drop use the same preparation path.

Dimensions align to 32 pixels (16-pixel vision patches / 2×2 merge) and fit the
selected area budget. Aspect ratio is approximated within that alignment; the
image is stretched to the aligned dimensions without cropping. The minimum
side is 32 pixels; extremely narrow images may have larger aspect distortion to
fit the budget. Ratios above 200:1 are rejected. Browser image decoding applies
orientation; transparent pixels are composited on white. Prepared uploads are
lossless PNG to avoid a second lossy JPEG encoding. PNG byte sizes depend on
image content: high fidelity is not guaranteed to be smaller than a compressed
original. Original uploads still have a 10 MiB / 20-million-pixel selection limit.

The client sends resized images and `media_kwargs.images_kwargs.do_resize: false`.
The server validates and decodes each upload, normalizes pixels and builds vision
patches/features, but does not geometrically resize these GUI inputs. Preprocessing,
feature and language caches remain available; prepared image bytes and processor
options participate in their identities. Cache entries are rebuilt after restart
or when the new resized upload differs from the previous original-file upload.

Total response time now includes client preparation, JSON serialization, upload
and receiving/parsing the answer. The GUI reports client image preparation and
encoded request size in the timing note. Server processing is still handler time,
including CPU preprocessing and GPU computation but excluding upload and initial
HTTP JSON parsing/validation. First preparation takes time; repeats reuse it.

## API compatibility

`media_kwargs` is accepted and forwarded to Clef's upstream processor. API callers
that omit it and `image_fidelity` use the checkpoint's native processor defaults;
there is no automatic 256² server override. Native defaults may process more
visual tokens than the browser default. Combined context admission and HTTP 413
remain in force; clients should query `/v1/models` for their token budget.
Normalization and patch extraction still run with `do_resize: false`; callers
using that option must supply appropriately aligned images.

For example, with a prepared PNG data URL:

```json
{
  "model": "clef",
  "state": "Inspect the supplied image.",
  "images": ["data:image/png;base64,..."],
  "media_kwargs": {"images_kwargs": {"do_resize": false}},
  "questions": {"ship": {"type": "noul", "instructions": "Is a ship visible?"}}
}
```

API callers can instead request native processor resizing with
`media_kwargs.images_kwargs.min_pixels` / `max_pixels`. The old `image_fidelity`
field remains an explicitly selected, deprecated extension for existing clients;
it no longer has a default. Supplying both `media_kwargs` and legacy fidelity
returns 400 instead of silently overriding either. `CLEF_MAX_IMAGE_PIXELS` now
controls only this legacy fidelity path. Existing caching and pooling extensions
remain supported. This change improves image-option compatibility; our HTTP
wrapper still has its own input validation and service extensions.

Usage reports actual `processed_images` dimensions and `processor_resize_enabled`.
The former refers to pre-pooling processed pixels, so optional pooling does not
misrepresent the dimensions. `image_fidelity` is null for native/client requests.
Health identifies browser resizing and exposes the checkpoint's default pixel
budget alongside the legacy fidelity limit.

## Verification

Browser tests used the supplied beach/ship photo at all four fidelities, repeated
standard, switched from low to high and back to standard, and pasted a synthetic
PNG alongside the photo. Received dimensions were 96×128, 192×288, 416×576 and
864×1152, all within budget and aligned. Originals remained available for changes.
The server-side transport fixture recorded only metadata in the report; the
private prepared images remain outside Git. Live browser validation on Full
returned a scene/object answer with a 183 KiB upload, 0.051 s initial preparation
and 0.430 s total. Its repeat showed 0.000 s preparation, 0.332 s server processing
and 0.364 s total. These are individual observations with pre-existing GPU cache
hits, not a controlled before/after latency benchmark or eleven-photo timing.

All three live services passed ten successful inference cases plus conflicting
option rejection (400) and over-limit rejection (413). Cases covered all four
client dimensions, repeat/cache equality, optional pooling, legacy/native budget
agreement, omitted processor options and a text regression. The known-answer
checks concerned a cruise ship and clothing color; no face identity task was used.
The legacy explicit budget and the equivalent native options produced identical
recorded answers and probabilities. Client canvas and server Pillow resizing
use different resampling implementations; they are not pixel-identical.

| Service | Standard client request | Cached repeat | Maximum client/legacy probability difference |
|---|---:|---:|---:|
| Full / RTX 3090 | 0.837 s | 0.250 s | 0.58 pp |
| Flash / RTX 3070 Ti | 1.108 s | 0.271 s | 0.29 pp |
| Flash / 2080 Ti | 0.926 s | 0.238 s | 0.30 pp |

The Full API fixture body fell from 3,001,541 to 188,483 bytes (**93.7%
smaller**) at standard fidelity, using the same original photo and questions.
This measures upload payload reduction, not a claimed inference speedup. The
fixture contains one photo and two questions; these checks do not establish
accuracy equivalence across a broader dataset. Selected known answers agreed.

CPU checks passed both real request schemas, default/explicit processor options,
legacy compatibility and conflict/limit handling. JavaScript syntax, project
provenance, UI checks and links passed. All fifteen deployed file hashes matched
install manifests. NVIDIA process mapping confirmed one GPU per service. Per-file backups were
preserved before installation.

Run `scripts/test_image_request.py` for CPU request checks. Live verification is
`scripts/test_client_image_api.py`, using browser-prepared private PNG fixtures
and the original photo via its documented arguments. Original photos/base64
payloads are not included in Git or saved measurement JSON. Raw metadata and
measurements are kept outside this repository.
