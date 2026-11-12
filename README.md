# Outbound Vision API

A backend-only FastAPI service. A client (a person with a 12 MP webcam today, an ESP32 trigger later) sends one image to `/api/v1/scan` and gets back every barcode and QR code in it. Each request also feeds a background collector that builds a Gemma-verified YOLO dataset, which will later train a lightweight detector.

```
image ─► decode + EXIF upright ─► Grounding DINO ("barcode") ─► containment suppression ─► NMS ─► padding
                                                                                                   │
              ┌───────────────────────────────────────────────────────────────────────────────────┤
              ▼                                                                                   ▼
   parallel code reader (pyzbar → zxing-cpp)  ─► [OCR] ─► [VLM agent] ─► JSON        dataset queue (put_nowait)
                                                                                                   │
                                                              background process: Gemma true/false per crop ─► ./dataset
```

## Requirements
- Linux x86_64, an NVIDIA GPU (RTX 50xx / sm_120 supported via CUDA 12.8 PyTorch wheels), NVIDIA driver ≥ 570.
- Docker with Compose v2 and the `nvidia-container-toolkit`.
- Optional: a Gemma server with an OpenAI-compatible API for dataset verification. Examples are llama.cpp `llama-server` with `--mmproj`, or vLLM with `--limit-mm-per-prompt image=8`. It must listen on `0.0.0.0` so the container can reach it through `host.docker.internal`.

## Quick start
```bash
git clone <repo> && cd <repo>
make up                               # creates .env from .env.example and ./dataset, then docker compose up -d
make health                           # wait for "ready": true
make scan IMG=path/to/image.jpg       # barcodes + latency breakdown
make logs
```
`make build` rebuilds the image. The Grounding DINO weights are downloaded during the build, so the container never fetches them at runtime.

## API

### `GET|POST /api/v1/scan`
Both methods go to the same handler. The image can be sent in any of these forms:

| Content-Type | Body |
|---|---|
| `multipart/form-data` | field `file` (or `image`), plus optional override fields |
| `application/json` | `{"image_base64": "...", ...overrides}`. A `data:` URL prefix is fine. |
| `image/jpeg`, `image/png`, `image/webp`, `application/octet-stream` | raw image bytes; overrides go in the query string |
| *(no body)* | `?image_base64=...`, **for small images only**: URL length limits rule out a 12 MP frame |

**Optional overrides** (body > query string > `.env`):

| Field | Effect |
|---|---|
| `prompt` | Detection prompt. Normalised to Grounding DINO form (`"Barcode"` → `"barcode."`). |
| `box_threshold`, `text_threshold` | Grounding DINO thresholds. |
| `padding` | Sets both `pad_x` and `pad_y`. |
| `pad_x`, `pad_y` | Padding per axis, as a fraction of the box width/height. |
| `enrich_prompt` | Runs a second detection pass (e.g. `"shipping label"`). OCR and the agent then read those crops instead of the barcode crops. Only used when `ENABLE_OCR` or `ENABLE_AGENT` is on. |

```bash
curl -F file=@box.jpg http://localhost:8000/api/v1/scan
curl -X GET -F file=@box.jpg http://localhost:8000/api/v1/scan
curl -H 'Content-Type: image/jpeg' --data-binary @box.jpg 'http://localhost:8000/api/v1/scan?padding=0.3'
curl -F file=@box.jpg -F enrich_prompt="shipping label" http://localhost:8000/api/v1/scan
```

**Response**
```json
{
  "request_id": "uuid (matches ./dataset/*/<request_id>.*)",
  "image_width": 4000, "image_height": 3000,
  "barcodes": [{"symbology": "CODE128", "value": "...", "bbox_global": {"x1":0,"y1":0,"x2":0,"y2":0},
                "segment_index": 0, "engine": "pyzbar"}],
  "segments": [{"segment_index": 0, "source": "barcode", "score": 0.77, "label": "barcode",
                "raw_bbox": {...}, "padded_bbox": {...}, "barcodes": [...], "decode_error": null,
                "ocr_result": null, "agent_result": null}],
  "timing_ms": {"decode_image_ms": 0, "segment_ms": 0, "code_reader_ms": 0, "enrich_ms": 0, "total_ms": 0},
  "dataset_queued": true
}
```
`barcodes` is the flat list across all segments, deduplicated by `(symbology, value)`. Each segment's own `barcodes` list keeps every hit.

**Errors**

| Status | Cause |
|---|---|
| 400 | Undecodable image or bad base64 |
| 413 | Image larger than `MAX_IMAGE_BYTES` |
| 415 | Unsupported content type |
| 422 | No image, or invalid overrides |
| 503 | Model not ready, CUDA out of memory, or the remote segmentor is down |
| 500 | Anything else (the response shows only the error type) |

### `GET /api/v1/health`
Returns readiness, device, segmentor mode, feature flags, and whether the dataset worker is alive. Interactive docs are at `/api/v1/docs`. Swagger UI cannot send a GET request with a body, so use POST there.

## Pipeline details
- **Segmentation.** Uses the Hugging Face `IDEA-Research/grounding-dino-tiny` model in-process on the GPU. Inference runs on a single-thread executor, so requests take turns on the GPU and the event loop never blocks. The frame is downscaled to `GDINO_MAX_SIDE` for inference, and boxes are mapped back to full resolution.
- **Outer-box suppression.** For box pairs with Area(Bᵢ) > Area(Bⱼ), if `Area(Bᵢ∩Bⱼ)/Area(Bⱼ) ≥ CONTAINMENT_THRESHOLD`, the outer box Bᵢ is dropped. Class-agnostic NMS (`NMS_IOU_THRESHOLD`) runs afterwards.
- **Padding.** Each kept box is padded by `SEGMENT_PAD_X`/`SEGMENT_PAD_Y` of its own width/height and clamped to the frame. This restores the quiet zones; there is no synthetic upscaling.
- **Decoding.** All crops decode in parallel on a persistent `ThreadPoolExecutor(SCANNER_THREADS)`. For each crop, pyzbar runs first; zxing-cpp runs only if pyzbar found nothing (DataMatrix, PDF417, dense codes). Coordinates are mapped back to the full frame.
- **Remote segmentor.** With `USE_SEGMENTOR_API=true` and `SEGMENTOR_API_URL=<.../api/v1/detect>`, segmentation goes to the hosted Grounding DINO API (the legacy notebook contract: detections in the `X-Detections` header) and no model is loaded locally. Images that have an EXIF orientation are re-encoded upright before sending, so the boxes line up with the frame.

## Dataset collector (`ENABLE_DATA_COLLECTION=true`)
A separate OS process, started with `spawn` so it never touches CUDA, takes jobs from a `multiprocessing.Queue`.
- The API enqueues the **original encoded upload bytes** (a few MB) with `put_nowait`. If the queue is full, the sample is dropped with a warning, and the HTTP response is never delayed.
- The worker decodes the bytes the same way the API does, cuts the padded crops again, and asks Gemma:
  - `GEMMA_BATCH_MODE=multi_image`: up to `GEMMA_MAX_BATCH_SIZE` crops per request, answered as `image i: true|false` lines (`temperature=0`, `max_tokens=8·K`).
  - `GEMMA_BATCH_MODE=single_token_concurrent`: one request per crop, all fired concurrently, each answered `true`/`false` (`max_tokens=2`).
- On a Gemma outage (connection error, timeout or 5xx), the request is retried, then a circuit breaker skips Gemma for `GEMMA_CIRCUIT_COOLDOWN_S`. Samples are still saved, as `unverified`. With `GEMMA_API_BASE_URL` empty, every sample is stored unverified.

```
dataset/
  images/{request_id}.jpg          full upright frame
  labels/{request_id}.txt          YOLO "0 cx cy w h" (raw box) for every POSITIVE segment; empty = background
  crops/positive|negative|unverified/{request_id}_seg{i}.jpg
  metadata/{request_id}.json       boxes, sizes, GDINO score, scanner_decoded, payloads, gemma_verified,
                                   final_label, label_complete, provenance (written last)
  classes.txt, data.yaml
```

**Label rule:**
- `positive` when `gemma_verified == True` **or** `scanner_decoded`.
- `negative` when Gemma said false and the scanner decoded nothing.
- `unverified` when Gemma gave no usable answer.

Images with any `unverified` segment have `label_complete: false`; filter those out (or review them) before training. On shutdown the worker drains its queue. If the drain runs past half of `DATASET_SHUTDOWN_TIMEOUT_S`, the remaining jobs are saved without asking Gemma.

## Enrichment flags
- `ENABLE_OCR=true`: OCR text for each crop (`ocr_result`). The OCR engine is not in the image by default. Build it in with `UV_EXTRA_ARGS="--extra ocr" make build`, or locally run `uv sync --extra ocr`. Nothing is loaded when the flag is off.
- `ENABLE_AGENT=true`: structured Gemma sticker reading for each crop (`agent_result`). It keeps the legacy extraction prompt, the closed sticker-type vocabulary, the confidence bands, and the mechanical `needs_verification` rule (confidence < 0.70 or type `unknown`). Configure it with `AGENT_API_*`, which falls back to `GEMMA_*` when empty.

All settings are listed with their defaults in [`.env.example`](.env.example).

## Local development (macOS / no GPU)
```bash
brew install zbar        # pyzbar's C library; the Makefile sets DYLD_FALLBACK_LIBRARY_PATH
make sync                # uv sync (CPU/MPS torch from PyPI on macOS; cu128 wheels on Linux)
make test                # fast suite: no GPU, no weights, no network
make test-slow           # also spawns the real dataset worker process
make lint
GDINO_DEVICE=cpu uv run uvicorn src.main:app --port 8000
```

## uv-only policy
Dependencies live in `pyproject.toml` and `uv.lock`, and nothing else. Add a dependency with `uv add <pkg>`, then commit `uv.lock`. `make lock-check` checks that the lockfile is up to date, and `make check-no-pip` checks that no `requirements.txt` or other installer invocations exist outside `legacy/`.

## Operations
- Run **one** uvicorn worker. Each worker would load its own model and start its own collector.
- Files in `./dataset` are written by root inside the container.
- There is no retention policy yet: watch disk usage, since a week of 12 MP frames adds up.
- `legacy/` holds the original agent-framework code as a reference only. It is not used, built, or linted.
