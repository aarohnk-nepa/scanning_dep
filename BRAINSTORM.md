# Outbound Vision Pipeline — Design Brainstorm

Research-only document per `BRAINSTORM_PROMPT.md`. No source files were modified. Sources are cited inline with URL + approximate access/publish date where found; anything not independently verifiable is marked **[UNVERIFIED]**. Research was done in four passes: (1) a full read of the existing `agent.py` / `segmentor.py` / `scanner.py` / `code_reader.py` / `schemas.py` / `segmentation_api.py` / Grounding DINO notebook, and three independent web-research passes covering (2) language/detector/barcode/OCR/result-structure, (3) camera/trigger/lighting hardware, and (4) Blackwell/Docker/deployment/latency.

---

## 1. Executive recommendation

| Area | Recommendation |
|---|---|
| Language | **Python**, end to end. Nothing in this pipeline is Python-glue-bound; every needed library's best binding is Python; it's also what the implementing LLM agent will write most reliably. |
| Detector | **Grounding DINO offline-only**, as an auto-labeling tool. Train and ship a small **Apache-2.0 single-class detector** (YOLOX-s or RT-DETR/D-FINE — explicitly *not* Ultralytics YOLO, which is AGPL-3.0) exported to TensorRT/ONNX. |
| OCR | **PaddleOCR (PP-OCRv4/v5)** as primary — PaddlePaddle 3.2.1+ has an official Blackwell build verified specifically on the RTX 5070. **RapidOCR (ONNX Runtime)** and **docTR (PyTorch)** as fallbacks if that claim doesn't hold up on the real box. |
| Barcode/QR | Keep **pyzbar-first, zxing-cpp-fallback**, keep the 2x-upscale-below-1000px retry. This is already the right design; don't pay for a commercial SDK. |
| Camera/trigger | **ESP32 + photoelectric break-beam sensor + USB3 Vision camera** (Basler ace2 or Daheng, 5–12MP) + diffuse polarized LED lighting in a light-blocking hood. Software trigger over USB-CDC is fine — settle time dominates, not trigger jitter. **Reject the iPhone route and off-the-shelf smart cameras.** |
| Deployment shape | One Dockerized **modular monolith** process, CUDA 12.8+/driver R570+ base (NGC or vendor-provided `-sm120` image), weights baked into the image, file/DB-drop output, `restart: unless-stopped` + healthcheck, Makefile-wrapped deploy/rollback. |
| Gemma flag | Boolean env var, default **false**. When true: called **only** when local OCR/barcode confidence is low or validation fails (never on every sticker), receives the **crop only**, hard timeout, never blocks the local result, credentials via mounted `.env`, never baked into the image. |

I disagree with a few of your instincts, specifically: Grounding DINO's own license is fine (Apache-2.0) — the real licensing landmine is Ultralytics YOLO (AGPL-3.0), which is the thing you'd naturally reach for to replace it. PaddleOCR is more viable on this exact GPU than its reputation suggests, given a direct RTX-5070-specific vendor claim (see §4.D). And the iPhone option, which reads as operationally "free" because you may already own compatible phones, in fact carries the only *recurring, indefinite* cost and a currently-documented reliability defect (thermal throttling) in this whole comparison — the camera/ESP32 route is cheaper over any multi-year horizon, not just more "industrial."

---

## 2. What exists today — read in full, kept as record not structure

Every file in the directory was read in full (`agent.py`, `segmentor.py`, `scanner.py`, `code_reader.py`, `schemas.py`, `segmentation_api.py`, `hosting-grounding-dino (1).ipynb`). No README, Dockerfile, requirements file, or `.env.example` exists anywhere in the directory. Summary of what's there and what to do with it:

**Orchestration (`agent.py`, `VisionAgent`)** runs on an external agent framework not present in this directory (`agent_runtime.base_agent`, `agent_platform.orchestrator.*`, `agent_platform.gemma_gateway_service`, `tools.base_tool`, `connectors.vision.*`). It dispatches to `vision.segmentor` and `vision.scanner` via a **string-keyed tool registry** (`context.call_tool(tool_name=..., payload=...)`), each wrapped in its own `BaseTool` with pydantic input/output schemas, a per-tool `timeout_s`, and a tenant/connector-factory layer that is a no-op here (every tenant gets the same connector) but is still real plumbing. An `AgentEvent`/`EventType` event bus streams `THOUGHT_SUMMARY` progress messages. **All of this is removable**: collapse `run(context)` into a plain function `process_box(image, ...)`, replace `call_tool` with direct function calls passing in-memory arrays, drop the event bus (no consumer needs it in one process), drop `BaseTool`/connector-factory/tenant layers, and stop base64-round-tripping between every stage.

**Worth keeping as-is or near-as-is:**
- Crop padding: `pad_x = (x2-x1)*padding`, `pad_y = (y2-y1)*padding`, default `padding=0.08`, floor/ceil, clamped to image bounds, min crop side 4px, long-side cap 2048px. Justified as "keeps QR quiet zones and edge-clipped text in frame" — keep as default.
- Barcode engine order and retry, exactly as implemented: pyzbar always first (one pass finds everything, including all 1D formats); zxing-cpp runs *only* if pyzbar found literally nothing, never merged; retry triggers only when nothing was found **and** the crop's longer side is under 1000px, upscaling 2x with `cv2.INTER_CUBIC` and dividing reported coordinates back down.
- Per-crop error isolation everywhere (one bad crop/detection never sinks the batch) — this pattern should be preserved in the new design for detection, decode, and OCR stages alike.
- EXIF/coordinate-frame handling in `segmentor.py`: match the frame first (re-decode with `IMREAD_IGNORE_ORIENTATION` if dimensions don't match the API's annotated image), then scale whatever's left — a two-stage fix worth keeping conceptually even once the detector is in-process (orientation metadata still needs handling regardless of who runs inference).
- The mechanical `needs_verification` rule in `agent.py`: forced true if `confidence<0.70` OR `type=="unknown"` OR the model itself said so — "a mechanical rule must not be delegated to a stochastic process." Keep this pattern; extend its triggers (see §4.E).
- The closed sticker-type vocabulary: `{shipping_label, product_label, handling, warning, other, unknown}`.
- The Gemma call *shape* (not the implementation, which isn't in this repo subset): OpenAI-style chat messages, one crop per call for failure isolation, `temperature=0`, `max_tokens=1024`, JSON response with `raw_text` required **first** (left-to-right generation — transcribe before committing to a verdict), confidence bands (0.90–1.0 sharp, 0.70–0.89 minor issues, 0.40–0.69 several uncertain, <0.40 inferring), explicit prohibition on "completing" partial digits, explicit 0/O, 1/l/I, 5/S handling.

**Pure overhead to delete:** the HTTP hop to a separate Grounding DINO microservice (`segmentation_api.py`'s `httpx` POST + `X-Detections` header parsing) once a detector is loaded in-process; all base64 encode/decode round-trips between agent/segmentor/scanner; the tenant/connector-factory layers; the event bus.

**One real discrepancy to resolve, not inherit silently:** `agent.py`'s own default `text_prompt="sticker"` does not match the notebook demo's prompt `"carton . shipping label . barcode . QR code ."`. These are different prompts in practice for an open-vocabulary detector — pick one deliberately if Grounding DINO is used at all (even just for auto-labeling, per §4.B).

**Gemma gateway gap:** the actual `agent_platform.gemma_gateway_service` client (host/port, API key, model name, connection pooling, retry/backoff) is **not in this repo subset** — only its external call surface is visible from `agent.py`. The new optional cloud-Gemma client has to be built from the request/response *shape* alone; there's no implementation to lift. One comment in `agent.py` implies the current backend is llama.cpp/llama-server (`"llama-server needs --mmproj for this"`) — i.e. today's Gemma is local, not cloud; moving it to a cloud-hosted endpoint is a protocol change, not just a config change.

The only environment variable found anywhere in these files is `VISION_SEGMENTATION_URL`.

---

## 3. Hard design constraints (restated, not re-litigated)

No agent framework, no tool registry, no event bus, no tenant/auth layers, no HTTP hop between internal stages, no base64 round-trips between stages — a modular monolith, one process, plain function calls, pixels stay as in-memory arrays end to end. Models load once at startup, stay warm on GPU; cold start measured separately from steady-state per-box latency. Default path fully offline; Docker with GPU access is mandatory. Pipeline: capture → detect → crop → barcode/OCR → structured result. Gemma stays, reduced to an optional, cloud-hosted, default-false, non-blocking step.

---

## 4. Sections A–H

### A. Language and runtime

**Where time actually goes.** For capture → GPU detector inference → crop → barcode decode (CPU) → OCR (GPU) → regex extraction → JSON, the two GPU steps run as native CUDA kernels under PyTorch/ONNX Runtime/TensorRT regardless of host language — a single forward pass on a stationary crop will run anywhere from ~5ms (small detector) to ~150ms+ (Grounding DINO at full res), one to three orders of magnitude above plausible Python interpreter overhead. Barcode decode is native C/C++ (ZBar, zxing-cpp) called through thin bindings — sub-ms-to-few-ms work either way. Regex extraction over a few hundred characters is microseconds in any language. **No step in this pipeline is plausibly Python-glue-bound.**

**Binding maturity** — every library this system needs (PyTorch, ONNX Runtime, TensorRT, pyzbar, zxing-cpp, PaddleOCR/RapidOCR/docTR) has Python as its only first-class, actively-maintained binding. Rust alternatives exist for some (`tch-rs`/`candle`, the `ort` crate) but are secondary and historically lag upstream feature/EP support — the `ort` crate's TensorRT execution-provider path has required building ONNX Runtime from source, described in its own docs as slow with real packaging friction **[per docs.rs/ort, 2025–2026 snapshots — unverified exact wording]**. C++ is native for TensorRT/ZBar/zxing-cpp but is the weak link for the ML-framework side — LibTorch's C++ API is a distant second to PyTorch's Python API in documentation and troubleshooting precedent.

**LLM-authored-code reliability.** General (not pipeline-specific) benchmarks are directional, not decisive: one benchmark (PROBE, arXiv 2607.13820) found Rust the highest non-compiling/failure-rate language tested, Python the highest pass rate; a separate 100-problem comparison found Python-vs-Rust "not statistically significant." Treat this as weak corroboration, not the main argument — the stronger, concrete argument is that every library in this pipeline has its best-documented, most-written-about interface in Python, so an LLM coding agent is far more likely to produce correct glue code against GroundingDINO/ONNX Runtime/TensorRT/pyzbar/zxing-cpp/PaddleOCR in Python than in Rust, where it would also need to get `unsafe` FFI plumbing right with far less precedent.

**SSH/headless debugging.** Python has the richest ecosystem here by a wide margin: `pdb`/`ipdb`, `python -i` for post-mortem REPL inspection, rich tracebacks, `nvidia-smi`/`py-spy`/`torch.cuda.memory_summary()` for live introspection, no compile step for print-debugging. Rust/C++ require a full recompile per debug iteration and push you toward `gdb`/`cuda-gdb` for tensor-shape/device-placement bugs — a much higher expertise floor than an LLM agent reliably clears unsupervised.

**Recommendation: Python** for the whole monolith, calling into native GPU/CPU libraries exactly as today. No credible case for Rust/C++ exists under the stated constraints (single process, plain calls, in-memory arrays) — heavy work runs in native code under any choice, and Python minimizes both binding risk and LLM-authored-bug risk.

**What would change this:** on-box profiling showing >20–30% of per-box wall-clock *outside* model `.forward()`/decode calls. That's a code-quality bug (un-vectorized loop, an extra `.cpu()`/`.numpy()` round-trip, an accidental CUDA sync point) to fix in Python, not evidence to switch languages.

**Main risk:** none specific to this choice; the risk sits entirely in correct GPU-memory/array handling, which is a design discipline, not a language property.

---

### B. Detector

| Option | Latency/VRAM (5070) | License | Verdict |
|---|---|---|---|
| Keep Grounding DINO | Not directly benchmarked on a 5070 anywhere found; closest data points ~52–165ms on various datacenter/edge GPUs **[extrapolated to ~80–200ms on a 5070, UNVERIFIED]** ([NVIDIA VSS perf docs](https://docs.nvidia.com/vss/3.3.0/appendix/performance/microservices/rt-cv-performance.md); [Roboflow hosted-model docs](https://docs.roboflow.com/models/supported-models/grounding-dino)) | **Apache-2.0**, confirmed by reading the LICENSE file directly ([github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE)) | Works, but heavy and open-vocabulary-ambiguous for a now-known single class |
| Auto-label with Grounding DINO → train small detector (YOLOX-s / RT-DETR / D-FINE / RF-DETR) → TensorRT/ONNX | Single-digit ms expected on a 5070 for a small detector (see §4.H) | All **Apache-2.0** — YOLOX confirmed via repo LICENSE; RT-DETR/D-FINE/RF-DETR per HF model cards and Roboflow pages **[secondary sources, primary LICENSE not directly re-confirmed for RT-DETR/PaddleDetection repo — flag before committing]** | **Recommended** |
| Classical CV (contours/color/template) | Fast, free | N/A | Too fragile to real sticker-placement/lighting variance; usable only as a refinement step on top of a learned detector's box, not as sole detector |

**Licensing is the decisive factor you didn't ask about but should weigh heavily:** Ultralytics YOLOv8–v12 is **AGPL-3.0**, and Ultralytics' own statement is that trained weights inherit AGPL-3.0 too — a paid Enterprise License is required to avoid open-sourcing the whole network-accessible application ([Roboflow model-licenses page](https://roboflow.com/model-licenses/yolo11); [ultralytics.com AGPL page](https://www.ultralytics.com/legal/agpl-3-0-software-license)). That's real exposure for a commercial warehouse product. YOLO-NAS's pretrained weights carry a separate non-commercial restriction **[secondary source, unverified against a primary license file]**. Grounding DINO itself, by contrast, is cleanly Apache-2.0 — worth stating explicitly since "research code" licenses are often assumed to be restrictive and this one isn't.

**Recommendation: option 2.** Use Grounding DINO offline, only as an auto-labeling tool, to train a small Apache-2.0 single-class detector exported to TensorRT/ONNX for production. This sidesteps the open-vocabulary prompt-ambiguity problem entirely (a trained single-class model has no prompt to misconfigure, resolving the `"sticker"` vs. `"carton . shipping label . barcode . QR code ."` discrepancy by making it moot) and should be meaningfully faster than Grounding DINO's full pipeline (including its `bert-base-uncased` text encoder, which a trained detector has no need for).

**Data-collection and labeling plan:**
1. Collect 300–800 real conveyor images spanning box sizes/materials, the existing 6-type sticker vocabulary, lighting across shifts, and multi-sticker boxes (Ultralytics' own rule of thumb tops out near 1,500 images/10k instances for general detection — scaled down here since this is one class in a fixed-camera, much more constrained setting; treat 300–500 as a floor to pilot with, validated empirically, not as a proven minimum) ([Ultralytics tips-for-best-results docs](https://docs.ultralytics.com/yolov5/tips_for_best_training_results)).
2. Run Grounding DINO offline over these images at `box_threshold≈0.35/text_threshold≈0.25` with a **deliberately chosen** prompt (resolve the discrepancy above first).
3. **Mandatory human QA pass** on every auto-label — bounding-box accept/reject/correct in a tool like CVAT or Label Studio — specifically to catch Grounding DINO false negatives and false positives (e.g. tagging a barcode patch as the whole sticker under the broader demo prompt).
4. Train YOLOX-s or RT-DETR-R18/R34 on the reviewed set, 15–20% held out, stratified by box type/lighting.
5. **Validate before trusting in production**: measure precision/recall/mAP on the held-out set, then run the new detector in **shadow mode** alongside whatever interim detector is live for 1–2 weeks of real traffic, log disagreements, have a human review a sample, and cut over only once it's equal-or-better.

**Main risk:** a trained single-class model won't generalize to box/sticker types absent from training — the exact tradeoff of specializing away from an open-vocabulary model. Mitigate by keeping Grounding DINO permanently available offline as the re-labeling tool (never deployed) and re-training periodically as new label types appear.

**Could not verify:** Grounding DINO's actual latency/VRAM on a real RTX 5070 (no direct benchmark found); whether the notebook's `transformers==4.46.3` pin is still required on current Grounding DINO/transformers — if Grounding DINO is kept even temporarily for auto-labeling, test this directly before trusting the pin.

---

### C. Barcode / QR decoding

**Keep pyzbar-first, zxing-cpp-fallback, keep the 2x-upscale-below-1000px retry** — this is already close to correct, not an area that needs a rework.

ZBar's DataMatrix gap is structural, not a bug: feature requests for it have sat open on ZBar's own tracker since 2012/2014 and were never implemented ([sourceforge.net/p/zbar/feature-requests](https://sourceforge.net/p/zbar/feature-requests)). PDF417 is listed as "supported" by pyzbar in one comparison but a user reports it not actually working in practice **[conflicting evidence — flag PDF417-in-pyzbar as unreliable, don't trust it]**. zxing-cpp confirms both DataMatrix and PDF417 support per its own docs, is **Apache-2.0**, and ships prebuilt wheels for Windows/macOS/Linux (x86-64 and ARM64) with no native system dependency ([pypi.org/project/zxing-cpp](https://pypi.org/project/zxing-cpp/)) — a materially lighter Docker story than pyzbar, which needs the `libzbar0` apt package (not pip-installable) baked into the image. ZBar's underlying C library is LGPL-2.1, which ZBar's own FAQ states is commercial-use-compatible with a static-linking exception, subject to notice/attribution obligations ([zbar.sourceforge.net licensing FAQ](https://zbar.sourceforge.net/iphone/sdkdoc/licensing.html)) — not a blocker, worth a one-line note in a license appendix.

**Multiple codes per crop:** both engines natively return all codes found in one pass (`zxingcpp.read_barcodes()` returns a collection; pyzbar's `decode()` returns a list) — no extra logic needed beyond what's already implemented.

**Alternatives considered and rejected:**
- OpenCV's classic `QRCodeDetector` — no rigorous independent benchmark found supporting it over ZBar; anecdotally weaker on multi-code/damaged images.
- OpenCV's WeChatQRCode (contrib) — one vendor benchmark shows it beating both ZBar and zxing-cpp on QR specifically (48.9% vs 39.0% vs 31.9% read rate on a 536-image set) ([Dynamsoft benchmark](https://www.dynamsoft.com/codepool/multiple-barcodes-reading-benchmark-and-comparison.html)) — QR-only, single vendor-run benchmark, worth an A/B test as an *additional* QR-specific fallback someday, not a reason to change the primary ordering now.
- **Dynamsoft Barcode Reader** — commercial, starts at **$1,499/year** ([dynamsoft.com/barcode-reader/pricing](https://www.dynamsoft.com/barcode-reader/pricing/)). Its claimed accuracy edge (per its own vendor benchmarks) is concentrated in damaged/occluded/angled real-world capture — exactly the condition this system does *not* operate under (stationary, well-lit, front-facing box at a stop bar). **Not worth paying for initially**; revisit only if production data shows a real DataMatrix/PDF417/damaged-code miss rate zxing-cpp can't close.

**Full frame first, or crops only?** Keep decoding scoped to padded crops, as today. A full-frame-first pass is a speculative optimization with a real downside (a full-frame hit doesn't tell you *which* sticker it belongs to) — only reconsider it if profiling later shows detector inference dominates latency and a meaningful share of boxes have trivially-readable full-frame codes.

**Is the retry threshold (1000px) still right?** Reasonable default, but somewhat arbitrary — log which engine/pass (pyzbar-native, pyzbar-upscaled, zxing-native) actually resolves each successful decode in production, and use that data, not intuition, to tune the threshold later.

**Main risk:** none significant — this is the most settled, lowest-risk section of the pipeline.

---

### D. OCR

| Option | License | GPU path | Blackwell/5070 risk | Accuracy signal |
|---|---|---|---|---|
| **PaddleOCR / PP-OCRv4/v5** | Apache-2.0 | PaddlePaddle GPU | **Officially supported on sm_120, verified specifically on the RTX 5070** — see reconciliation below | PP-OCRv5 claims accuracy "comparable to SOTA billion-parameter VLMs" per its own arXiv paper (2603.24373) — self-reported |
| RapidOCR (PP-OCR models via ONNX Runtime) | Apache-2.0 | onnxruntime-gpu | Lower risk — sidesteps PaddlePaddle's own wheel story; exact sm_120 kernel coverage of the stock PyPI wheel **[unverified, test directly]** | Same underlying PP-OCR models |
| docTR | Apache-2.0 | PyTorch (or TF) | **Lowest risk of all options** — PyTorch's Blackwell support is the most concretely confirmed of anything surveyed (2.7.0+, CUDA 12.8 wheels) | CER 19.7%/WER 32.0% on a SROIE receipt benchmark, 2nd of 8 engines in that one test **[single benchmark, receipts not labels, directional only]** |
| EasyOCR | Listed as both Apache-2.0 and MIT by different sources **[conflicting, verify against repo directly]** | PyTorch | Low (PyTorch-based) | Weaker than docTR in the one benchmark found |
| Tesseract | Apache-2.0 | CPU only | N/A | Baseline-grade on small/angled/glare-affected text, general knowledge not a specific benchmark |
| Surya | **GPL-3.0 code**; weights under a revenue-gated license (free under ~$2M revenue, commercial license required above) **[unverified against Surya's own primary license files]** | PyTorch | Low | Top performer in the one benchmark found (CER 0.1915) |
| PaddleOCR-VL | License not directly confirmed — check HF model card **[unverified]** | PaddlePaddle | Same Blackwell story as PaddleOCR, this is actually the pipeline with the RTX-5070-specific claim (see below) | Claims 94.5% OmniDocBench v1.5 — self-reported; likely overkill/wrong-shaped for short structured label fields vs. a classical detect+recognize OCR |

**Reconciliation (two research passes initially disagreed here):** one research pass flagged PaddlePaddle's historical lag on new CUDA architectures as a real Blackwell risk and leaned toward RapidOCR/docTR to avoid it. A separate pass, researching Docker/Blackwell specifics independently, found that **PaddlePaddle 3.2.1+ ships an official `-sm120`-tagged build and Docker image (`pip install paddlepaddle-gpu==3.2.1` from the `cu129` index, or the `paddleocr-vl:latest-nvidia-gpu-sm120` image), and PaddleOCR-VL's own documentation states accuracy/speed have been verified specifically on the RTX 5070** among Blackwell cards — not just "Blackwell in general" ([PaddleOCR-VL NVIDIA Blackwell docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html); [PaddlePaddle install tables](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html)). That's more specific and more recent evidence than the general risk flag, so it wins: PaddlePaddle/PaddleOCR is genuinely de-risked on this exact GPU, contrary to its general reputation. Note the Blackwell-verified claim is specifically on the **PaddleOCR-VL** doc page, not necessarily the classic PP-OCRv4 detect+recognize pipeline more typical for simple label OCR — the underlying PaddlePaddle 3.3.0/sm_120 framework support should carry over, but re-verify for whichever specific PaddleOCR pipeline variant is actually used.

**Recommendation: PaddleOCR (PP-OCRv4/v5) as primary**, using the Blackwell-specific pins above. Keep **RapidOCR** (ONNX Runtime, sidesteps PaddlePaddle entirely) and **docTR** (PyTorch, the single most concretely Blackwell-confirmed framework surveyed) as fallbacks if an on-box smoke test contradicts the vendor's own claim. **Explicitly reject Surya** despite its apparent accuracy edge — GPL-3.0 code plus a revenue-gated weight license is real commercial risk without further legal review, worse than the Apache-2.0 alternatives.

**First implementation task for this section:** pull the official `-sm120` PaddleOCR image on the real box and confirm `sm_120` kernels actually load before building anything on top of this choice.

**Main risk:** the Blackwell-verification claim is vendor-sourced (PaddleOCR's own docs), not independently reproduced here — treat as strong but not airtight evidence, which is exactly why a fallback chain is specified rather than a single point of failure.

---

### E. Result structure

**Precedence:** barcode-decoded payloads are authoritative for any field they encode; OCR is used for fields with no corresponding code and as a cross-check against barcode-decoded fields otherwise.

**Barcode-first field recognition, concretely:**
- A UPS tracking number is Code 128-encoded, 18 characters, starting with `1Z` ([Seagull barcode guide](https://barcodeguide.seagullscientific.com/Content/Symbologies/ups_tracking.htm)) — recognizing the `1Z` prefix + length lets you classify a decoded payload without any OCR.
- GS1-128 Application Identifiers (FNC1 + AI prefix) are a documented, checksum-verifiable scheme for SSCC/lot (AI 10)/quantity (AI 30/37) fields on shipping labels ([GS1-128 reference](https://racklify.com/encyclopedia/what-is-a-gs1-128-shipping-label-and-how-it-encodes-ssccs/)) — the single most concrete, symbology-documented path to high-confidence structured extraction with zero ML involved.
- A reference table of per-carrier tracking-number length/checksum conventions exists ([github.com/trycom/tracking_number_data](https://github.com/trycom/tracking_number_data)) but the exact checksum algorithms per carrier were **not confirmed in detail** — verify before hard-coding. FedEx's exact barcode symbology spec was **not found** at all — source this from FedEx's own developer docs before implementation.
- Validate decoded payloads against known checksum algorithms (UPS `1Z`, USPS mod10/S10) where documented; a failed checksum is an automatic low-confidence flag even if the barcode nominally decoded.

**OCR-side extraction:** keyword anchoring (`SHIP TO:`, `TRACKING:`, `LOT`, `QTY`, `SKU` — treat text immediately following as the value) as the backbone, plus regex for well-known shapes (tracking-number-shaped strings even without a decodable barcode, ZIP codes, warehouse-specific SKU formats, quantity patterns). Cross-check: when both a barcode-decoded and an OCR-read human-readable version of the same field exist (standard on shipping labels — barcodes almost always have human-readable text printed alongside specifically for this purpose), a mismatch is a stronger, more specific low-confidence signal than either source's own confidence number.

**Low-confidence flagging:** extend, don't replace, the existing mechanical rule (`confidence<0.70 OR type=="unknown"` forces `needs_verification`). Add: checksum failure, barcode-vs-OCR mismatch for the same field, a code visually present but undecoded, a required field entirely absent from both sources. Several independent, cheap, unit-testable triggers beat relying on one model's self-reported confidence — a meaningfully more robust design than the current always-on-VLM status quo.

**Where Gemma plugs in:**
- **Receives:** the crop only, never the full frame. Argument is both privacy (the crop alone is strictly less data-exposure surface than the full box/pallet/surrounding-conveyor context) and cost/latency (bandwidth and per-call cost scale with image size for a cloud VLM call). Note explicitly: a cropped sticker still carries PII (name, address) — "crop only" is a necessary mitigation, not a sufficient one; see §4.G for the full privacy discussion.
- **Returns:** keep the existing JSON contract shape (`raw_text` first, `fields`, `has_machine_code`, `type`, `summary`, `confidence`, `needs_verification`) so merge/validation code can be shared between the local-result path and the optional-cloud-result path rather than maintaining two schemas. Consider adding per-field confidence (not just one whole-sticker number) as a design improvement — this is a recommendation, not something found in research.
- **Merge policy:** enrich/cross-check, never blind-override. A checksum-validated barcode value is never silently replaced by Gemma's read; a disagreement between them is logged as a *stronger* review signal than either source's confidence alone. For OCR-only fields, Gemma's answer can enrich/replace the local extraction if its confidence clears a threshold and doesn't contradict a keyword-anchored local extraction too strongly. If local extraction found nothing for a required field, Gemma's answer is taken as the only source, still gated by the mechanical rule.
- **When to call it:** only when local confidence is low or validation fails — **not on every sticker.** Reasons: cost (a paid per-call cloud API), latency (a network round-trip is orders of magnitude slower than the entire local pipeline, which defeats the point of a fully-offline default path), and privacy (fewer crops ever leave the premises). This reuses the exact trigger logic that already decides `needs_verification` today, which is a natural extension of a mechanism already proven out in the existing code, not a new invention.

**Main risk:** the exact checksum/symbology rules for several carriers (FedEx especially) need primary-source verification before being hard-coded — don't ship unverified checksum logic as if it were confirmed.

---

### F. Camera and trigger hardware

**1. Settling.** A fixed delay alone is fragile (one number has to cover the worst case, box-mass/bounce-dependent variance is likely high for gravity-fed arrival). Recommended combination: a break-beam sensor as the "box arrived, start paying attention" event (sub-ms sensor latency, free) → a **consecutive-frame stability check** on a cheap low-res preview stream (compare successive frames, require several stable frames in a row before triggering — this is the specific approach recommended by both an OpenPnP wiki page on exactly this problem and a Roboflow guide on detecting when a part has stopped moving) ([OpenPnP wiki](https://openpnp-wiki.readthedocs.io/en/add-mkdocs/wiki/Camera-Settling.html); [Roboflow](https://roboflow.com/answers/detect-part-stops-moving)) → bounded by a fixed max-wait safety net (~300–500ms) so a box that never fully stills still gets photographed → a short sharpness-scored 3–5-frame burst capture at the moment stability is detected, keeping the sharpest frame (a pattern drawn from burst-photography keeper-frame-selection patents, extrapolated to this use case, not a proven industrial pattern — **[reasoned extrapolation]**) ([US patent 10,523,894](https://patents.google.com/patent/US10523894)). One integrator source puts stopper-station dwell time at ~50–200ms typical ([iFactory conveyor integration guide](https://ifactoryapp.com/ai-vision-camera/conveyor-integration-vision-stations-manufacturing)) — a useful anchor, but settle time should be measured empirically (see Phase 0) before finalizing a budget.

**2. Position/pose/FOV/DOF.** Exact box-position/skew/sticker-placement tolerance at your specific stop bar **could not be found or estimated from general sources — this is a direct question for you** (see §7). What general rules of thumb give you once you know that tolerance: OCR legibility needs roughly **20–25px per character dimension**, smallest stroke ≥3px ([Automate.org](https://www.automate.org/vision/industry-insights/considerations-in-ocr-ocv-applications/anc); [The Imaging Source](https://www.theimagingsource.com/en-us/resource/glossary/image-resolution-in-machine-vision/)) — translating a plausible FOV and label text size into a 5–12MP sensor requirement rather than a 1–2MP webcam sensor. For depth of field across varying box heights, a **higher f-number (f/5.6–f/11)** trades light for DOF headroom, which is the standard machine-vision answer for exactly this "objects of varying height on a conveyor" scenario ([The Imaging Source lens guide](https://www.theimagingsource.com/en-us/resource/glossary/how-to-choose-a-lens-for-machine-vision/)); a telecentric lens is the documented alternative if the height range turns out to be large.

**3. Lighting.** Diffuse LED bar/ring lighting (CCS America, Advanced Illumination tier, roughly **$600–$1,700** — e.g. CCS LDL2 bar ~$728–$1,655, Advanced Illumination bar lights $625–$920) ([Graftek](https://graftek.com/?p=9160); [Edmund Optics](https://www.edmundoptics.com/p/6-white-ultraseal-washdown-bar-light/57174/)), **polarized** to kill glare on glossy labels (costs light throughput, so budget a brighter base fixture), inside a **light-blocking hood/shroud** over the capture zone — explicitly recommended by machine-vision guidance as the most reliable fix against bay-door daylight and fluorescent-flicker drift on a fixed-exposure camera ([vision-doctor.com](https://www.vision-doctor.com/en/solving-daylight-problems.html)).

**4. Trigger.** Hardware camera-trigger-input latency is microsecond-scale and deterministic; a software trigger (sensor → ESP32 → USB-CDC → host → camera API call) is millisecond-scale and variable ([The Imaging Source](https://www.theimagingsource.com/en-us/resource/glossary/hardware-triggering-and-strobe-synchronization/); [Baumer](https://www.baumer.com/ca/en/when-its-a-matter-of-precision-process-synchronization-with-industrial-cameras/n/process-synchronization)). Given settle time is hundreds of milliseconds regardless, a few ms of software-trigger jitter is a rounding error — **a software trigger over USB-CDC is adequate; don't pay a premium for a hardware-trigger-input camera to chase microsecond determinism against a multi-hundred-ms bottleneck.**

**5. Camera candidates.** A USB3 Vision camera in the **5–12MP class** is the right tier (stationary capture means no rolling-shutter motion-blur concern, but OCR legibility needs real resolution, ruling out a plain webcam). Budget option: Daheng Imaging USB3 Vision board-level cameras from resellers, ~$230–$390, via the **Aravis** (LGPL-2.1) library for vendor-agnostic GenICam access on Linux. Supported option: **Basler ace2**, $300–$2,400 depending on sensor, via the **pylon SDK**, which has the most mature published Linux/Docker integration documentation of the options surveyed, including a dedicated guide for using Basler USB cameras in a Docker container. USB device passthrough (`--device=/dev/videoN` or the whole USB bus plus a cgroup rule) is the standard Docker pattern; avoid GigE Vision unless the camera needs to sit far from the host (USB3's practical cable-length ceiling is a few meters without a repeater) — GigE in Docker needs `--network host`, since macvlan has been reported broken for camera discovery by multiple users.

**6. iPhone — rejected for a permanent install.** Distribution has **no simple, free, durable option** at any tier found: free provisioning expires in 7 days; the Developer Program is **$99/year**; Enterprise is **$299/year** and likely requires 100+ employees to even qualify ([developer.apple.com comparison page](https://developer.apple.com/support/compare-memberships); third-party eligibility summary — **[unverified against an Apple primary source]**). This is a **recurring, indefinite cost with administrative overhead**, something the camera+ESP32 route simply doesn't have. Separately, and more disqualifying for a 24/7 kiosk: current Apple Community threads (2025–2026) report sustained camera use thermally throttling recent iPhones within **15–20 minutes**, and MagSafe wireless charging explicitly pauses during continuous camera use ([Apple Community threads](https://discussions.apple.com/thread/255768328); [9to5mac](https://9to5mac.com/2024/05/31/psa-magsafe-probably-isnt-charging-your-iphone-when-using-continuity-camera)) — a currently-documented reliability defect for exactly the use case being proposed. LiDAR's depth map (~256×192 resolution, ~5m range per third-party sources — **[unverified against Apple's own docs]**) is useful only for coarse presence/height sensing, which a $10–30 ToF sensor does equally well; it adds nothing for reading a flat sticker. The iPhone may still be a reasonable zero-hardware-build *prototyping* tool (walking around snapping label photos for training data) — a very different, non-24/7 use case from what's being evaluated here.

**7. Off-the-shelf alternatives — not a fit.** Smart cameras (Cognex In-Sight, Keyence IV3, roughly **$1,200–$5,000+**, quote-only) bundle camera+lighting+barcode+basic OCR, but their OCR is tuned for fixed-format text (lot/date codes in a known font/position), not open-vocabulary shipping labels with variable layout from many shippers — exactly the variability the current Gemma-based reading step exists to handle — and they'd duplicate GPU spend already committed to for the rest of this pipeline. Dedicated barcode tunnels/presentation scanners (Datalogic/Honeywell, **$300–$12,000** depending on class) do barcode decode only, no OCR at all — a reasonable *optional cross-validation add-on* later if barcode read-rate specifically proves to be the weak link, never a full replacement.

**Recommendation and layout:** ESP32 + break-beam + USB3 Vision camera (5–12MP) + polarized diffuse LED in a hood, software trigger adequate.

```
Side view:
                       diffuse LED bar/ring (polarized)
                                \         /
                 light-blocking  \       /
                 hood/shroud --> +-------+
                                 |   C   |  <- camera, straight down
                                 +---+---+
                                     | FOV cone
                                ____/|\____
                      _________/_____|_____\_________
   gravity roller -> /         : BOX |     :          \  stop bar
   bed (box slides)             break-beam sensor      (fixed, box
                                 at the bar  |           rests here
                                             |           after settling)
                                          ESP32 --USB--> Linux PC (RTX 5070)

Top view:
   travel -->
   +--------------------------------------------------+
   |   wide roller bed, lateral wander + skew possible |
   |                              ================ <- stop bar
   |                              | capture zone  ||
   |                              |    [BOX]      ||
   |                              ================||
   |                                     ^            |
   |                            break-beam emitter/   |
   |                            receiver pair          |
   +--------------------------------------------------+
```

**Could not verify/needs your input:** actual box-position/skew/sticker-placement tolerance at your real stop bar (general sources give no numbers; this is inherently site-specific — see §7).

---

### G. Deployment and operations

**Blackwell/CUDA/driver specifics for the RTX 5070** (consumer Blackwell, compute capability `sm_120` — architecturally distinct from datacenter Blackwell B100/B200 at `sm_100` and the DGX Spark GB10 at `sm_121`; code built only for those will not run on this card): minimum Linux driver **R570+** ([NVIDIA Blackwell RTX migration guide](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)); current production driver branch is **R620** as of Nov 2026 ([Phoronix](https://phoronix.com/news/NVIDIA-620.70.06-Linux-Driver)) — just install current rather than hand-pinning an old minor version. **CUDA 12.8 is the first toolkit with native sm_120 support**; the ecosystem has already moved to 12.9/13.x as the "stable" default in PyTorch's own release notes by late 2026 ([PyTorch dev-discuss](https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325)) — re-check each framework's install matrix at build time rather than hardcoding a version now. **TensorRT 10.8+** is the first version tied to Blackwell support ([NVIDIA dev blog](https://developer.nvidia.com/blog/new-ai-sdks-and-tools-released-for-nvidia-blackwell-geforce-rtx-50-series-gpus/)).

Framework-by-framework Blackwell wheel risk, most to least risky: **ONNX Runtime GPU** (stock wheel's exact sm_120 SASS coverage for your pinned version is unconfirmed — NVIDIA's own guide recommends being ready to build from source with `CMAKE_CUDA_ARCHITECTURES=120` if the prebuilt wheel fails, since the CUDA EP doesn't ship PTX fallback) ≈ **PaddlePaddle** (safe at 3.2.1+/CUDA 12.9+, with an official `-sm120` Docker image — see §4.D) ≈ **PyTorch** (safe if the CUDA-12.8+/13.x index is pinned explicitly — `pip install torch` with no index URL is a real footgun that can silently grab a CPU-only or stale-CUDA wheel) > **TensorRT** (safest — NVIDIA's own first-party SDK, clean 10.8+ cutoff).

**Docker base image:** prefer an NGC PyTorch/TensorRT container (NVIDIA states 25.01+ NGC deep-learning containers are optimized for Blackwell) or the official PaddleOCR `-sm120` image over a bare `nvidia/cuda` base — this offloads the "does the wheel actually contain sm_120 SASS" risk onto NVIDIA's/Baidu's own build and test matrix ([PyTorch 25.02 NGC release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html)). Even a "Blackwell-optimized" container can have sub-component gaps (the 25.02 PyTorch NGC container's own release notes flag cuSPARSELt and nvFuser as *not* supporting sm_120) — check the exact pinned tag's release notes, don't assume the label covers everything. Note `nvidia/cuda`'s `latest` tag is deprecated on Docker Hub; always pin an explicit tag.

**"Works out of the box":** **bake model weights into the image**, don't volume-mount them — this is a single air-gapped machine with one non-ML technician and no shared model registry, so one atomic self-consistent artifact beats a code+weights pair that can silently drift apart. Version by semantic tag **and** digest pin in `docker-compose.yml` (tags are mutable, digests aren't); keep the last N images on disk for one-line rollback; wrap everything in a small Makefile (`make deploy VERSION=...`, `make rollback VERSION=...`, `make status`, `make logs`) so the technician never types raw `docker` commands; gate a deploy on a health-check pass before declaring success, auto-rolling back if it fails.

**Output interface:** a **file/DB drop** (one JSON file, or a SQLite/Postgres row, per box) is the simplest option that's still robust — no broker process to keep alive, trivially recoverable across restarts on either side, easiest for a non-ML technician to inspect directly (`cat`/`sqlite3` beats debugging a broker or HTTP client). A thin REST endpoint can layer on top later if real-time push is actually needed. Reject a message broker (Redis/RabbitMQ/MQTT) for now — disproportionate operational surface (another process that can crash, another thing in the health check) for a single-PC, single-downstream-consumer deployment.

**Reliability:** `restart: unless-stopped` + a Docker `HEALTHCHECK` directive is the right level of process supervision at this scale (no external supervisor needed). `json-file` logging driver with explicit `max-size`/`max-file` rotation — Docker does **not** rotate `json-file` logs by default, so this must be set explicitly or logs grow unbounded. Image/audit retention: at 1–5MB per captured image, rough fill-times on the 2TB SSD range from **~7 months at low throughput (~2,900 boxes/day) to ~14 days at high throughput (~43,200 boxes/day)** — implement a retention policy as code from day one (keep N days full-res, then delete/downsample, plus a hard disk-usage alarm at e.g. 80% full), don't rely on manual cleanup. Minimal health view: a `/health` endpoint reporting last successful capture time, GPU memory, queue depth, rolling error rate — plus `docker compose logs -f` is enough; a full Grafana-style dashboard is more operational overhead than this deployment's scale warrants.

**Cloud Gemma flag plumbing:** a boolean env var (`ENABLE_CLOUD_VLM=false` default) sourced from a `.env` file **mounted into the container, never baked into the image** (listed in `.dockerignore`), with a shipped `.env.example` template defaulted to false. The cloud call must carry a **hard client-side timeout** and the local result must be emitted immediately without waiting on it — if the cloud call succeeds later, enrich/update the already-emitted record; if it fails or times out, log and move on. A circuit-breaker-style cooldown after repeated failures avoids every box paying a multi-second timeout cost during an outage. **Privacy:** sending only the cropped sticker (never the full frame) meaningfully narrows exposure (no box contents, no facility layout, no other staff in frame) but is **not fully sufficient** — the sticker itself still typically carries a full name and address, i.e. real PII, in transit to a third party. This is as much a policy/legal decision (the chosen cloud provider's data-retention/training-use terms, and the warehouse's own privacy obligations to its customers) as an engineering one — flag it as such rather than letting the "crop only" mitigation read as a complete answer. Recall also from §2: the existing Gemma gateway's actual client/credential-handling code isn't visible in this repo subset, so this client has to be built fresh from the request/response shape, not lifted.

**Failure behavior to define explicitly:** no sticker found → return `sticker_count: 0`, not an error. Unreadable sticker → return the result with `needs_verification=true` and whatever partial fields were extracted, never silently drop it. Camera disconnect → the trigger/capture layer should report a distinct "camera unavailable" status, not a decode/detect error, so ops can tell hardware failure from pipeline failure at a glance. GPU error → fail the box with a clear error status (not a hang); the health endpoint should reflect a degraded GPU state. Cloud timeout → never affects the local result's delivery, per the non-blocking design above; only the enrichment is missing, logged as such.

---

### H. Latency budget

| Stage | Target | Basis |
|---|---|---|
| Settle-wait | *(handled by camera/trigger, see §4.F)* | ~150–300ms+ plausible, per stopper-station dwell-time figures — must be measured empirically |
| Capture | 5–15ms | typical USB3/GigE frame grab |
| Detector inference (small TensorRT detector) | 5–15ms | extrapolated from T4 (~1–1.5ms for YOLOv8n/YOLOv11n with TensorRT FP16) and RTX-4070-class (~3ms) benchmarks ([arXiv 2607.11356](https://arxiv.org/pdf/2607.11356); [Ultralytics comparison page](https://docs.ultralytics.com/compare/yolo11-vs-yolov8)) — **cross-generation extrapolation, re-measure on the real 5070** |
| Crop | <2ms | array-slicing once detector boxes are known |
| Barcode/QR decode | 5–20ms | CPU-bound, roughly doubles worst-case if the zxing-cpp fallback chain runs |
| OCR inference | 15–50ms | wide range — published PP-OCR numbers conflict significantly by model variant/image size (e.g. 12.8ms/page on an RTX 5080 in one third-party benchmark vs. ~99–128ms for the server detector model alone elsewhere) ([GigaGPU benchmark](https://gigagpu.com/paddleocr-on-rtx-5080-benchmark/)) — **must be re-measured on actual cropped-sticker-sized inputs with the chosen model, not assumed from any one published number** |
| Field extraction/parsing | <5ms | regex/rule-based, cheap in any language |
| Result output | <10ms | local disk write or local REST round-trip |
| **Local pipeline total (excl. settle-wait)** | **~50–120ms** | sum with margin |
| Cloud Gemma call (separate, async, non-blocking) | 1–5+ seconds, hard timeout in that range | network RTT + cloud inference, tracked as its own metric stream, never folded into the box's local-processing SLA |

**Achievable throughput:** at a ~50–120ms local compute budget, the pipeline could in principle sustain on the order of 500–1,200 boxes/minute run back-to-back — but in virtually any real deployment, **physical settle-wait and conveyor arrival cadence will be the actual bottleneck, not this compute budget.** State this plainly rather than over-engineering the compute path against a ceiling it likely isn't.

**Benchmarking methodology for the real machine:** structured per-stage timers (one JSON line per box, stage→duration_ms), not just an end-to-end number; discard a 20–50-box warm-up period before measuring (CUDA context init, TensorRT engine build, cuDNN algorithm auto-tuning are all slow on the first few calls); report **p50/p95/p99, not mean** — tail latency is what determines whether a target boxes-per-minute rate holds up without a backlog forming; run under **sustained, realistic load** for tens of minutes, not a handful of isolated frames, to surface thermal throttling, GPU memory fragmentation, or disk-write backlog that a short test would miss; track the cloud Gemma call's latency distribution completely separately, since its variance characteristics are fundamentally different from the local stages.

**Could not verify:** exact OCR/detector latency on the real RTX 5070 with the actual chosen models and real cropped-sticker-sized inputs — every number above is an extrapolation from other GPUs/models and must be re-measured, which is exactly why it's listed first in the phased plan below.

---

## 5. Proposed modular-monolith layout

```
capture/        camera interface, ESP32 trigger listener (USB-CDC serial), frame grab, EXIF/orientation normalize
settle/         consecutive-frame stability check, bounded max-wait timeout, burst capture + sharpness scoring
detect/         single-class sticker detector (TensorRT/ONNX engine), loaded once at startup, kept warm
crop/           padding/crop math (8% default), min-crop-side filter, long-side cap
decode/         pyzbar-then-zxing-cpp wrapper, 2x upscale retry below 1000px
ocr/            OCR engine wrapper (PaddleOCR primary, RapidOCR/docTR fallback), loaded once at startup
extract/        symbology/checksum rules, regex + keyword anchoring, barcode/OCR merge, sticker vocabulary + mechanical verification rule
cloud_enrich/   optional Gemma client — only constructed at all if the flag is true; crop-only payload, hard timeout, non-blocking
pipeline/       orchestrator: plain function calls box->box, in-memory arrays end to end, per-stage timers
output/         result sink (file/DB drop default), image retention/rotation
config/         env/flag loading (ENABLE_CLOUD_VLM, thresholds, paths, credentials)
ops/            health endpoint, logging setup, retention cron, error-rate tracking
```

**Data flow:** camera frame → `settle/` loop on a low-res preview until stable or timeout → best-of-burst full-res frame → `detect/` (boxes, in-process, no HTTP) → `crop/` (numpy arrays, no serialization, no base64) → `decode/` and `ocr/` run per crop → `extract/` merges barcode+OCR into fields, applies the mechanical verification rule → **if** the cloud flag is on **and** `needs_verification` is true: `cloud_enrich/` runs async with a hard timeout, crop-only payload, result merges in (never overrides a checksum-validated field) when/if it returns → `output/` writes the result (and retains the image) → `ops/` updates health/error-rate metrics. The local result is never blocked waiting on `cloud_enrich/`.

---

## 6. Phased plan — cheapest experiment first

**Phase 0 (no purchases, settles the biggest unverified claims cheaply):**
- On-box CUDA/Blackwell smoke test: pull `torch`, `onnxruntime-gpu`, `tensorrt`, and the PaddleOCR `-sm120` image on the real machine; confirm each reports `sm_120`/Blackwell kernel support before committing further to any of §4.A/B/D's choices.
- A short phone video of real boxes hitting the actual stop bar — settles the settle-time, bounce, lateral-wander, and skew unknowns flagged in §4.F cheaper than buying any camera.
- A Grounding DINO latency/VRAM benchmark on a handful of real sample images on the real GPU — settles whether even temporary use for auto-labeling is as cheap as assumed.

**Phase 1:** build the modular-monolith skeleton (capture via any webcam or sample images → crop → decode → OCR → extract) end to end, measure true per-stage latency with real timers, and lock in the OCR engine choice based on the Phase 0 smoke-test result rather than this document's extrapolation alone.

**Phase 2:** data collection + detector — collect real box images, auto-label with Grounding DINO (prompt resolved per §4.B), mandatory human QA pass, train YOLOX-s/RT-DETR, export to TensorRT/ONNX, swap into the pipeline, validate in shadow mode before cutover.

**Phase 3:** real camera/trigger hardware — procure the ESP32, break-beam sensor, chosen USB3 Vision camera, and lighting/hood; replace the Phase 1 placeholder capture with the real trigger/settle-detection path.

**Phase 4:** deployment hardening — finalize Docker/Compose with the Blackwell-specific base image, health/logging/retention, the optional Gemma flag's env/secret plumbing, and the Makefile-wrapped deploy/rollback tooling.

**Phase 5:** shadow-mode pilot — run the full new pipeline alongside whatever is currently handling outbound (even if that's a manual process) for a real production period before full cutover.

---

## 7. What could not be verified, and questions for you

**Could not verify (be honest about this before implementation starts):**
- Grounding DINO's actual latency/VRAM on a real RTX 5070 — every number in this document is extrapolated from other GPUs, not measured.
- Exact box-position/skew/sticker-placement tolerance at your specific stop bar — no general source gives numbers for this; it's inherently site-specific.
- Exact OCR inference latency on real cropped-sticker-sized inputs — published PP-OCR numbers conflict by model variant and image size.
- Whether the notebook's `transformers==4.46.3` pin is still required for current Grounding DINO.
- FedEx's exact tracking-barcode symbology specification (not found in this research at all).
- Whether PDF417 decoding in pyzbar actually works in practice — one source lists it as supported, a user report says it doesn't; treat zxing-cpp as the authority for PDF417 regardless.

**Five questions for you:**
1. What are the box size/height range and the sticker size/placement tolerance at the stop bar? This directly sizes field of view, required resolution, and depth-of-field/lens choice in §4.F, and nothing in general machine-vision literature can substitute for your actual numbers.
2. What's the target throughput (boxes/minute)? This validates whether the proposed settle-wait + compute latency budget in §4.H is acceptable and sizes the disk-retention policy in §4.G.
3. Exactly which fields does the downstream system need (tracking number, address, SKU, lot, quantity — and in what format), and is a file/DB drop (§4.G's recommendation) acceptable, or does it need an HTTP push/webhook instead?
4. How is the cloud Gemma endpoint actually exposed — host/port, auth scheme, which provider — so the optional client in `cloud_enrich/` can be designed concretely rather than against an assumed shape?
5. Can you provide sample label photos, and ideally a short video of real boxes hitting the actual stop bar? This seeds both the Phase 0 settle-time measurement and the Phase 2 detector training data, and is the cheapest possible way to replace several of this document's extrapolations with real numbers.
