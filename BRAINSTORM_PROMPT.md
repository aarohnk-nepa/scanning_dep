# TASK: Brainstorm and research only. Do NOT implement anything.

You are acting as a senior systems architect. I want a design brainstorm and a written
recommendation, not code. Do not modify, create or delete any source files. The only file you
may write is `BRAINSTORM.md` at the repo root. Small throwaway benchmark scripts in a scratch
directory are fine if they help settle a question, but say so when you use one.

## Context

I'm building a warehouse OUTBOUND system. A camera sits over the end of a GRAVITY-FED
conveyor. A bar at the end of the conveyor stops each box, so the box is STATIONARY (after it
settles) when the picture is taken. The main sticker faces the camera. The image goes to
software that extracts the sticker data and returns it. The default runtime path must run
entirely ON-DEVICE inside the warehouse, with no cloud calls.

Target host (a single PC running LINUX, I can SSH into it):
CyberPowerPC gaming desktop, Intel Core Ultra 7 270K Plus, NVIDIA RTX 5070 12GB,
32GB DDR5, 2TB SSD. The GPU is Blackwell: check what CUDA version, driver and framework
wheels (PyTorch, ONNX Runtime, TensorRT, PaddlePaddle) it needs, and what that means for
Docker base images.

The finished repo will be copied onto that machine and must work out of the box, so Docker
(with GPU access) is a hard requirement.

## What exists today (read all of it first)

Every file in this directory came from a previous design. Read them all, plus the attached
Grounding DINO notebook (`hosting-grounding-dino__1_.ipynb`):
`agent.py`, `segmentor.py`, `scanner.py`, `code_reader.py`, `schemas.py`,
`segmentation_api.py`.

Current pipeline: an agent platform orchestrates two tools.
1. `vision.segmentor` POSTs the image to a separate Grounding DINO HTTP service, gets
   detections in an `X-Detections` header, and crops each box with 8% padding.
2. A Gemma multimodal model reads each crop, one request per crop, through a gateway service.
3. `vision.scanner` decodes QR codes/barcodes with pyzbar first, zxing-cpp as fallback,
   with a 2x upscale retry for small crops.

Treat this as a record of lessons, not as a structure to preserve. Note what is worth
keeping (crop padding, pyzbar-then-zxing ordering, the retry, per-crop error isolation,
EXIF/coordinate-frame handling, the mechanical needs-verification rule, the closed sticker
type vocabulary) and what is overhead to delete.

## Hard design constraints

1. REMOVE ALL ORCHESTRATION OVERHEAD. No agent framework, no tool registry, no event bus,
   no tenant/auth layers, no HTTP hop between internal stages, no base64 round-trips between
   stages. Pixels stay in memory as arrays from capture to result.
2. Target shape: a MODULAR MONOLITH, one process, optimized for latency and throughput.
   Clear internal module boundaries, but plain function calls between them.
3. Models are loaded once at startup and kept warm on the GPU. Measure and report
   cold-start time separately from steady-state per-box time.
4. The default path is fully offline. Docker (GPU-enabled) is mandatory; ship it so a
   technician can run one command.
5. Pipeline to evaluate: capture -> sticker detection -> crop with padding -> barcode/QR
   decoding + OCR -> structured result.
6. Gemma (the VLM) is KEPT but reduced to an optional step controlled by a single boolean
   setting that DEFAULTS TO FALSE. The model itself is hosted in the CLOUD, not on the
   warehouse machine, so no VRAM is spent on it locally. When the flag is false, the
   pipeline must not import, configure or contact anything cloud-related. When true, it is
   a fallback or enrichment step, and you should recommend when it triggers (for example
   only for stickers whose OCR/barcode result is low-confidence or fails validation, versus
   every sticker). Design it so the system never blocks on it (hard timeout, cached or
   queued result, graceful degradation to the local result).

## Questions to investigate and answer

### A. Language and runtime
My instinct is Python, because the heavy work is GPU inference in native code and Python
gives the best ecosystem (pyzbar, zxing-cpp, OCR libraries, TensorRT/ONNX bindings). Challenge
this. Compare Python, Rust and C++ for THIS workload. Where does time actually go per box?
Is there any step where Python glue would be the bottleneck? Note that the implementation
will later be done by Claude Opus 5.5, so weigh how reliably it writes each language and
how easy the result is to debug over SSH. Give a clear recommendation and what evidence would
change it.

### B. Detector
Grounding DINO is accurate but heavy for a single-class problem on clean, stationary images.
Compare: (1) keep Grounding DINO (what are its real latency and VRAM on a 5070?), (2) use
Grounding DINO offline only to auto-label real box photos and train a small single-class
detector (YOLO-family or similar) exported to TensorRT/ONNX, (3) classical CV for a facing,
well-lit, roughly fixed-position sticker. Consider licensing of each candidate. Recommend one
and describe the data-collection and labeling plan it needs.

### C. Barcode / QR decoding
Keep pyzbar-first, zxing-cpp-fallback? Compare against alternatives for multiple codes per
crop, damaged codes, DataMatrix/PDF417 (which ZBar cannot read), speed and Docker packaging.
Should decoding run on the full frame first and crops second, or the reverse? Is the 2x
upscale retry still sensible?

### D. OCR
I was leaning towards PaddleOCR but want the BEST on-device option for a 12GB RTX 5070
reading printed shipping labels. Compare candidates (for example PaddleOCR/PP-OCR, EasyOCR,
docTR, RapidOCR, Tesseract, and any current high-accuracy small models you find). Judge on
accuracy for small printed text, speed on GPU, install/Docker pain on Blackwell GPUs, and
licensing. Also consider fusing: barcode text should be preferred over OCR for the same
field, with OCR as cross-check and for text that has no code.

### E. Result structure
Without a local LLM, how do we turn OCR text into structured fields (tracking number,
address, SKU, lot, quantity)? Propose a simple, testable approach (regexes, label-keyword
anchoring, barcode-symbology rules) and how to flag low-confidence results for human review.
Decide how the optional cloud Gemma step plugs in: what it receives (only the crop, never
the full frame?), what it returns, and how its answer is merged with or checked against the
local result.

### F. Camera and trigger hardware (RESEARCH THIS PROPERLY, with web search)
Because the box is stopped by a bar, motion blur is mostly NOT the issue. The real
questions are:
- Impact and settling: a gravity-fed box hits the stop bar and may bounce, rock, rotate or
  vibrate. How do we know it is settled before capturing (sensor timing, a short delay,
  consecutive-frame stability check, or capture-and-pick-best)? Quantify the tradeoff
  against throughput.
- Position and pose tolerance: how much does box position, skew and sticker placement vary
  at the stop bar, and what does that mean for field of view, resolution and focus
  (depth of field with boxes of differing heights)?
- Lighting: diffuse LED, glare on glossy labels, ambient warehouse light changes, and
  whether an enclosure/hood is worth it.
- Trigger: option 1 is a USB or industrial camera plus an ESP32 with an IR/photoelectric
  sensor at the stop bar. How does the trigger reach the host reliably (USB serial, network,
  GPIO), debouncing, and hardware vs software trigger latency?
- Camera candidates and rough costs: UVC webcams vs machine-vision cameras (e.g. USB3
  Vision/GigE) and which Linux/Python SDKs they need inside Docker, including device
  passthrough to the container.
- Option 2 is an iPhone using its back camera and LiDAR depth. Investigate what a custom
  iOS app needs today for installation and distribution (Apple Developer Program cost and
  the limits of free provisioning, TestFlight, enterprise or custom distribution), how the
  phone would send frames to the host (network, latency, reliability), whether LiDAR depth
  adds anything for READING a flat sticker (resolution, range) or only for box
  presence/position/height, and whether a phone left running 24/7 at a stop bar is
  operationally sensible (heat, battery, updates, auto-lock). Since the box is stationary,
  judge this option mainly on operational burden, not image quality.
- Whether an off-the-shelf alternative (fixed-mount industrial smart camera, dedicated
  barcode scanner tunnel) changes the picture, as a sanity check on our approach.
Give a recommendation with a cost and risk comparison, and a simple physical layout sketch
in text.

### G. Deployment and operations
- Docker and GPU on Linux: NVIDIA driver, Container Toolkit, base image choice, and
  anything specific to the RTX 5070 (Blackwell).
- "Works out of the box": how models and weights are baked in or mounted, what is pinned,
  and how a technician installs, updates and rolls back over SSH.
- Output interface: how do downstream systems receive results (local HTTP endpoint,
  message queue, file/DB drop)? Pick the simplest that is robust, and note what the
  warehouse's WMS side would need.
- Reliability: watchdogs, auto-restart, logging, storing every image with its result for
  audit and retraining, disk growth on the 2TB SSD, and a health/status view.
- Cloud Gemma flag: where the setting lives, how the endpoint and credentials are supplied
  without baking secrets into the image, what happens when the warehouse internet is down,
  and the privacy implication of customer photos leaving the site (crop only? redaction?).
  Check how the existing gateway service in `agent.py`'s imports is likely to be structured
  and what must change for a cloud endpoint, noting anything you cannot see.
- Failure behaviour: no sticker found, unreadable sticker, camera disconnect, GPU error,
  cloud timeout. What does the system tell the conveyor-side consumer?

### H. Latency budget
Propose a per-box latency budget (settle wait, capture, decode, detect, crop, barcode,
OCR, parse, output, and the optional cloud call listed separately), rough target numbers,
and how to benchmark each stage on the real machine. Estimate achievable boxes per minute.

## Deliverable

Write `BRAINSTORM.md` containing:
1. A one-page executive recommendation (language, detector, OCR, barcode engines, camera
   and trigger route, deployment shape, Gemma flag design).
2. For each section A-H: options considered, comparison table where useful, your
   recommendation, and the main risk.
3. A proposed modular-monolith layout (directories/modules, one line each, plus the data
   flow between them). No code.
4. A phased plan: what to prove first with the cheapest experiment, then build order.
5. The list of things you could NOT verify (be honest about uncertainty), and the specific
   questions you need ME to answer before implementation (box sizes and height range,
   sticker sizes and placement tolerance, throughput target, which fields downstream needs,
   how the cloud Gemma endpoint is exposed, sample label photos).

Rules: cite sources for hardware, pricing, compatibility and version claims, and prefer
current sources since these things change quickly. Mark anything from memory as
unverified. Disagree with me where you think I'm wrong. Ask me at most five questions at the
end rather than stopping to ask in the middle.