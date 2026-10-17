# Reader image size policy and OCR engine

Measured on 2026-10-17 against LM Studio with `qwen/qwen3-vl-8b` (loaded context 13721 tokens).
Test pages were synthetic: a printed page rendered at 300 dpi, then warped into a simulated
3024x4032 phone photo (perspective, rotation, shadow gradient, noise). Real phone photos are
untested. CER/WER are against the known source text.

## Does the READER accept a full-resolution photo?

Yes. A 3024x4032 JPEG is accepted at the current context. LM Studio downsizes images itself,
so the cost is capped (~3.1k prompt tokens) and the 13.7k context is not at risk from the image.
Output length is the real constraint: a dense page transcribes to ~1k tokens, so keep
`max_tokens` generous and treat `finish_reason == "length"` as a failure, never as a result.

| long edge sent | image size | prompt tokens | time | CER (dense) | CER (normal) |
|---|---|---|---|---|---|
| full | 3024x4032 | 3103 | 36 s | 0.002 | 0.011 |
| 2048 | 1536x2048 | 3103 | 36 s | 0.002 | 0.011 |
| 1600 | 1200x1600 | 1931 | 32 s | 0.007 | 0.011 |
| 1280 | 960x1280 | 1231 | 30 s | 0.007 | 0.011 |
| 1024 | 768x1024 | 799 | 28 s | 0.011 | 0.013 |

(dense = 45 lines at ~9 pt equivalent; normal = 17 lines at ~15 pt.)

## Policy

- Downscale on our side to a long edge of **2048 px** (`SB_READER_MAX_EDGE`, default 2048), never upscale.
  Full resolution gives the same tokens and accuracy, so 2048 only saves upload size and time.
- Below 1600 px small print starts to degrade (CER roughly 3-5x on dense pages): do not go lower.
- Cleanup (rotation, flatten, deskew, shadows, contrast) runs at full resolution first, then the
  downscale is the last step.
- Alternative: raising the LM Studio context length is a config-only change if a future model
  or larger images need more room; it costs RAM/VRAM.

## Classic OCR cross-check engine: RapidOCR

Tesseract needs a separate Windows binary, so it is not pip-only. RapidOCR (PaddleOCR PP-OCRv6
models on ONNX Runtime) installs with pip on Windows, ships its models, and ran offline here:

| input | long edge | time | CER | WER |
|---|---|---|---|---|
| clean page | 2048 | 3.1 s | 0.008 | 0.023 |
| simulated photo | 2048 | 2.5 s | 0.008 | 0.023 |
| simulated photo | 1600 | 1.8 s | 0.008 | 0.023 |

Pinned: rapidocr 3.10.0, onnxruntime 1.31.0, opencv-python-headless 5.0.0.93 (works with numpy 2.5.4).
