# PaddleOCR Integration for Koharu

This document explains the PaddleOCR integration added to Koharu for Chinese text detection and recognition.

## What Was Added

### 1. New `paddle-ocr` Crate
Location: `paddle-ocr/`

A new workspace member that implements PaddleOCR functionality in Rust using ONNX models:
- **Detection**: Detects text regions in images using PaddleOCR's detection model
- **Recognition**: Recognizes Chinese text from cropped image regions
- **Full Pipeline**: Combined detection and recognition in one call

Key files:
- `paddle-ocr/src/lib.rs` - Main implementation with `PaddleOCR` struct
- `paddle-ocr/src/main.rs` - CLI tool for testing
- `paddle-ocr/Cargo.toml` - Dependencies and features

### 2. Integration with Main Application

#### Updated Files:
- `koharu/src/onnx.rs`:
  - Added `paddle_ocr` field to `Model` struct
  - Added `ocr_paddle()` method - use PaddleOCR on detected regions
  - Added `detect_and_ocr_paddle()` method - full PaddleOCR pipeline

- `koharu/src/command.rs`:
  - `ocr_paddle()` command - recognize text using PaddleOCR
  - `detect_and_ocr_paddle()` command - detect and recognize in one step

- `koharu/src/app.rs`:
  - Registered new Tauri commands

- `Cargo.toml`:
  - Added `paddle-ocr` to workspace members
  - Added to workspace dependencies

- `koharu/Cargo.toml`:
  - Added `paddle-ocr` dependency
  - Added to CUDA features

### 3. Model Export Script
Location: `scripts/export_paddleocr_to_onnx.py`

Python script to convert PaddleOCR models to ONNX format:
```bash
python scripts/export_paddleocr_to_onnx.py
```

This will:
1. Download PaddleOCR PP-OCRv4 models
2. Convert detection and recognition models to ONNX
3. Export character dictionary
4. Save to `models/paddleocr/`

### 4. Documentation
Updated `README.md`:
- Added PaddleOCR to features list
- Added to ONNX models section
- Added usage instructions

## How to Use

### Backend API

Three new Tauri commands are available:

#### 1. `ocr_paddle(index: usize)`
Uses PaddleOCR for recognition on already detected text regions:
```typescript
await invoke('ocr_paddle', { index: 0 })
```
Use this after calling `detect()` to use PaddleOCR instead of manga-ocr.

#### 2. `detect_and_ocr_paddle(index: usize)`
Full pipeline using PaddleOCR's detection and recognition:
```typescript
await invoke('detect_and_ocr_paddle', { index: 0 })
```
This is a one-step process that detects and recognizes Chinese text.

#### 3. Existing Commands Still Work
- `detect()` - comic-text-detector (works for any language)
- `ocr()` - manga-ocr (optimized for Japanese)

### Testing the CLI Tool

Test the paddle-ocr crate directly:

```bash
# Full OCR pipeline
cargo run -p paddle-ocr -- -i path/to/image.png

# Detection only
cargo run -p paddle-ocr -- -i path/to/image.png --detect-only

# With CUDA
cargo run -p paddle-ocr --features cuda -- -i path/to/image.png
```

## Model Setup

### Option 1: Use Pre-converted Models (Recommended)
Upload the exported ONNX models to Hugging Face and update the repository name in `paddle-ocr/src/lib.rs`:

```rust
let repo = api.model("YOUR_USERNAME/paddleocr-onnx".to_string());
```

### Option 2: Export Models Yourself

1. Install dependencies:
```bash
pip install paddleocr paddle2onnx onnx
```

2. Run export script:
```bash
python scripts/export_paddleocr_to_onnx.py
```

3. Upload models to Hugging Face:
   - `models/paddleocr/ch_PP-OCRv4_det_infer.onnx`
   - `models/paddleocr/ch_PP-OCRv4_rec_infer.onnx`
   - `models/paddleocr/ppocr_keys_v1.txt`

## Architecture

```
User Input (Chinese Image)
        ↓
┌───────────────────────┐
│   Tauri Command       │
│   detect_and_ocr_     │
│   paddle()            │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   koharu/src/onnx.rs  │
│   Model::detect_and_  │
│   ocr_paddle()        │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   paddle-ocr crate    │
│   PaddleOCR::         │
│   inference()         │
└───────────────────────┘
        ↓
┌───────────────────────┐
│   ONNX Runtime        │
│   - Detection Model   │
│   - Recognition Model │
└───────────────────────┘
        ↓
    Text Results
```

## Workflow Options

### For Japanese Manga:
1. `detect()` - Use comic-text-detector
2. `ocr()` - Use manga-ocr

### For Chinese Comics (Option A):
1. `detect()` - Use comic-text-detector (works for Chinese too)
2. `ocr_paddle()` - Use PaddleOCR for recognition

### For Chinese Comics (Option B):
1. `detect_and_ocr_paddle()` - Use PaddleOCR's native detection + recognition

## Next Steps

1. **Export Models**: Run the export script to generate ONNX models
2. **Upload to Hugging Face**: Create a repository for the models
3. **Update Model Path**: Change the repo name in `paddle-ocr/src/lib.rs`
4. **Test**: Build and test with Chinese images
5. **UI Integration**: Add buttons in the UI to call `ocr_paddle` and `detect_and_ocr_paddle`
6. **Optional**: Add language selection in the UI to choose between manga-ocr and PaddleOCR

## Performance Notes

- PaddleOCR detection uses dynamic input sizes (resized to multiples of 32)
- Recognition expects height=48 with dynamic width
- Both models support batch processing but currently process images one at a time
- CUDA acceleration is supported when building with `--features cuda`

## Troubleshooting

### Models Not Loading
- Check Hugging Face repository name and access
- Verify models were exported correctly
- Check file names match exactly in the code

### Poor Detection Results
- Try adjusting the confidence threshold
- PaddleOCR detection may work better with clearer, less stylized text
- For manga/comics, comic-text-detector may be more suitable

### Chinese Characters Not Recognized
- Verify the character dictionary was exported correctly
- Check that the recognition model is the Chinese version (ch_PP-OCRv4)
- Ensure input images are clear and properly oriented
