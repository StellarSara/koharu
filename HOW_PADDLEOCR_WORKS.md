# How PaddleOCR Works in Koharu - Complete Guide

## UI Overview

**Yes, there ARE UI buttons for OCR!** They're in the right panel under the "Detection" accordion section.

## How Manga OCR Currently Works

### Workflow:
1. User opens an image/manga page
2. Click **"Detect"** button → Runs `comic-text-detector` to find text regions
3. Click **"OCR"** button → Runs `manga-ocr` to recognize Japanese text in detected regions
4. Text appears in the "Text Blocks" section below

### Files Involved:
- **Backend**: `koharu/src/command.rs` - `detect()` and `ocr()` commands
- **Frontend**: `ui/components/Panels.tsx` - DetectionPanel with buttons
- **State**: `ui/lib/store.ts` - Zustand store managing state

## How PaddleOCR Now Works

### New Features Added:

#### 1. OCR Engine Selector (NEW!)
A dropdown in the Detection panel to choose:
- **Manga OCR (Japanese)** - Original, for Japanese manga
- **PaddleOCR (Chinese)** - New, for Chinese text

#### 2. Three Workflow Options:

**Option A: Japanese Manga (Original)**
```
1. Select "Manga OCR (Japanese)" from dropdown
2. Click "Detect" → comic-text-detector finds regions
3. Click "OCR" → manga-ocr recognizes Japanese text
```

**Option B: Chinese with Comic Detector**
```
1. Select "PaddleOCR (Chinese)" from dropdown
2. Click "Detect" → comic-text-detector finds regions
3. Click "OCR" → PaddleOCR recognizes Chinese text
```

**Option C: Full PaddleOCR Pipeline** ⭐ NEW BUTTON!
```
1. Select "PaddleOCR (Chinese)" from dropdown
2. Click "Detect+OCR" button → PaddleOCR does both detection AND recognition in one step
   (This button only appears when PaddleOCR is selected)
```

## UI Changes Made

### `ui/lib/store.ts`
Added:
- `ocrEngine` state: `'manga-ocr' | 'paddleocr'`
- `setOcrEngine()` action: Switch between engines
- `detectAndOcrPaddle()` action: Full PaddleOCR pipeline
- Smart `ocr()` function: Automatically uses the selected engine

### `ui/components/Panels.tsx`
Added to `DetectionPanel`:
- OCR Engine dropdown selector
- "Detect+OCR" button (shown only when PaddleOCR is selected)
- Tooltips explaining each option

## Visual Layout

```
┌─────────────────────────────────┐
│      Detection Panel            │
├─────────────────────────────────┤
│ OCR Engine:                     │
│ [Manga OCR (Japanese) ▼]       │  ← NEW DROPDOWN
│                                 │
│ Confidence threshold: [••••]   │
│ NMS threshold:       [••••]    │
│                                 │
│  [Detect]  [OCR]               │  ← Existing buttons
│                                 │
│  [Detect+OCR (Full)]           │  ← NEW (PaddleOCR only)
└─────────────────────────────────┘
```

## Backend Commands Available

All these are wired up to the UI:

| Command | Tauri Command | What It Does |
|---------|---------------|--------------|
| Detect | `detect` | Comic-text-detector finds regions |
| OCR (Manga) | `ocr` | Manga-ocr recognizes Japanese |
| OCR (Paddle) | `ocr_paddle` | PaddleOCR recognizes Chinese |
| Full Pipeline | `detect_and_ocr_paddle` | PaddleOCR detects + recognizes |

## Data Flow Example

### User clicks "Detect+OCR" with PaddleOCR selected:

```
UI Button Click
      ↓
useAppStore.detectAndOcrPaddle()
      ↓
invoke('detect_and_ocr_paddle', { index: 0 })
      ↓
Tauri Command Handler (koharu/src/command.rs)
      ↓
Model.detect_and_ocr_paddle() (koharu/src/onnx.rs)
      ↓
PaddleOCR.inference() (paddle-ocr/src/lib.rs)
      ↓
ONNX Models (detection + recognition)
      ↓
Returns Document with text_blocks filled
      ↓
UI Updates: Text blocks appear in panel
```

## Key Advantages

### Manga OCR:
- ✅ Optimized for Japanese manga
- ✅ Handles vertical text well
- ✅ Good with stylized fonts

### PaddleOCR:
- ✅ Optimized for Chinese characters
- ✅ Supports simplified & traditional Chinese
- ✅ Has its own detection (option C)
- ✅ Can work with comic-text-detector too

## Testing Without Building

You can test just the PaddleOCR crate:

```bash
# Build the paddle-ocr library
cargo build -p paddle-ocr

# Test with CLI
cargo run -p paddle-ocr -- -i test_image.png

# With CUDA
cargo run -p paddle-ocr --features cuda -- -i test_image.png
```

## What You Need to Complete

### 1. Export Models (One-time setup)
```bash
pip install paddleocr paddle2onnx onnx
python scripts/export_paddleocr_to_onnx.py
```

This creates:
- `models/paddleocr/ch_PP-OCRv4_det_infer.onnx`
- `models/paddleocr/ch_PP-OCRv4_rec_infer.onnx`
- `models/paddleocr/ppocr_keys_v1.txt`

### 2. Upload to Hugging Face
Create a repo like `mayocream/paddleocr-onnx` and upload the 3 files

### 3. Update Model Path
In `paddle-ocr/src/lib.rs`, line 18:
```rust
let repo = api.model("mayocream/paddleocr-onnx".to_string());
```

### 4. Build & Test
```bash
# Install UI dependencies
cd ui && bun install && cd ..

# Build with CUDA
bun tauri build --features cuda

# Or without CUDA
bun tauri build
```

### 5. Try It Out!
1. Launch Koharu
2. Open a Chinese manga/comic image
3. Select "PaddleOCR (Chinese)" from dropdown
4. Click "Detect+OCR" button
5. Watch the magic happen! ✨

## Troubleshooting

### "Models not found" error
- Check Hugging Face repo name is correct
- Verify models were uploaded
- Check internet connection

### Chinese text not recognized
- Make sure you selected "PaddleOCR (Chinese)" in the dropdown
- Try the "Detect+OCR" button for better results
- Check image quality and orientation

### UI dropdown not appearing
- Make sure you rebuilt after the changes
- Check browser console for errors (F12)
- Verify `bun install` ran successfully in `ui/`

## Summary

**The UI already had OCR buttons!** The existing "Detect" and "OCR" buttons in the Detection panel now:

1. Work with **both** Manga OCR and PaddleOCR (selected via dropdown)
2. PaddleOCR adds a **third button** "Detect+OCR" for full pipeline
3. Everything is fully integrated - just need to export the models!

The workflow is intuitive:
- Japanese manga? Use Manga OCR (default)
- Chinese text? Switch to PaddleOCR and click buttons
- Want one-step? Use "Detect+OCR" button
