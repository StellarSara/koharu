release: 0.7.3

## 🎉 New Features

### PaddleOCR Support for Chinese Text
Added full PaddleOCR integration for Chinese text detection and recognition, alongside the existing Manga OCR for Japanese text.

**Key Features:**
- **OCR Engine Selector**: Choose between "Manga OCR (Japanese)" and "PaddleOCR (Chinese)" in the Detection panel
- **Three OCR Workflows**:
  - Japanese manga: Use Manga OCR (original workflow)
  - Chinese text (hybrid): Use comic-text-detector + PaddleOCR recognition
  - Chinese text (full pipeline): Use PaddleOCR's native detection + recognition in one step
- **New "Detect+OCR" Button**: One-click detection and recognition for Chinese text using PaddleOCR
- **Smart OCR Selection**: The OCR button automatically uses the selected engine

### Technical Implementation
- New `paddle-ocr` workspace crate with ONNX model support
- Integrated PaddleOCR detection and recognition models (PP-OCRv4)
- Added 3 new Tauri commands: `ocr_paddle`, `detect_and_ocr_paddle`
- CUDA acceleration support for PaddleOCR models
- Character dictionary with thousands of Chinese characters

### UI Improvements
- OCR engine dropdown selector in Detection panel
- Conditional "Detect+OCR" button that appears when PaddleOCR is selected
- Updated tooltips and labels for better clarity
- Improved state management for OCR engine selection

## 📚 Documentation
- Added comprehensive integration guide (`PADDLEOCR_INTEGRATION.md`)
- Added user guide (`HOW_PADDLEOCR_WORKS.md`)
- Updated README with PaddleOCR features and models
- Added Python script for exporting PaddleOCR models to ONNX format

## 🔧 Changes
- Updated workspace dependencies to include `paddle-ocr` crate
- Enhanced detection panel with engine selection
- Improved OCR workflow flexibility

## 📦 Models
PaddleOCR models need to be exported and uploaded to Hugging Face:
- `ch_PP-OCRv4_det_infer.onnx` - Detection model
- `ch_PP-OCRv4_rec_infer.onnx` - Recognition model  
- `ppocr_keys_v1.txt` - Character dictionary

See `scripts/export_paddleocr_to_onnx.py` for model export instructions.

## 🌏 Language Support
- 🇯🇵 Japanese: Manga OCR (optimized for manga/vertical text)
- 🇨🇳 Chinese: PaddleOCR (simplified & traditional Chinese)

## 🚀 Usage
1. Open a manga/comic image
2. Select your desired OCR engine from the dropdown:
   - "Manga OCR (Japanese)" for Japanese text
   - "PaddleOCR (Chinese)" for Chinese text
3. Use the appropriate workflow:
   - For Japanese: Click "Detect" → "OCR"
   - For Chinese (hybrid): Click "Detect" → "OCR"
   - For Chinese (full): Click "Detect+OCR"
4. View recognized text in the Text Blocks section

---

**Full Changelog**: https://github.com/StellarSara/koharu/compare/0.7.2...0.7.3
