"""
Export PaddleOCR models to ONNX format.

This script exports PaddleOCR's detection and recognition models to ONNX format
for use with the paddle-ocr Rust crate.

Requirements:
    pip install paddleocr paddle2onnx onnx
"""

import os
import paddle
from paddleocr import PaddleOCR
import paddle2onnx


def export_detection_model(save_dir="./models/paddleocr"):
    """Export PaddleOCR detection model to ONNX."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize PaddleOCR (this will download models if needed)
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='ch',
        det_model_dir=None,  # Will use default PP-OCRv4
        rec_model_dir=None,
        use_gpu=False,
        show_log=False
    )
    
    # Get the detection model path
    det_model_path = ocr.text_detector.det_model_dir
    print(f"Detection model path: {det_model_path}")
    
    # Convert to ONNX
    model_file = os.path.join(det_model_path, "inference.pdmodel")
    params_file = os.path.join(det_model_path, "inference.pdiparams")
    
    onnx_model = paddle2onnx.command.c_paddle_to_onnx(
        model_file=model_file,
        params_file=params_file,
        opset_version=13,
        enable_onnx_checker=True,
        enable_auto_update_opset=True
    )
    
    output_path = os.path.join(save_dir, "ch_PP-OCRv4_det_infer.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model)
    
    print(f"✓ Detection model exported to: {output_path}")


def export_recognition_model(save_dir="./models/paddleocr"):
    """Export PaddleOCR recognition model to ONNX."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='ch',
        det_model_dir=None,
        rec_model_dir=None,
        use_gpu=False,
        show_log=False
    )
    
    # Get the recognition model path
    rec_model_path = ocr.text_recognizer.rec_model_dir
    print(f"Recognition model path: {rec_model_path}")
    
    # Convert to ONNX
    model_file = os.path.join(rec_model_path, "inference.pdmodel")
    params_file = os.path.join(rec_model_path, "inference.pdiparams")
    
    onnx_model = paddle2onnx.command.c_paddle_to_onnx(
        model_file=model_file,
        params_file=params_file,
        opset_version=13,
        enable_onnx_checker=True,
        enable_auto_update_opset=True
    )
    
    output_path = os.path.join(save_dir, "ch_PP-OCRv4_rec_infer.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model)
    
    print(f"✓ Recognition model exported to: {output_path}")


def export_char_dict(save_dir="./models/paddleocr"):
    """Export character dictionary."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize PaddleOCR to get the character dict
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='ch',
        use_gpu=False,
        show_log=False
    )
    
    # Get character list from the recognizer
    char_dict = ocr.text_recognizer.character
    
    output_path = os.path.join(save_dir, "ppocr_keys_v1.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for char in char_dict:
            f.write(char + "\n")
    
    print(f"✓ Character dictionary exported to: {output_path}")
    print(f"  Total characters: {len(char_dict)}")


def verify_onnx_models(save_dir="./models/paddleocr"):
    """Verify the exported ONNX models."""
    import onnx
    
    print("\nVerifying exported models...")
    
    det_path = os.path.join(save_dir, "ch_PP-OCRv4_det_infer.onnx")
    rec_path = os.path.join(save_dir, "ch_PP-OCRv4_rec_infer.onnx")
    
    try:
        det_model = onnx.load(det_path)
        onnx.checker.check_model(det_model)
        print(f"✓ Detection model is valid")
        print(f"  Inputs: {[input.name for input in det_model.graph.input]}")
        print(f"  Outputs: {[output.name for output in det_model.graph.output]}")
    except Exception as e:
        print(f"✗ Detection model validation failed: {e}")
    
    try:
        rec_model = onnx.load(rec_path)
        onnx.checker.check_model(rec_model)
        print(f"✓ Recognition model is valid")
        print(f"  Inputs: {[input.name for input in rec_model.graph.input]}")
        print(f"  Outputs: {[output.name for output in rec_model.graph.output]}")
    except Exception as e:
        print(f"✗ Recognition model validation failed: {e}")


def main():
    """Main export function."""
    print("Exporting PaddleOCR models to ONNX format...\n")
    
    save_dir = "./models/paddleocr"
    
    # Export models
    export_detection_model(save_dir)
    export_recognition_model(save_dir)
    export_char_dict(save_dir)
    
    # Verify
    verify_onnx_models(save_dir)
    
    print("\n" + "="*60)
    print("Export complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Upload the models to Hugging Face:")
    print(f"   - {save_dir}/ch_PP-OCRv4_det_infer.onnx")
    print(f"   - {save_dir}/ch_PP-OCRv4_rec_infer.onnx")
    print(f"   - {save_dir}/ppocr_keys_v1.txt")
    print(f"2. Update the model repository name in paddle-ocr/src/lib.rs")


if __name__ == "__main__":
    main()
