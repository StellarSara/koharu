use hf_hub::api::sync::Api;
use image::{DynamicImage, GenericImageView};
use ndarray::{s, Array, Ix3};
use ort::{inputs, session::Session, value::TensorRef};

#[derive(Debug)]
pub struct PaddleOCR {
    det_model: Session,
    rec_model: Session,
    char_dict: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TextBox {
    pub points: Vec<(f32, f32)>,
    pub text: String,
    pub confidence: f32,
}

impl PaddleOCR {
    pub fn new() -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model("mayocream/paddleocr-onnx".to_string());
        let det_model_path = repo.get("ch_PP-OCRv4_det_infer.onnx")?;
        let rec_model_path = repo.get("ch_PP-OCRv4_rec_infer.onnx")?;
        let char_dict_path = repo.get("ppocr_keys_v1.txt")?;

        let det_model = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(det_model_path)?;

        let rec_model = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(rec_model_path)?;

        let char_dict = std::fs::read_to_string(char_dict_path)
            .map_err(|e| anyhow::anyhow!("Failed to read char dict file: {e}"))?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        Ok(Self {
            det_model,
            rec_model,
            char_dict,
        })
    }

    /// Run text detection on the entire image
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<Vec<(f32, f32)>>> {
        // Preprocess image for detection
        let (tensor, ratio_h, ratio_w) = self.preprocess_detection(image)?;

        // Run detection model
        let inputs = inputs! {
            "x" => TensorRef::from_array_view(tensor.view())?,
        };
        let outputs = self.det_model.run(inputs)?;
        let det_out = outputs[0].try_extract_array::<f32>()?;

        // Convert to 4D view
        let det_out_4d = det_out.view().into_dimensionality::<ndarray::Ix4>()?;

        // Post-process detection output to get bounding boxes
        let boxes = Self::postprocess_detection(&det_out_4d, ratio_h, ratio_w)?;

        Ok(boxes)
    }

    /// Run text recognition on a cropped image region
    pub fn recognize(&mut self, image: &DynamicImage) -> anyhow::Result<(String, f32)> {
        // Preprocess image for recognition
        let tensor = self.preprocess_recognition(image)?;

        // Run recognition model
        let inputs = inputs! {
            "x" => TensorRef::from_array_view(tensor.view())?,
        };
        let outputs = self.rec_model.run(inputs)?;
        let rec_out = outputs[0].try_extract_array::<f32>()?;

        // Convert to 3D view
        let rec_out_3d = rec_out.view().into_dimensionality::<ndarray::Ix3>()?;

        // Post-process recognition output to get text
        let (text, confidence) = Self::postprocess_recognition(&self.char_dict, &rec_out_3d)?;

        Ok((text, confidence))
    }

    /// Full pipeline: detect text regions and recognize text in each region
    pub fn inference(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<TextBox>> {
        // Detect text regions
        let boxes = self.detect(image)?;

        // Recognize text in each region
        let mut results = Vec::new();
        for box_points in boxes {
            // Get bounding rectangle for the box
            let (x_min, y_min, x_max, y_max) = self.get_bounding_rect(&box_points);
            
            // Crop the region
            let width = (x_max - x_min).max(1.0) as u32;
            let height = (y_max - y_min).max(1.0) as u32;
            let x = x_min.max(0.0) as u32;
            let y = y_min.max(0.0) as u32;
            
            let cropped = image.crop_imm(x, y, width, height);
            
            // Recognize text
            match self.recognize(&cropped) {
                Ok((text, confidence)) => {
                    results.push(TextBox {
                        points: box_points,
                        text,
                        confidence,
                    });
                }
                Err(e) => {
                    eprintln!("Recognition failed for box: {:?}", e);
                    continue;
                }
            }
        }

        Ok(results)
    }

    fn preprocess_detection(&self, image: &DynamicImage) -> anyhow::Result<(Array<f32, Ix3>, f32, f32)> {
        let (orig_width, orig_height) = image.dimensions();
        
        // Resize to multiple of 32
        let max_side_len = 960;
        let h = orig_height as f32;
        let w = orig_width as f32;
        let ratio = (max_side_len as f32) / h.max(w);
        let resize_h = (h * ratio).round() as u32;
        let resize_w = (w * ratio).round() as u32;
        
        // Round to multiple of 32
        let resize_h = ((resize_h as f32 / 32.0).round() * 32.0) as u32;
        let resize_w = ((resize_w as f32 / 32.0).round() * 32.0) as u32;
        
        let resized = image.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);
        let rgb_image = resized.to_rgb8();
        
        // Normalize: (img / 255.0 - mean) / std
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        
        let mut tensor = Array::zeros((3, resize_h as usize, resize_w as usize));
        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            tensor[[0, y, x]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            tensor[[1, y, x]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            tensor[[2, y, x]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }
        
        let ratio_h = resize_h as f32 / orig_height as f32;
        let ratio_w = resize_w as f32 / orig_width as f32;
        
        Ok((tensor, ratio_h, ratio_w))
    }

    fn postprocess_detection(output: &ndarray::ArrayView4<f32>, ratio_h: f32, ratio_w: f32) -> anyhow::Result<Vec<Vec<(f32, f32)>>> {
        // Simple thresholding and contour detection
        // This is a simplified version - PaddleOCR uses more sophisticated post-processing
        let threshold = 0.3;
        let mut boxes = Vec::new();
        
        let output_slice = output.slice(s![0, 0, .., ..]);
        let (height, width) = (output_slice.shape()[0], output_slice.shape()[1]);
        
        // Create binary mask
        let mut mask = vec![vec![0u8; width]; height];
        for y in 0..height {
            for x in 0..width {
                if output_slice[[y, x]] > threshold {
                    mask[y][x] = 255;
                }
            }
        }
        
        // Find contours (simplified - looking for rectangular regions)
        // In production, you'd want to use a proper contour finding algorithm
        let min_area = 9.0;
        for y in 0..height.saturating_sub(10) {
            for x in 0..width.saturating_sub(10) {
                if mask[y][x] > 0 {
                    // Try to find a rectangular region
                    let mut x_max = x;
                    let mut y_max = y;
                    
                    // Expand to find region bounds
                    while x_max < width && mask[y][x_max] > 0 {
                        x_max += 1;
                    }
                    while y_max < height && mask[y_max][x] > 0 {
                        y_max += 1;
                    }
                    
                    let box_width = (x_max - x) as f32;
                    let box_height = (y_max - y) as f32;
                    
                    if box_width * box_height > min_area {
                        // Convert back to original image coordinates
                        let points = vec![
                            (x as f32 / ratio_w, y as f32 / ratio_h),
                            (x_max as f32 / ratio_w, y as f32 / ratio_h),
                            (x_max as f32 / ratio_w, y_max as f32 / ratio_h),
                            (x as f32 / ratio_w, y_max as f32 / ratio_h),
                        ];
                        boxes.push(points);
                        
                        // Clear this region to avoid duplicate detections
                        for cy in y..y_max.min(height) {
                            for cx in x..x_max.min(width) {
                                mask[cy][cx] = 0;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(boxes)
    }

    fn preprocess_recognition(&self, image: &DynamicImage) -> anyhow::Result<Array<f32, Ix3>> {
        // PaddleOCR recognition expects height=48, dynamic width
        let img_h = 48;
        let (orig_w, orig_h) = image.dimensions();
        let ratio = img_h as f32 / orig_h as f32;
        let img_w = ((orig_w as f32 * ratio).round() as u32).max(1);
        
        let resized = image.resize_exact(img_w, img_h, image::imageops::FilterType::Lanczos3);
        let rgb_image = resized.to_rgb8();
        
        // Normalize
        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];
        
        let mut tensor = Array::zeros((3, img_h as usize, img_w as usize));
        for (x, y, pixel) in rgb_image.enumerate_pixels() {
            let x = x as usize;
            let y = y as usize;
            tensor[[0, y, x]] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            tensor[[1, y, x]] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            tensor[[2, y, x]] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }
        
        Ok(tensor)
    }

    fn postprocess_recognition(char_dict: &[String], output: &ndarray::ArrayView3<f32>) -> anyhow::Result<(String, f32)> {
        // output shape: [1, seq_len, num_classes]
        let seq_len = output.shape()[1];
        
        let mut text = String::new();
        let mut confidences = Vec::new();
        let mut last_index = 0;
        
        for t in 0..seq_len {
            let probs = output.slice(s![0, t, ..]);
            let (max_index, max_prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));
            
            // CTC decoding: skip blanks (index 0) and repeated characters
            if max_index > 0 && max_index != last_index {
                if let Some(char_str) = char_dict.get(max_index - 1) {
                    text.push_str(char_str);
                    confidences.push(*max_prob);
                }
            }
            last_index = max_index;
        }
        
        let avg_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };
        
        Ok((text, avg_confidence))
    }

    fn get_bounding_rect(&self, points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
        let x_coords: Vec<f32> = points.iter().map(|p| p.0).collect();
        let y_coords: Vec<f32> = points.iter().map(|p| p.1).collect();
        
        let x_min = x_coords.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = x_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_min = y_coords.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = y_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        (x_min, y_min, x_max, y_max)
    }
}
