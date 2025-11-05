use std::sync::Arc;

use anyhow::Result;
use comic_text_detector::ComicTextDetector;
use lama::Lama;
use manga_ocr::MangaOCR;
use paddle_ocr::PaddleOCR;
use tokio::sync::Mutex;

use crate::image::SerializableDynamicImage;
use crate::state::TextBlock;

#[derive(Debug, Clone)]
pub struct Model {
    detector: Arc<Mutex<ComicTextDetector>>,
    ocr: Arc<Mutex<MangaOCR>>,
    paddle_ocr: Arc<Mutex<PaddleOCR>>,
    lama: Arc<Mutex<Lama>>,
}

impl Model {
    pub fn new() -> Result<Self> {
        Ok(Self {
            detector: Arc::new(Mutex::new(ComicTextDetector::new()?)),
            ocr: Arc::new(Mutex::new(MangaOCR::new()?)),
            paddle_ocr: Arc::new(Mutex::new(PaddleOCR::new()?)),
            lama: Arc::new(Mutex::new(Lama::new()?)),
        })
    }

    pub async fn detect(
        &self,
        image: &SerializableDynamicImage,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<TextBlock>, SerializableDynamicImage)> {
        let mut detector = self.detector.lock().await;
        let result = detector.inference(image, conf_threshold, nms_threshold)?;

        let mut text_blocks: Vec<TextBlock> = result
            .bboxes
            .into_iter()
            .map(|bbox| TextBlock {
                x: bbox.xmin.round() as u32,
                y: bbox.ymin.round() as u32,
                width: (bbox.xmax - bbox.xmin).round() as u32,
                height: (bbox.ymax - bbox.ymin).round() as u32,
                confidence: bbox.confidence,
                ..Default::default()
            })
            .collect();

        text_blocks.sort_unstable_by(|a, b| {
            (a.y + a.height / 2)
                .partial_cmp(&(b.y + b.height / 2))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok((text_blocks, result.segment.into()))
    }

    pub async fn ocr(
        &self,
        image: &SerializableDynamicImage,
        blocks: &[TextBlock],
    ) -> Result<Vec<TextBlock>> {
        let mut ocr = self.ocr.lock().await;

        blocks
            .iter()
            .map(|block| {
                let crop = image.crop_imm(block.x, block.y, block.width, block.height);
                let text = ocr.inference(&crop)?;

                Ok(TextBlock {
                    text: text.into(),
                    ..block.clone()
                })
            })
            .collect()
    }

    pub async fn ocr_paddle(
        &self,
        image: &SerializableDynamicImage,
        blocks: &[TextBlock],
    ) -> Result<Vec<TextBlock>> {
        let mut paddle_ocr = self.paddle_ocr.lock().await;

        blocks
            .iter()
            .map(|block| {
                let crop = image.crop_imm(block.x, block.y, block.width, block.height);
                let (text, confidence) = paddle_ocr.recognize(&crop)?;

                Ok(TextBlock {
                    text: text.into(),
                    confidence,
                    ..block.clone()
                })
            })
            .collect()
    }

    pub async fn detect_and_ocr_paddle(
        &self,
        image: &SerializableDynamicImage,
    ) -> Result<Vec<TextBlock>> {
        let mut paddle_ocr = self.paddle_ocr.lock().await;
        
        let text_boxes = paddle_ocr.inference(image)?;
        
        let text_blocks = text_boxes
            .into_iter()
            .map(|text_box| {
                let (x_min, y_min, x_max, y_max) = Self::get_bounding_rect(&text_box.points);
                
                TextBlock {
                    x: x_min.max(0.0) as u32,
                    y: y_min.max(0.0) as u32,
                    width: (x_max - x_min).max(1.0) as u32,
                    height: (y_max - y_min).max(1.0) as u32,
                    confidence: text_box.confidence,
                    text: Some(text_box.text),
                    translation: None,
                }
            })
            .collect();
        
        Ok(text_blocks)
    }

    fn get_bounding_rect(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
        let x_coords: Vec<f32> = points.iter().map(|p| p.0).collect();
        let y_coords: Vec<f32> = points.iter().map(|p| p.1).collect();
        
        let x_min = x_coords.iter().cloned().fold(f32::INFINITY, f32::min);
        let x_max = x_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let y_min = y_coords.iter().cloned().fold(f32::INFINITY, f32::min);
        let y_max = y_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        (x_min, y_min, x_max, y_max)
    }

    pub async fn inpaint(
        &self,
        image: &SerializableDynamicImage,
        mask: &SerializableDynamicImage,
        dilate_kernel_size: u8,
        erode_distance: u8,
    ) -> Result<SerializableDynamicImage> {
        let mask = imageproc::morphology::grayscale_dilate(
            &mask.to_luma8(),
            &imageproc::morphology::Mask::square(dilate_kernel_size),
        );
        let mask = imageproc::morphology::erode(
            &mask,
            imageproc::distance_transform::Norm::L2,
            erode_distance,
        );

        let mut lama = self.lama.lock().await;
        let result = lama.inference(image, &image::DynamicImage::ImageLuma8(mask))?;

        Ok(result.into())
    }
}
