use clap::Parser;
use ort::execution_providers::CUDAExecutionProvider;
use paddle_ocr::PaddleOCR;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: String,

    #[arg(short, long, default_value = "false")]
    detect_only: bool,
}

fn main() -> anyhow::Result<()> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        .commit()?;

    let cli = Cli::parse();

    let mut model = PaddleOCR::new()?;
    let image = image::open(&cli.input)?;

    if cli.detect_only {
        println!("Running detection only...");
        let boxes = model.detect(&image)?;
        println!("Detected {} text regions:", boxes.len());
        for (i, box_points) in boxes.iter().enumerate() {
            println!("  Region {}: {:?}", i + 1, box_points);
        }
    } else {
        println!("Running full OCR pipeline...");
        let results = model.inference(&image)?;
        println!("OCR Results:");
        for (i, result) in results.iter().enumerate() {
            println!(
                "  {}: '{}' (confidence: {:.2})",
                i + 1,
                result.text,
                result.confidence
            );
        }
    }

    Ok(())
}
