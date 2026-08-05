//! Zero-shot text classification using a CUDA GPU execution provider.
//!
//! Demonstrates `GLiClass::new_with_runtime()` as a convenience alternative to
//! constructing `orp::model::Model` directly (as shown in samples 2–4).
//!
//! Run with: `cargo run --example sample5 --features cuda`
//!
//! If a CUDA-capable GPU is unavailable at runtime, `ort` falls back to CPU automatically.

use gliclass::{GLiClass, input::text::TextInput, params::Parameters};
use ort::execution_providers::CUDAExecutionProvider;

fn main() -> gliclass::util::result::Result<()> {
    const TOKENIZER_PATH: &str = "models/gliclass-small-v1.0/tokenizer.json";
    const MODEL_PATH: &str = "models/gliclass-small-v1.0/onnx/model.onnx";

    let runtime = orp::params::RuntimeParameters::default()
        .with_execution_providers([CUDAExecutionProvider::default().build()]);

    let gliclass = GLiClass::new_with_runtime(TOKENIZER_PATH, MODEL_PATH, Parameters::default(), runtime)?;

    let input = TextInput::from_str(
        &["One day I will see the world!"],
        &["travel", "dreams", "sport", "science", "politics"],
    );

    let classes = gliclass.inference(input)?;
    println!("Scores: {:?}", classes.scores);

    Ok(())
}
