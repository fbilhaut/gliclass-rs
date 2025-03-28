//! Complete example for zero-shot text classification.
//! Classifies multiple sentences with their own set of labels, and prints
//! the best one for each text.

fn main() -> gliclass::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gliclass-small-v1.0/tokenizer.json";
    const MODEL_PATH: &str = "models/gliclass-small-v1.0/onnx/model.onnx";

    let params = gliclass::params::Parameters::default();
    let pipeline = gliclass::pipeline::ClassificationPipeline::new(TOKENIZER_PATH, &params)?;
    let model = orp::model::Model::new(MODEL_PATH, orp::params::RuntimeParameters::default())?;
            
    let inputs = gliclass::input::text::TextInput::from_str_per_text(
        &[
            "Rust is a systems programming language focused on safety, speed, and concurrency, with a strong ownership model that prevents memory errors without needing a garbage collector.",
            "Traveling is the perfect way to explore new cultures through their food, from savoring street tacos in Mexico to indulging in fresh sushi in Japan.",
            "Traveling for science allows researchers to explore new environments, gather crucial data, and collaborate with experts worldwide to expand our understanding of the universe.",
        ],
        &[
            &["performance", "user interface"], // expecting 'performance'
            &["gastronomy", "plane"], // expecting 'gastronomy'
            &["conferencing", "teaching"], // expecting 'conferencing'
        ]
    );    

    let classes = model.inference(inputs, &pipeline, &params)?;

    for i in 0..classes.len() {
        println!("Text {i} => {}", classes.best_label(i, None).unwrap());
    }

    Ok(())
}