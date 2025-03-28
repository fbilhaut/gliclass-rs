//! Complete example for zero-shot text classification.
//! Reproduces the example from <https://github.com/Knowledgator/GLiClass>, and
//! checks the results against the numbers obtained with the original implementation.

use gliclass::{GLiClass, params::Parameters, input::text::TextInput};

fn main() -> gliclass::util::result::Result<()> {    
    const TOKENIZER_PATH: &str = "models/gliclass-small-v1.0/tokenizer.json";
    const MODEL_PATH: &str = "models/gliclass-small-v1.0/onnx/model.onnx";    

    let gliclass = GLiClass::new(TOKENIZER_PATH, MODEL_PATH, Parameters::default())?;
            
    let input = TextInput::from_str(
        &["One day I will see the world!"],
        &["travel", "dreams", "sport", "science", "politics"],
    ); 

    let classes = gliclass.inference(input)?;
    println!("Scores: {:?}", classes.scores);

    // check the results against the results obtained with the original implementation
    assert!(gliclass::util::test::is_close_to_a(
        &classes.scores.slice(ndarray::s![0, ..]), 
        &[0.9999985694885254, 0.9999986886978149, 0.9999117851257324, 0.9996119141578674, 0.8172177672386169],
        0.00001,
    ));

    Ok(())
}


