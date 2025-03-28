//! An inference engine for [GLiClass](https://github.com/Knowledgator/GLiClass) models. 
//! 
//! These language models are efficient for zero-shot topic classification or derivatives like sentiment analysis. 
//! They can also be used for efficient re-ranking.
//! 
//! GLiClass stands for "Generalist and Lightweight Model for Sequence Classification", after an original work from 
//! [Knowledgator](https://knowledgator.com), which was itself inspired by [GLiNER](https://github.com/urchade/GLiNER).

pub mod util;
pub mod params;
pub mod tokenizer;
pub mod input;
pub mod output;
pub mod pipeline;


/// Convenience front-end for easy use with default runtime parameters (CPU).
/// For more advanced use, see examples and the `orp` crate.
pub struct GLiClass {
    params: params::Parameters,
    pipeline: pipeline::ClassificationPipeline,
    model: orp::model::Model,
}

impl GLiClass {
    /// Loads the model given a tokenizer, an ONNX model, and the required parameters
    pub fn new<P: AsRef<std::path::Path>>(tokenizer_path: P, model_path: P, params: params::Parameters) -> crate::util::result::Result<Self> {
        Ok(Self {
            pipeline: pipeline::ClassificationPipeline::new(tokenizer_path, &params)?,
            model: orp::model::Model::new(model_path, orp::params::RuntimeParameters::default())?,
            params,            
        })
    }

    /// Performs classification on the given output
    pub fn inference(&self, input: input::text::TextInput) -> crate::util::result::Result<output::classes::Classes> {
        self.model.inference(input, &self.pipeline, &self.params)
    }
}

