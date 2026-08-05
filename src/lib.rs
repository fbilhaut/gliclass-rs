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


/// Convenience front-end for GLiClass inference.
///
/// Use [`new()`](Self::new) for CPU inference. Use [`new_with_runtime()`](Self::new_with_runtime)
/// to supply custom [`RuntimeParameters`](orp::params::RuntimeParameters) such as a CUDA
/// execution provider. For lower-level control, use `orp::model::Model` directly (see examples).
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

    /// Loads the model with custom [`RuntimeParameters`](orp::params::RuntimeParameters),
    /// allowing selection of an execution provider such as CUDA for GPU acceleration.
    ///
    /// Requires the corresponding feature flag (e.g. `--features cuda`).
    /// If the requested provider is unavailable at runtime, `ort` falls back to CPU.
    ///
    /// # Example
    /// ```no_run
    /// use ort::execution_providers::CUDAExecutionProvider;
    /// use gliclass::{GLiClass, params::Parameters};
    ///
    /// let runtime = orp::params::RuntimeParameters::default()
    ///     .with_execution_providers([CUDAExecutionProvider::default().build()]);
    /// let gliclass = GLiClass::new_with_runtime("tokenizer.json", "model.onnx", Parameters::default(), runtime)?;
    /// # Ok::<(), gliclass::util::result::Error>(())
    /// ```
    pub fn new_with_runtime<P: AsRef<std::path::Path>>(tokenizer_path: P, model_path: P, params: params::Parameters, runtime_params: orp::params::RuntimeParameters) -> crate::util::result::Result<Self> {
        Ok(Self {
            pipeline: pipeline::ClassificationPipeline::new(tokenizer_path, &params)?,
            model: orp::model::Model::new(model_path, runtime_params)?,
            params,
        })
    }

    /// Performs classification on the given output
    pub fn inference(&self, input: input::text::TextInput) -> crate::util::result::Result<output::classes::Classes> {
        self.model.inference(input, &self.pipeline, &self.params)
    }
}

