//! Orp pipeline for GLiClass classification
 
use std::path::Path;
use orp::pipeline::{Pipeline, PreProcessor, PostProcessor};
use crate::params::Parameters;
use crate::tokenizer::Tokenizer;
use crate::util::result::Result;

/// Pipeline for GLiClass classification, to be executed using [`orp`](https://github.com/fbilhaut/orp).
pub struct ClassificationPipeline {
    tokenizer: Tokenizer,
}


impl ClassificationPipeline {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P, _params: &Parameters) -> Result<Self> {
        Ok(Self { 
            tokenizer: Tokenizer::new(tokenizer_path, None)?
        })
    }
}


impl<'a> Pipeline<'a> for ClassificationPipeline {
    type Input = super::input::text::TextInput;
    type Output = super::output::classes::Classes;
    type Context = super::input::text::Labels;
    type Parameters = Parameters;

    fn pre_processor(&self, params: &Self::Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composable::composed![
            super::input::prompt::InputToPrompt::with_params(params),
            super::input::encoded::PromptsToEncoded::new(&self.tokenizer),
            super::input::tensors::InputTensors::try_from,
            super::input::tensors::InputTensors::try_into
        ]
    }

    fn post_processor(&self, _params: &Self::Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composable::composed![
            super::output::tensors::OutputTensors::try_from,
            |tensors| super::output::classes::Classes::try_from(tensors)
        ]
    }
}