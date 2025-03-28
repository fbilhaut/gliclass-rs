use composable::Composable;
use crate::tokenizer::Tokenizer;
use super::{prompt::PromptInput, text::Labels};
use ndarray::Array2;

/// Encoded sequences
pub struct EncodedInput {
    pub labels: Labels,    
    pub input_ids: Array2<i64>,
    pub attention_masks: Array2<i64>,    
}


pub struct PromptsToEncoded<'a> {
    tokenizer: &'a Tokenizer,
}

impl<'a> PromptsToEncoded<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self { tokenizer }
    }
}

/// Transformation from prompts to encoded sequences
impl Composable<PromptInput, EncodedInput> for PromptsToEncoded<'_> {
    fn apply(&self, input: PromptInput) -> Result<EncodedInput, Box<dyn std::error::Error + Send + Sync>> {
        let (input_ids, attention_masks) = self.tokenizer.tokenize(input.prompts)?;
        Ok(EncodedInput {
            labels: input.labels,
            input_ids,
            attention_masks,
        })
    }
}
