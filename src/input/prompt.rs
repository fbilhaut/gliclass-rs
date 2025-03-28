use composable::Composable;
use crate::{util::error::InputError, params::Parameters};

use super::text::{Labels, TextInput};

const LABEL_PREFIX: &str = "<<LABEL>>";
const PROMPT_SEPARATOR: &str = "<<SEP>>";

/// Prompts build from input texts and labels
pub struct PromptInput {
    pub prompts: Vec<String>,
    pub labels: Labels,
}

pub struct InputToPrompt {
    prompt_first: bool,
}

/// Transformation from text input to prompts.
///
/// Prompt format: `[sequence]<<LABEL>>label1<<LABEL>label2...<<SEP>>[sequence]`. 
/// The actual text comes before or after, depending on the `prompt_first` parameter.
impl InputToPrompt {
    pub fn new(prompt_first: bool) -> Self {
        Self { prompt_first }
    }

    pub fn with_params(params: &Parameters) -> Self {
        Self::new(params.prompt_first())
    }

    fn make_prompt(labels: &Vec<String>) -> String {
        let mut result = String::new();
        for label in labels {
            result.push_str(LABEL_PREFIX);
            result.push_str(&label.to_lowercase());
        }
        result.push_str(PROMPT_SEPARATOR);
        result
    }

    fn get_labels(labels: &Labels, index: usize) -> Result<&Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        labels.get(index).ok_or_else(|| InputError::new("per-text labels must be aligned with texts").boxed())
    }
}

impl Default for InputToPrompt {
    fn default() -> Self {
        Self { prompt_first: false }
    }
}

/// Transformation from input text to prompts
impl Composable<TextInput, PromptInput> for InputToPrompt {
    fn apply(&self, input: TextInput) -> composable::Result<PromptInput> {
        let mut prompts = Vec::with_capacity(input.texts.len());
        for (index, text) in input.texts.into_iter().enumerate() {
            let labels = Self::get_labels(&input.labels, index)?;
            let mut prompt = Self::make_prompt(labels);
            if self.prompt_first {
                prompt.push_str(&text);
                prompts.push(prompt);
            }
            else {
                let mut text = text;
                text.push_str(&prompt);
                prompts.push(text);                
            }
        }
        Ok(PromptInput { 
            prompts, 
            labels: input.labels 
        })
    }
}