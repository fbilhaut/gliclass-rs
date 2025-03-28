//! Wrapper around HuggingFace tokenizers
use std::path::Path;
use crate::util::result::Result;

/// Wrapper around HuggingFace tokenizers
pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Tokenizer {

    pub fn new<P: AsRef<Path>>(tokenizer_path: P, max_length: Option<usize>) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)?;
        
        if let Some(length) = max_length {
            let mut truncation = tokenizers::TruncationParams::default();
            truncation.max_length = length;
            tokenizer.with_truncation(Some(truncation))?;
        }

        let mut padding = tokenizers::PaddingParams::default();
        padding.strategy = tokenizers::PaddingStrategy::BatchLongest;    
        
        tokenizer.with_padding(Some(padding));
        
        Ok(Self { tokenizer })
    }

    pub fn tokenize<'s, E: Into<tokenizers::EncodeInput<'s>> + Send>(&self, input: Vec<E>) -> Result<(ndarray::Array2<i64>, ndarray::Array2<i64>)> {
        let encodings = self.tokenizer.encode_batch(input, true)?;
        let max_tokens = encodings.first().map(|x| x.len()).unwrap_or(0);
        let mut input_ids = ndarray::Array2::zeros((0, max_tokens));
        let mut attn_masks = ndarray::Array2::zeros((0, max_tokens));
        for encoding in encodings {
            input_ids.push_row(ndarray::ArrayView::from(&Self::to_i64(encoding.get_ids()).to_vec()))?;
            attn_masks.push_row(ndarray::ArrayView::from(&Self::to_i64(encoding.get_attention_mask()).to_vec()))?;
        }
        Ok((input_ids, attn_masks))
    }

    fn to_i64(array: &[u32]) -> Vec<i64> {
        array.iter().map(|x| *x as i64).collect()
    }

}