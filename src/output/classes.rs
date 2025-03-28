use std::collections::{BTreeMap, HashMap};
use ordered_float::OrderedFloat;

use crate::{input::text::Labels, util::result::Result};
use super::tensors::OutputTensors;

const TENSOR_LOGITS: &str = "logits";

/// End-result of the classification inference pipeline.
pub struct Classes {
    pub scores: ndarray::Array2<f32>,
    pub labels: Labels,
}


impl Classes {
    pub fn try_from(tensors: OutputTensors) -> Result<Self> {        
        let logits = tensors.outputs.get(TENSOR_LOGITS).ok_or_else(|| format!("expected tensor not found in model output: {TENSOR_LOGITS}"))?;
        let scores = logits.try_extract_tensor::<f32>()?;
        let scores = scores.into_dimensionality::<ndarray::Ix2>()?;
        let scores = crate::util::math::sigmoid_a(&scores);
        Ok(Self { scores, labels: tensors.labels })
    }

    /// Returns the scores for each label, for the given text index, in their original order
    pub fn scores(&self, index: usize) -> Option<Vec<f32>> {
        if index < self.scores.nrows() { Some(self.scores.row(index).to_vec()) } else { None }
    }

    /// Returns the scores by label, for the given text index (with an optional threshold)
    pub fn labeled_scores(&self, index: usize, threshold: Option<f32>) -> Option<HashMap<&String, f32>> {
        Some(self.labels.get(index)?
            .into_iter()
            .zip(self.scores(index)?)
            .filter(|(_, score)| *score >= threshold.unwrap_or(0.0))            
            .collect()
        )
    }

    /// Returns the ordered scores by label, for the given text index (with an optional threshold)
    pub fn ordered_scores(&self, index: usize, threshold: Option<f32>) -> Option<BTreeMap<OrderedFloat<f32>, &String>> {
        Some(self.scores(index)?
            .into_iter()
            .zip(self.labels.get(index)?)
            .filter(|(score, _)| *score >= threshold.unwrap_or(0.0))
            .map(|(score, label)| (OrderedFloat::from(score), label))
            .collect()
        )
    }

    /// Returns the best label for the given text index
    pub fn best_label(&self, text_index: usize, threshold: Option<f32>) -> Option<&str> {
        let label_index = self
            .scores(text_index)?
            .into_iter().enumerate()
            .filter(|(_, score)| *score >= threshold.unwrap_or(0.0))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)?;
        self.labels
            .get(text_index)?
            .get(label_index)
            .map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.scores.nrows()
    }

}
