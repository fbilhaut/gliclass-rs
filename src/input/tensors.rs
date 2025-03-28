use ort::session::SessionInputs;
use crate::util::result::Result;
use super::{encoded::EncodedInput, text::Labels};


const TENSOR_INPUT_IDS: &str = "input_ids";
const TENSOR_ATTN_MASKS: &str = "attention_mask";


/// Input tensors, ready for inferences
pub struct InputTensors<'a> {
    pub inputs: SessionInputs<'a, 'a>,
    pub labels: Labels,
}


impl TryFrom<EncodedInput> for InputTensors<'_> {
    type Error = crate::util::result::Error;

    fn try_from(input: EncodedInput) -> Result<Self> {
        Ok(Self {
            labels: input.labels,
            inputs: ort::inputs!{
                TENSOR_INPUT_IDS => input.input_ids,
                TENSOR_ATTN_MASKS => input.attention_masks,    
            }?.into(),            
        })
    }
}


impl<'a> TryInto<(SessionInputs<'a, 'a>, Labels)> for InputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_into(self) -> Result<(SessionInputs<'a, 'a>, Labels)> {
        Ok((self.inputs, self.labels))
    }    
}
