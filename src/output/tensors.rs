use ort::session::SessionOutputs;
use super::super::input::text::Labels;


/// Output tensors, right from the model
pub struct OutputTensors<'a> {
    pub outputs: SessionOutputs<'a, 'a>,
    pub labels: Labels,
}

impl<'a> TryFrom<(SessionOutputs<'a, 'a>, Labels)> for OutputTensors<'a> {
    type Error = crate::util::result::Error;

    fn try_from(value: (SessionOutputs<'a, 'a>, Labels)) -> Result<Self, Self::Error> {
        Ok(OutputTensors { outputs: value.0, labels: value.1 })
    }
}
