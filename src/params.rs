//! GLiClass Parameters

/// Parameters for the GLiClass pipeline.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Parameters {    
    prompt_first: bool,
}


impl Parameters {

    /// Load parameters from a `config.json` file as provided with the models
    pub fn from_json<P: AsRef<std::path::Path>>(path: P) -> crate::util::result::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let result = serde_json::from_reader(reader)?;
        Ok(result)
    }

    /// This parameter must be set according to the expectations of the loaded model.
    /// 
    /// Examples:
    /// * `gliclass-xxx-1.0` => `false`
    /// * `gliclass-modern-xxx-v2.0` => `true`
    /// * for other models see the accompanying `config.json` file
    pub fn with_prompt_first(mut self, b: bool) -> Self {
        self.prompt_first = b;
        self
    }
    
    pub fn prompt_first(&self) -> bool {
        self.prompt_first
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self { 
            prompt_first: false 
        }
    }
}

