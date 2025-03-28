//! GLiClass Parameters

/// Parameters for the GLiClass pipeline.
pub struct Parameters {    
    prompt_first: bool,
}

impl Parameters {
    /// This parameter must be set according to the expectations of the loaded model.
    /// 
    /// Examples:
    /// * `gliclass-xxx-1.0` => `false`
    /// * `gliclass-modern-base-v2.0-xxx` => `true`
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

