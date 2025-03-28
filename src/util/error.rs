use std::{error::Error, fmt::Display};

/// Input-related error
#[derive(Debug, Clone)]
pub struct InputError {
    message: String,
}

impl InputError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn into_err<T>(self) -> Result<T, Box<dyn Error + Send + Sync>> {
        Err(self.boxed())
    }

    pub fn boxed(self) -> Box<dyn Error + Send + Sync> {
        Box::new(self)
    }
}

impl Error for InputError {}

impl Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}