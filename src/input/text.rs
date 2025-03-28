/// Input text and labels
pub struct TextInput {
    pub texts: Vec<String>,
    pub labels: Labels,
}

/// Labels, either a single list for all texts, or a list per text.
pub enum Labels {
    /// The same set of labels will be used for all the provided texts
    Unique(Vec<String>),
    /// Specify a set of label per input text (in such case the texts and labels arrays must have the same length).
    PerText(Vec<Vec<String>>),
}

impl Labels {
    /// Returns the list of labels for the text at the given index.
    pub fn get(&self, text_index: usize) -> Option<&Vec<String>> {
        match self {
            Labels::Unique(labels) => Some(labels),
            Labels::PerText(labels) => { labels.get(text_index) }
        }
    }
}

impl TextInput {
    /// Creates an input with the same set of labels for every text
    pub fn new(texts: Vec<String>, labels: Vec<String>) -> Self {
        Self {
            texts,
            labels: Labels::Unique(labels),
        }
    }

    /// Creates an input with the same set of labels for every text
    pub fn from_str(texts: &[&str], labels: &[&str]) -> Self {        
        Self::new(
            texts.iter().map(ToString::to_string).collect(),
            labels.iter().map(ToString::to_string).collect(),
        )
    }

    /// Creates an input with a different set of labels for each text.
    /// The texts and labels arrays must have the same length, otherwise an error will be raised at some point.
    pub fn new_per_text(texts: Vec<String>, labels: Vec<Vec<String>>) -> Self {
        Self {
            texts,
            labels: Labels::PerText(labels),
        }
    }

    /// Creates an input with a different set of labels for each text.
    /// The texts and labels arrays must have the same length, otherwise an error will be raised at some point.
    pub fn from_str_per_text(texts: &[&str], labels: &[&[&str]]) -> Self {        
        Self::new_per_text(
            texts.iter().map(ToString::to_string).collect(),
            labels.iter().map(|inner| inner.iter().map(ToString::to_string).collect()).collect(),
        )
    }
}
