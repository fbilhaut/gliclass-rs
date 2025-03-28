# 🏷️ `gliclass-rs`: GLiClass inferences in Rust

## 💬 Introduction

`gliclass-rs` is an inference engine for [GLiClass](https://github.com/Knowledgator/GLiClass) language models. 

These models are efficient for **zero-shot topic classification** or derivatives like sentiment analysis. They can also be used for efficient re-ranking.

"GLiClass" stands for "Generalist and Lightweight Model for Sequence Classification", after an original work from [Knowledgator](https://knowledgator.com), which was itself inspired by [GLiNER](https://github.com/urchade/GLiNER).

`gliclass-rs` is built in pure Rust, as an [`🧩 orp`](https://github.com/fbilhaut/orp) pipeline.


## 🎓 Examples

```toml
[dependencies]
"gliclass-rs" = "0.9.x"
```

```rust
use gliclass::{GLiClass, params::Parameters, input::text::TextInput};

let gliclass = GLiClass::new("tokenizer.json", "model.onnx", Parameters::default())?;
        
let input = TextInput::from_str(
    &[
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "Traveling is the perfect way to explore new cultures through their food.",
        "Traveling for science allows researchers to collaborate with experts worldwide.",
    ],
    &["computing", "science", "programming", "travel", "food", "politics"]
);    

let classes = gliclass.inference(input)?;

for i in 0..classes.len() {
    println!("Text {i}: {}", classes.best_label(i, None).unwrap());        
}
```

Please refer the the source code in `examples` for complete examples.


## 🧬 Models

Currently `gliclass-rs` has been tested with the following models:

|Model                      |Download                                                                    |prompt_first|
|---------------------------|----------------------------------------------------------------------------|------------|
|`gliclass-small-1.0`       |[HF Hub](https://huggingface.co/knowledgator/gliclass-small-v1.0)           |`false`     |
|`gliclass-large-1.0`       |[HF Hub](https://huggingface.co/knowledgator/gliclass-large-v1.0)           |`false`     |
|`gliclass-modern-base-v2.0`|[HF Hub](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0-init)|`true`      |

It should work out-of-the-box with other equivalent models, please report your own experiments. 

⚠️ Take care of setting the `prompt_first` parameter according the selected model's expectations (the appropriate value is indicated in the `config.json` that goes with the model).


## 👉 Related

This project follows the same principles as the ones below. Refer to their documentation for more details:

* 🌿 [gline-rs](https://github.com/fbilhaut/gline-rs): inference engine for GLiNER models
* 🧲 [gte-rs](https://github.com/fbilhaut/gte-rs): general text embedding and re-ranking
