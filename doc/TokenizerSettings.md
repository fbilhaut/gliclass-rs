# Discussion with Ihor Stepanov about tokenizer settings

(see https://discord.com/channels/1109081497002066022/1109082615295787059/1355585580389498902, 2025-03-29)

Ihor:

> One small question, for ModernGLiClass, when initializing the tokenizer, we need to set add_prefix_space to True. Have you considered it in this crate?

Frédérik:

> About add_prefix_space: I didn't specifically took care of it, thank you very much for pointing that out. 
> I did a quick test and indeed it slightly changes the results (according to my test cases the numbers are a little "better", in the sense of closer to what would be expected, but the difference is subtle).
> So I am considering adding an option in gliclass-rs to force this parameter into the pre-tokenizer, but I noticed that in the HF's version of tokenizer json for gliclass-modern-large-v2.0 (for example), this parameter is set to false. Is there a reason for that ? I would seem logical to me that this parameter would be set in this file according to what you consider the best for your model, rather that adding an option in the code itself. Or did I miss something ?
> (BTW it seems that this parameter is actually set to true for the ByteLevel pre-tokenizer, unless something is specified in the json descriptor).

Ihor:

> In our experiments, this parameter improved the  F1 score on average by 2.5% across dozens of datasets. It changes the way how input is tokenized and is really important.
> In our tokenizer.json, it was stored as the default value false. But I agree that changing it in the tokenizer file is better. 
