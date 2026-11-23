---

base_model: google/gemma-4-12B-it
tags:

- speculative-decoding
- dflash2
- legospec

---

# LEGOSpec DFlash2 for Gemma 4 12B IT

This is the DFlash2 draft checkpoint for `google/gemma-4-12B-it`, trained by LEGOSpec:  A Composable and Efficient Framework for Speculative Draft Design and Training. .

`model_type: qwen3` 

Using Qwen3 backbone.

## Training Data

900K samples from [OpenPerfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend), 300K samples from [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) and 200K samples from [WildChat](https://huggingface.co/datasets/allenai/WildChat). We use the target model to regenerate the answer at `T=1` .

## Results

Mean acceptance length τ at T = 1 for Gemma-4-12B-it with this DFlash2 draft:


| Alpaca | MT-Bench | GSM8K | MATH-500 | MBPP | LCB  |
| ------ | -------- | ----- | -------- | ---- | ---- |
| 3.72   | 4.05     | 6.45  | 6.01     | 5.28 | 4.65 |


