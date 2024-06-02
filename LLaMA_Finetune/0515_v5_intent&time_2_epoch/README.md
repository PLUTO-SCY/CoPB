---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /data2/shaochenyang/Workspace/llama_Fac/Meta-Llama-3-8B-Instruct
model-index:
- name: 0515_v5_intent&time_2_epoch
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 0515_v5_intent&time_2_epoch

This model is a fine-tuned version of [/data2/shaochenyang/Workspace/llama_Fac/Meta-Llama-3-8B-Instruct](https://huggingface.co//data2/shaochenyang/Workspace/llama_Fac/Meta-Llama-3-8B-Instruct) on the scy_TPB dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8915

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.10.0
- Transformers 4.40.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1