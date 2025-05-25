# Weekly Report (May 19, 2025 - May 25, 2025)

## 1. Summary of Work

1. We evaluated the previously proposed method in our proposal.
2. We replaced EvoPlay’s policy-value network with more advanced pretrained models, such as ESM2 and DPLM.
3. We reviewed subsequent developments building upon AlphaZero, particularly MuZero.

## 2 Detailed Work

### 2.1 Evaluation of the Previous Method

In the previous weeks, we proposed utilizing ESM-3's structural tokenizer to redefine the design space in protein representation, from 20 amino acid tokens to 4096 structure tokens.
This aimed to enrich the model’s capacity to capture structural nuances.
However, upon implementation, we discovered that the enlarged token space significantly impact on convergence due to its complexity and search inefficiency.
Consequently, we explored alternative approaches.

### 2.2 Replace EvoPlay's Policy-value Network

The original EvoPlay framework employs a 1D convolutional neural network (CNN) for its policy-value module.
However, numerous recent studies have demonstrated that pretrained protein language models yield superior performance on various downstream tasks, particularly in protein-related prediction and generation.

To leverage these advancements, we integrated **ESM2** and **DPLM** into EvoPlay:

- **ESM2** (Evolutionary Scale Modeling 2) is a family of transformer-based protein language models developed by Meta AI.
  Trained on massive protein sequence datasets, ESM2 captures contextual information and outputs high-quality embeddings that represent protein structure and function.
- **DPLM** (Diffusion Protein Language Model) is a recently proposed model that integrates the strengths of generative diffusion modeling with protein language modeling.
  It excels in generating realistic and diverse protein sequences while preserving functional properties.

These replacements are expected to enhance EvoPlay’s performance in exploring protein design spaces.

### 2.3 Search AlphaZero's Successive Work

AlphaZero, introduced by DeepMind in 2018, is a landmark reinforcement learning algorithm that achieved superhuman performance in games like Go.
One notable advancement of AlphaZero is MuZero, also developed by DeepMind.
Unlike AlphaZero, which requires a known environment model, MuZero learns to model the environment dynamics implicitly.
It integrates planning and representation learning, enabling it to achieve strong performance in environments where rules are unknown or too complex to model explicitly.
