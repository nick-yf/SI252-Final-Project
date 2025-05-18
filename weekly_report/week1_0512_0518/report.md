# Weekly Report (May 12, 2025 – May 18, 2025)

## 1. Summary of Work
1. Review the code base of EvoPlay, understand the algorithmic framework, including the MCTS simulation details and the loss function used for training the policy-value network.
2. Finish the inference experiment of EvoPlay on the PAB1 proteins, and try to analyze the results.
3. Dive deep into the ESM3 paper (Thomas Hayes et al. ,Simulating 500 million years of evolution with a language model.Science387,850-858(2025).DOI:10.1126/science.ads0018) and understand the algorithmic framework, focus on the ESM3 token space and the tokenization process.

---
## 2. Detailed Work

### 2.1 Codebase Review of EvoPlay
We reviewed the EvoPlay codebase to understand its core components.
Special attention was given to the Monte Carlo Tree Search (MCTS) simulation, including the selection, expansion, rollout, and backpropagation steps.
We examined how the MCTS integrates with the policy-value network.
Additionally, We studied the loss function used for training the network, which combines cross-entropy loss for the policy head and mean squared error (MSE) for the value head.
This dual-headed architecture is tailored to guide evolutionary exploration in sequence design.

### 2.2 Inference Experiment on PAB1 Proteins
We completed the inference experiment using EvoPlay on the PAB1 protein family.
The model was tasked with proposing evolutionary variants, and the resulting sequences were analyzed in terms of predicted fitness according to exisiting orcale models.
We used metrics containing sequence plausibility according to a orcale model.

### 2.3 Reading and Analysis of the ESM3 Paper
We conducted an in-depth review of the ESM3 paper, which the PPT can be found here: [ESM3_Journal_Club.pptx](<./ESM3-Review.pdf>).
We focused on understanding the transformer architecture adapted for protein evolution simulation and the token space design.
Unlike traditional amino acid tokenization, ESM3 employs a context-sensitive vocabulary that captures the structural context of the amino acids.
We paid particular attention to the tokenization pipeline, including how the VQ-VAE was trained to generate a vocabulary of structural tokens.
The paper's methods for evolutionary trajectory sampling and the use of structure-aware conditioning were particularly insightful and may inform future hybridization between EvoPlay and ESM3-style modeling.

### 2.4 Some thoughts about the structure tokens and the amino acid sequence
The ESM3 paper introduces a novel tokenization approach that captures the structural context of the protein.
Within the structural encoder and the decoder of ESM3, it used a GNN-based VQ-VAE to generate a vocabulary of structural tokens, which means that each structural token is a vector or token representation of a local structure.
We concern that such structural tokens may not be directly comparable to the amino acid sequence tokens used in EvoPlay.
However, there also exists contextual information within the amino acid sequence, such as the co-evolutionary information between amino acids and within the Multi-Sequence Alignment (MSA).
So, the contextual structural tokens can also be used to under the EvoPlay framework.
