# Weekly Report (May 26, 2025 - June 1, 2025)

## 1. Summary of Work
1. We explored more possibilities using ESM2.
2. We analyze the feasibility of applying MuZero to replace the original AlphaZero module in our setting.
3. We reviewed other papers about the directed evolution of proteins or sampling in protein sequence space, which may provide insights for our work.
4. We start to organize the outline and write the final report of our project.

## 2 Detailed Work

### 2.1 Expoloration with ESM2
In last week, we tried some naive approaches to use the sequence tokenizer of ESM2 to replace the original amino acid representation (1-hot embedding with 1D CNN) in EvoPlay.
However, we found that the naive replacement did not yield satisfactory results. So in this week, we explored more possibilities using ESM2, including:
- Try more neural network architectures to replace the simple MLP feedforward network in naive solution.
- Explore more speed up methods to accelerate the training process.

### 2.2 MuZero for EvoPlay
The main diffence between MuZero and AlphaZero is that MuZero learns to model the environment dynamics implicitly, so MuZero can ignore some certain constraints, like traditional game rules.
However, in our setting, the environment is not a game but a protein design space, which is more complex and less structured than traditional games, so we do not have much prior knowledge and constraints about the environment.
Therefore, the advantages of MuZero may not be fully realized in our setting.
However, we still think it is worth exploring the feasibility of applying MuZero to replace the original AlphaZero module in our setting, so we did some preliminary analysis and exploration on this topic.

### 2.3 Review of Directed Evolution and Sampling in Protein Sequence Space
We review the following papers:
- Frey, Nathan C., et al. "Protein discovery with discrete walk-jump sampling." arXiv preprint arXiv:2306.12360 (2023). https://doi.org/10.48550/arXiv.2306.12360
- Kirjner, Andrew, et al. "Improving protein optimization with smoothed fitness landscapes." arXiv preprint arXiv:2307.00494 (2023). https://doi.org/10.48550/arXiv.2307.00494

Though reviewing these papers, we try to find out more applications of directed evolution and sampling in protein sequence space, which may provide insights for our work.
