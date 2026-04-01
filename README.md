# DSALT: Dynamic Sparse Attention with Landmark Tokens

> Companion code for the paper *"Noise Accumulation and Rank Collapse in Dense Self-Attention: DSALT"*  
> Preprint: [Zenodo](https://doi.org/10.5281/zenodo.19312827)

---

[Open the notebook in Colab](https://colab.research.google.com/github/LeonardoCofone/DSALT/blob/main/dsalt.ipynb)

## Overview

Standard Transformer decoders use dense self-attention: every token attends to every previous token. Because softmax weights are strictly positive, semantically irrelevant tokens always contribute to a token's representation, a structural source of noise that accumulates across heads and layers, accelerating **rank collapse** and degrading performance on long contexts.

**DSALT** addresses this at the source. Instead of attending to all previous tokens, each token attends only to:
- an **adaptive local window** whose size is predicted dynamically from the token's hidden state
- a small set of **landmark tokens**, the *k* most informationally dense tokens outside the window, selected by representation norm

This reduces the attention set from *n* to *w(i) + k ≪ n*, cutting noise without losing long-range information.


## Results (controlled experiment, identical model size and training budget)

| Metric | Dense (Transformer) | DSALT| Δ |
|---|---|---|---|
| Validation Perplexity | 172.62 | 156.78 | −9.2% |
| Sliding-Window PPL (long context) | 152.86 | 132.97 | −13.0% |
| Avg. Noise Norm ‖η‖ | 0.4072 | 0.3451 | −15.3% |
| Avg. Effective Rank | 456.1 | 477.8 | +4.8% |
| Avg. Attention Entropy | 2.959 | 3.305 | +11.7% |
| Avg. it/s | 0.37 | 0.37 | 0% |

The perplexity gap widens on longer sequences, exactly as the cumulative noise hypothesis predicts.


## Repository Structure

```
dsalt.ipynb        # Full experiment: models, training, evaluation, plots
README.md
```

The notebook is self-contained and reproduces all results in the paper.

## How to Run

### On Kaggle (recommended, free GPU)

1. Upload `dsalt.ipynb` to [kaggle.com/code](https://www.kaggle.com/code)
2. Enable GPU accelerator (P100 or T4)
3. Click **Run All**

Outputs are saved to `/kaggle/working/`: `comparison.png`, `layerwise_collapse.png`, `dense_history.json`, `dense_model.pt`.

### Locally

```bash
pip install torch datasets transformers accelerate matplotlib
jupyter notebook dsalt.ipynb
```

A CUDA GPU is strongly recommended. The experiment runs for 5500 steps on 10% of WikiText-103.


## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 512 | Embedding dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Transformer layers |
| `seq_len` | 1024 | Training sequence length |
| `w_min` | 64 | Minimum adaptive window size |
| `w_max` | 512 | Maximum adaptive window size |
| `k_landmarks` | 8 | Landmark tokens per head |
| `max_steps` | 5500 | Training steps |

Both models (Dense and DSALT) are trained with identical architecture, data, optimizer, and seed.

## Method Summary

### Adaptive Window
Each token *i* computes a window size dynamically:

```
w(i) = w_min + floor(sigmoid(f(x_i)) * (w_max - w_min))
```

where `f` is a single learned linear layer.

### Landmark Token Selection
The *k* tokens with highest L2 norm outside the local window are selected as landmarks:

```
L(i) = top-k { ‖x_j‖₂ : j < i − w(i) }
```

### Full Attention Set
```
A(i) = local_window(i) ∪ L(i)
```

Complexity drops from O(n²d) to O(n(w+k)d)

## What the Notebook Measures

Beyond perplexity, the notebook tracks seven metrics throughout training to validate the theoretical claims:

- **Training loss** and **validation perplexity**
- **Second singular value σ₂** of the attention matrix (higher = slower rank collapse)
- **Effective rank** of token representations (higher = more diverse)
- **Residual norm ratio** across layers (layer-wise rank collapse visualization)
- **Attention entropy** (higher = less noise concentration)
- **Noise norm ‖η‖**, directly measured irrelevant contributions
- **Step time**, wall-clock cost per iteration


## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{cofone2025dsalt,
  author    = {Cofone, Leonardo},
  title     = {Noise Accumulation and Rank Collapse in Dense Self-Attention: DSALT},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19312827},
  url       = {https://doi.org/10.5281/zenodo.19312827}
}
```

## License

MIT