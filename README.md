# The Universal Normal Embedding (UNE)

Official code for **The Universal Normal Embedding** (arXiv:2603.21786), studying how **unguided diffusion** represents images.

## What this repository is about

We advance the **Universal Normal Embedding (UNE)** hypothesis: that there is a **shared, Gaussian latent representation** underlying both **generative models** and **image encoders**—two lines of work that are usually treated separately, but may be **noisy linear views** of the same underlying space.

On the empirical side, we show that the **initial noise** used in unguided diffusion is not semantically inert: it carries **structured, rankable information** about the image that will be generated. For example, a **linear classifier** fit on the starting noise can predict high-level attributes of the output—such as whether the generated face will read as male or female—without looking at the final pixels. That pattern extends to other attributes and settings described in the paper.

Subsequently, we use those linear classifiers to edit generated images by manipulating the initial noise, and apply geometric techniques to identify editing directions orthogonal to undesired attribute changes, thereby removing spurious attribute changes during editing.

## Links

| Resource | Link |
|----------|------|
| **Paper** | [Paper on arXiv](https://arxiv.org/abs/2603.21786) |
| **Project website** | [UNE project page](https://rbetser.github.io/UNE/) |
| **Dataset** | [NoiseZoo on Hugging Face](https://huggingface.co/datasets/chentasker/NoiseZoo) |

## Repository layout

- **`utils/`** — Training linear probes (LDA, logistic regression, etc.), data helpers, diffusion pipeline wrappers, and utilities for **linear editing** in latent / noise space (including orthogonalization and optional gaussianization for interpretable edit strength).
- **`examples/`** — Notebook walkthroughs (see [`examples/README.md`](examples/README.md)).

## Example notebooks

| Notebook | What it demonstrates |
|----------|------------------------|
| `examples/classification_demo.ipynb` | Linear classification on latent representations (e.g. CelebA attributes). |
| `examples/editing_demo.ipynb` | **Linear editing** by moving latents along directions learned from linear probes (e.g. attribute directions). |

GPU is recommended for decoding and diffusion-related steps.

## Setup

Python 3 with PyTorch is assumed. From the repository root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins CUDA-oriented PyTorch builds (see `--extra-index-url` inside the file); PyTorch can still run on CPU when no GPU is available.

## Running the notebooks

From the project root:

```bash
jupyter notebook examples/classification_demo.ipynb
```

The notebooks add the parent directory to `sys.path` so `utils` imports resolve when run from `examples/`.

## Citation

If you use this code or the associated dataset, please cite the paper.

## License

See [`LICENSE`](LICENSE).
