# Examples

Demo notebooks for latent space classification and editing.

## Prerequisites

- GPU recommended for decoding

## Notebooks

| Notebook | Description |
|----------|-------------|
| `classification_demo.ipynb` | Train linear classifiers (LDA, LR) on latent representations for CelebA attributes |
| `editing_demo.ipynb` | Edit images by moving latents along learned attribute directions |

## Running

From the project root:
```bash
jupyter notebook examples/classification_demo.ipynb
```

Or from `examples/`:
```bash
cd examples
jupyter notebook classification_demo.ipynb
```

The notebooks add `..` to `sys.path` so imports work from the project root.
