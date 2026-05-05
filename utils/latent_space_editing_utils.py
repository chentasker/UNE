"""
Latent space editing utilities for diffusion models.

Provides linear editing in the latent space: moving samples along learned attribute
directions (e.g., from LDA/linear classifiers), with optional orthogonalization
and Yeo-Johnson gaussianization for more interpretable intensity scaling.
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer
from PIL import Image, ImageDraw
import imageio
from utils.pipeline_wrappers import SDPipelineWrapper


def get_orthogonal_direction(direction, direction_fixed):
    """Project direction onto the subspace orthogonal to direction_fixed (Gram-Schmidt)."""
    return direction - (np.sum(direction * direction_fixed) / np.sum(direction_fixed * direction_fixed)) * direction_fixed


def get_alpha(noises, W, b):
    """Compute alpha: scalar such that (sample + alpha*W) lies on the decision boundary."""
    return - (noises @ W + b) / (W @ W.T)


def edit_sample(sample, direction, intensity, direction_fixed=None):
    """
    Linearly edit a latent sample along a direction.
    If direction_fixed is given, direction is first orthogonalized against it.
    """
    if direction_fixed is not None:
        direction = get_orthogonal_direction(direction, direction_fixed)
    return sample + intensity * direction


def fit_yeo_johnson(projections):
    """Fit Yeo-Johnson transform to make projections approximately Gaussian. Returns transform, inverse."""
    x = np.asarray(projections).reshape(-1,1)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)  # returns approx Gaussian
    pt.fit(x)
    return pt.transform, pt.inverse_transform


def edit_sample_gaussianized(noises, sample_idx, W, b, intensity):
    """
    Edit sample using intensity in Gaussianized projection space.
    Yeo-Johnson transform makes alpha (projection onto W) approximately Gaussian,
    so intensity has more interpretable effect (e.g., in standard-deviation units).

    Note: the diffusion latents are Gaussian by design, so this may not have a huge effect,
    except for non-Gaussian or not standardized directions that appeared due to finite data.
    """
    alpha_dataset = - (noises @ W + b) / (W @ W.T)
    alpha_sample = alpha_dataset[sample_idx]

    # Gaussianize alpha using Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson', standardize=True)  # returns approx Gaussian
    pt.fit(alpha_dataset.reshape(-1,1))
    transform, inverse_transform = pt.transform, pt.inverse_transform

    target_intensity = transform(np.array([[alpha_sample]])) + intensity
    alpha_new = inverse_transform(target_intensity)
    return noises[sample_idx] + (alpha_new-alpha_sample)*W

def make_video(pipe: SDPipelineWrapper, start_latent, num_steps, direction=None, end_latent=None,
               filename='out.avi', draw_step=False, fps=5):
    """
    Generate a video by interpolating from start_latent along direction (or to end_latent).
    Decodes each frame and saves as lossless AVI.
    """
    if direction is None and end_latent is None:
        raise ValueError('Either direction or end_latent must be provided')
    if direction is None:
        direction = (end_latent - start_latent) / (num_steps - 1)

    writer = imageio.get_writer(
        filename,
        fps=fps,
        codec='ffv1',     # lossless codec
        format='FFMPEG'
    )
    
    for i in range(num_steps):
        latent = start_latent + direction*i
        reconstructed_image = pipe.decode_latent(latent, show_tqdm=False)
        if draw_step:
            draw = ImageDraw.Draw(reconstructed_image)
            draw.text((10,10), f"{i}", fill=(0,0,0), font_size=20)
        writer.append_data(np.array(reconstructed_image))
    writer.close()
