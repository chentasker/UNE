"""
Data loading and preparation utilities.
"""
import os
import numpy as np
import glob
import pickle
from datetime import datetime
import re
from torchvision import transforms
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download


def train_test_split(n, test_size=0.2, random_state=42):
    """Split indices into train/test. n can be an int (length) or array of indices."""
    if isinstance(n, int):
        indices = np.arange(n)
    else:
        indices = np.array(n)
    
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx

def load_noises(model_names):
    """Load precomputed latent encodings (noises) for each model from .npy files."""
    if isinstance(model_names, str):
        model_names = [model_names]
    noises = [None]*len(model_names)
    for model_index, model_name in enumerate(model_names):
        noises_path = hf_hub_download(
            repo_id="chentasker/NoiseZoo",
            filename=f"data/noises_{model_name}.npy",
            repo_type="dataset",
        )
        noises[model_index] = np.load(noises_path)
    return noises

def load_results(model_names, results_dir="../results"):
    """Load classifier results (W_clf, b_clf, etc.) from pickle files. Uses latest by date."""
    if not isinstance(model_names, list):
        model_names = [model_names]
    all_results = [None] * len(model_names)
    for i, model_name in enumerate(model_names):
        results_files = glob.glob(os.path.join(results_dir, f"results_{model_name}_*.pkl"))
        results_files = [f for f in results_files if re.match(rf".*results_{re.escape(model_name)}_[0-9]+\.pkl$", f)]
        if len(results_files) == 0:
            print(f"No results file for {model_name}")
            continue
        with open(results_files[-1], "rb") as f:
            all_results[i] = pickle.load(f)

    assert all([all_results[0].keys() == all_results[i].keys() for i in range(len(model_names))])
    print("Loaded all results, keys: ")
    print(list(all_results[0].keys()))
    return all_results

def save_results(all_results, results_dir="./data"):
    """Save classifier results to pickle files with timestamp in filename."""
    if not isinstance(all_results, list):
        all_results = [all_results]
    os.makedirs(results_dir, exist_ok=True)
    for i in range(len(all_results)):
        filename = f"results_{all_results[i]['model_name']}_{datetime.now().strftime("%y%m%d")}.pkl"
        filename = os.path.join(results_dir, filename)
        with open(filename, "wb") as f:
            pickle.dump(all_results[i], f)

def get_celeba_attributes(split='train'):
    ds = get_ds('celeba', split)
    return np.array(ds['attributes'])

def get_celeba_attribute_names():
    return {
        0: '5_o_Clock_Shadow',
        1: 'Arched_Eyebrows',
        2: 'Attractive',
        3: 'Bags_Under_Eyes',
        4: 'Bald',
        5: 'Bangs',
        6: 'Big_Lips',
        7: 'Big_Nose',
        8: 'Black_Hair',
        9: 'Blond_Hair',
        10: 'Blurry',
        11: 'Brown_Hair',
        12: 'Bushy_Eyebrows',
        13: 'Chubby',
        14: 'Double_Chin',
        15: 'Eyeglasses',
        16: 'Goatee',
        17: 'Gray_Hair',
        18: 'Heavy_Makeup',
        19: 'High_Cheekbones',
        20: 'Male',
        21: 'Mouth_Slightly_Open',
        22: 'Mustache',
        23: 'Narrow_Eyes',
        24: 'No_Beard',
        25: 'Oval_Face',
        26: 'Pale_Skin',
        27: 'Pointy_Nose',
        28: 'Receding_Hairline',
        29: 'Rosy_Cheeks',
        30: 'Sideburns',
        31: 'Smiling',
        32: 'Straight_Hair',
        33: 'Wavy_Hair',
        34: 'Wearing_Earrings',
        35: 'Wearing_Hat',
        36: 'Wearing_Lipstick',
        37: 'Wearing_Necklace',
        38: 'Wearing_Necktie',
        39: 'Young'
    }
                
def get_ds(ds='celeba', split=None):
    """Load dataset by name: celeba, celeba-hq, imagenet, cub200, afhq, etc."""
    if ds == 'celeba':
        if split is None:
            split = 'validation'
        ds = load_dataset("eurecom-ds/celeba", split=split)
        
    elif ds == 'celeba-hq':
        if split is None:
            split = 'validation'
        pass
    
    elif ds == 'imagenet':
        if split is None:
            split = 'train'
        ds = load_dataset("ILSVRC/imagenet-1k", split=split)
        # Filter 20 classes
        mask = np.isin(np.array(ds["label"]), np.arange(20))
        indices = np.where(mask)[0]
        ds = ds.select(indices)
        
    else:
        raise KeyError(f"Unkown ds: {ds}")
    return ds

def preprocess_img(img, resolution=512):
    """Center-crop to square and resize to resolution. Returns PIL Image."""
    size = min(img.size)
    transform = transforms.Compose([
        transforms.CenterCrop(size),  # crop to smallest side
        transforms.Resize((resolution, resolution)),
    ])
    return transform(img)
