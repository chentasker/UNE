"""
Train linear classifiers (LDA, Logistic Regression, SGD) on latent representations.

Used for CelebA attribute classification in diffusion/encoder latent spaces.
Supports PCA reduction and returns (W, b) for use with edit_sample().
"""
#%%
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

# For main
from utils.data_utils import train_test_split, get_celeba_attributes, save_results
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def train_attr(X_train, Y_train,
               X_test, Y_test,
               method='lda', # 'lda', 'lr', 'sgd'
               kwargs={}):
    """
    Train a single linear classifier for one binary attribute.
    Returns (acc, auc, W, b) where W, b define the decision boundary.
    """

    if method.lower() == 'lr':
        solver = kwargs.get('solver', 'saga')
        max_iter = kwargs.get('max_iter', 500)
        clf = LogisticRegression(solver=solver,
                                 max_iter=max_iter)
        clf.fit(X_train, Y_train)
        W = clf.coef_.ravel()
        b = clf.intercept_[0]

    elif method.lower() == 'lda':
        n1 = Y_train.sum()
        n0 = len(Y_train) - n1
        mean1 = np.mean(X_train[Y_train==1], axis=0)
        mean0 = np.mean(X_train[Y_train==0], axis=0)
        W = mean1 - mean0
        b = -0.5 * (mean1 + mean0) @ W + np.log(n1/n0)

    elif method.lower() == 'sgd':
        batch_size = kwargs.get('batch_size', 64)
        lr = kwargs.get('lr', 0.001)
        n_epochs = kwargs.get('n_epochs', 5)
        weight_decay = kwargs.get('weight_decay', 0.01)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        X_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.as_tensor(Y_train, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = torch.nn.Linear(X_train.shape[1], 1).to(device)

        _, _, w_lda, b_lda = train_attr(X_train, Y_train, X_test, Y_test, 'lda')
        with torch.no_grad():
            model.weight.copy_(torch.as_tensor(w_lda.copy()).view_as(model.weight))
            model.bias.copy_(torch.as_tensor(b_lda.copy()).view_as(model.bias))

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model.train()
        for epoch in range(n_epochs):
            running_loss = 0.0
            for batch_X, batch_Y in loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            with torch.no_grad():
                test_scores = X_test @ model.weight.ravel().detach().cpu().numpy() + model.bias.detach().cpu().numpy()
                test_acc = np.mean((test_scores > 0).astype(int) == Y_test)
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}, Test Accuracy: {test_acc}")
        
        W = model.weight.ravel().detach().cpu().numpy()
        b = model.bias.detach().cpu().numpy()
    
    test_scores = X_test @ W + b
    acc = np.mean((test_scores > 0).astype(int) == Y_test)
    auc = roc_auc_score(Y_test, test_scores)
    return acc, auc, W, b

def train_all_attr(X_train, Y_train,
                   X_test, Y_test,
                   method=None, kwargs={}):
    """Train classifiers for all attributes. Returns (accs, aucs, Ws, bs)."""
    num_attr = Y_train.shape[1]
    results = []
    for att_idx in tqdm(range(num_attr)):
        results.append(
            train_attr(X_train, Y_train[:,att_idx],
                       X_test, Y_test[:,att_idx],
                       method, kwargs)
        )
    accs = np.array([r[0] for r in results])
    aucs = np.array([r[1] for r in results])
    Ws = [r[2] for r in results]
    bs = [r[3] for r in results]
    return accs, aucs, Ws, bs

def get_equiv_linear(*args):
    """Compose multiple linear layers (W, b) into a single equivalent (W_equiv, b_equiv)."""
    assert len(args) % 2 == 0, "You must pass pairs of (W, b)."
    layers = [(args[i], args[i+1]) for i in range(0, len(args), 2)]
    
    W_equiv = layers[0][0]
    b_equiv = layers[0][1]
    for W, b in layers[1:]:
        W_equiv = W @ W_equiv
        b_equiv = b_equiv @ W.T + b
        
    return W_equiv, b_equiv.flatten()

def train_classifier(noises, Y, model_name,
                     pca_dim=None,
                     do_standard=True,
                     train_idx=None,
                     method='lda', kwargs=None):
    """
    Full pipeline: PCA (optional) -> StandardScaler -> linear classifier.
    Returns results dict with W_equiv, b_equiv (in original latent space) for use with edit_sample.
    W_equiv = W_clf @ W_sc @ W_pca (composition PCA -> Scaler -> Classifier).
    """
    if kwargs is None:
        kwargs = {}
    indices = np.arange(noises.shape[0])
    if train_idx is None:
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    else:
        test_idx = np.array([i for i in indices if i not in train_idx])

    X = noises.copy()
    X_train = X[train_idx]
    X_test = X[test_idx]

    # PCA
    pca = None
    if pca_dim is not None and X.shape[1] > pca_dim:
        print(f"Applying PCA (dim = {pca_dim})")
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print("Explained variance ratio:", pca.explained_variance_ratio_.sum())
        W_pca = pca.components_  # (n_comp, latent_dim)
        b_pca = -pca.mean_ @ pca.components_.T  # (n_comp,)
    else:
        W_pca = np.eye(X.shape[1])
        b_pca = np.zeros(X.shape[1])

    # StandardScaler
    if do_standard:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        W_sc = np.diag(1.0 / scaler.scale_)
        b_sc = -scaler.mean_ / scaler.scale_
    else:
        scaler = None
        W_sc = np.eye(X_train.shape[1])
        b_sc = np.zeros(X_train.shape[1])

    # Classifier
    print(f"Training {method} classifier on {model_name} latents...")
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    accs, aucs, Ws, bs = train_all_attr(X_train, Y_train, X_test, Y_test, method, kwargs)

    num_attributes = Y.shape[1]
    W_clf = np.array(Ws)  # (n_attrs, n_comp)
    b_clf = np.array(bs)  # (n_attrs,)

    # Compose: W_equiv = W_clf @ W_sc @ W_pca (per attribute)
    W_equiv_list = []
    b_equiv_list = []
    for i in range(num_attributes):
        We, be = get_equiv_linear(W_pca, b_pca, W_sc, b_sc, W_clf[i], b_clf[i])
        W_equiv_list.append(We)
        b_equiv_list.append(np.squeeze(be))
    W_equiv = np.array(W_equiv_list)  # (n_attrs, latent_dim)
    b_equiv = np.array(b_equiv_list)  # (n_attrs,)

    results = {
        'model_name': model_name,
        'W_clf': W_clf,
        'b_clf': b_clf,
        'W_equiv': W_equiv,
        'b_equiv': b_equiv,
        'W_pca': W_pca,
        'b_pca': b_pca,
        'W_sc': W_sc,
        'b_sc': b_sc,
        'pca': pca,
        'scaler': scaler,
        'test_accs': accs,
        'test_aucs': aucs,
        'noises': noises,
        'noises_reduced': X.copy(),
        'train_idx': train_idx,
        'test_idx': test_idx,
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, nargs='+', help="sd15, sd21, sdxl, etc., or raw")
    parser.add_argument("--method", type=str, default='lda', help="lda, lr, sgd")
    parser.add_argument("--split", type=str, default='train', help="train, validation")
    parser.add_argument("--pca_dim", type=int, default=None, help="PCA dimension (default: 500 for diffusion, 310 for encoders)")
    parser.add_argument("--noises_dir", type=str, default=None, help="Directory for noises (default: data/celeba_<split>_noises)")
    args = parser.parse_args()

    pca_dim_for_model = {'sd15': 500, 'sd21': 500, 'lcmv7': 500, 'sdxl': 500,
                        'vitL14': 310, 'vitB16': 310, 'vitL14_oc': 310, 'vitB16_oc': 310, 'dinov3': 310}

    print('Preparing dataset')
    Y = get_celeba_attributes(args.split)
    train_indices, test_indices = train_test_split(len(Y))

    print('args:', args)
    all_results = []
    for model in args.model:
        if model == 'raw':
            from pipeline_wrappers import preprocess_img
            from prepare_data import get_ds
            ds = get_ds('celeba', split=args.split)
            print('Preprocessing ds...')
            X = []
            for i in tqdm(range(len(ds))):
                X.append(preprocess_img(ds[i]['image'], return_type='tensor'))
            noises = torch.stack(X).flatten(1).detach().numpy()
            results = train_classifier(noises, Y, model, pca_dim=None, train_idx=train_indices, method=args.method)
        else:
            noises_dir = args.noises_dir or f'data/celeba_{args.split}_noises'
            noises_path = f'{noises_dir}/noises_{model}.npy'
            noises = np.load(noises_path)
            pca_dim = args.pca_dim or pca_dim_for_model.get(model, 500)
            results = train_classifier(noises, Y, model, pca_dim=pca_dim, train_idx=train_indices, method=args.method)

        all_results.append(results)

    save_results(all_results, results_dir='./data')

    if len(all_results) == 1:
        r = all_results[0]
        print('Test Accuracies:\n', r['test_accs'])
        print('Test AUCs:      \n', r['test_aucs'])
