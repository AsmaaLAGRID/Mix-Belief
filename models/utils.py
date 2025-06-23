import os
import csv
import re
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.metrics import geometric_mean_score
from sklearn.utils import resample
from collections import Counter
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError



def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_counts(loader, num_cls):
    # Initialize a counter
    class_counts = Counter()

    # Iterate through the data loader
    for batch in loader:
        labels = batch.pop("label")
        # Update the counter with labels
        class_counts.update(labels.tolist())

    counts_array = [class_counts[i] for i in range(num_cls)]

    return counts_array

def build_optimizer(cfg, model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    num_training_steps = len(loader) * cfg.train.epochs
    num_warmup_steps = int(num_training_steps * cfg.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    return optimizer , scheduler

def get_perm(x):
    """get random permutation"""
    batch_size = x.size()[0]
    device = x.device
    index = torch.randperm(batch_size).to(device)
    return index

def compute_metrics(y_true, y_pred, probs, n_classes):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    gm = geometric_mean_score(y_true, y_pred, average='macro')
    gm = gm.item()
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    y_true_tensor = torch.tensor(y_true, dtype=torch.long)
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    mcce = MulticlassCalibrationError(num_classes=n_classes, n_bins=n_classes, norm='l1') 
    mcce_value = mcce(probs_tensor, y_true_tensor)
    mcce_value = mcce_value.item()
    return {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'gm': gm, 
        'mcce': mcce_value,
        'cm': cm
    }

def display_cm(save_dir, wandb, cm, run_name, seed=None, epoch=None, step=None, data='Validation'):

    num_classes = cm.shape[0]

    # Ajustement automatique de la taille de la figure
    fig_size = max(6, int(num_classes * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f"{data} CM — {run_name}")
    plt.tight_layout()

    if data == 'Val':
        filename = f"{run_name}_epoch{epoch:02d}_step{step:04d}_cm.png"
    elif data == 'Test':
        filename = f"{run_name}_Test_cm.png"
    
    cm_dir = os.path.join(save_dir, f"seed_{seed}")
    os.makedirs(cm_dir, exist_ok=True)
    out_path = os.path.join(cm_dir, filename)
    fig.savefig(out_path, dpi=150)
    wandb.log({f"{data}/cm": wandb.Image(fig)})

    plt.close(fig)

    return out_path
    
# TODO revise this function to integrate uncertainty information #Done
def save_metrics_to_csv(path: str, mode: str, metrics: dict, epoch=None, step=None, seed=None):
    if mode not in ['val', 'test']:
        raise ValueError("Incorrect mode")

    base_header = ['mode']
    if mode == 'val':
        base_header += ['seed', 'epoch', 'step']
    base_header += ['loss', 'acc', 'prec', 'rec', 'f1', 'gm', 'mcce']

    extra_keys = []

    for key in ['mean_u', 'mean_uncertainty']:
        if key in metrics:
            extra_keys.append(key)
    
    for k in metrics.keys():
        if re.match(r'class_\d+_(mean_u|belief_\d+)', k):
            extra_keys.append(k)
    
    extra_keys = sorted(extra_keys, key=lambda x: (
        int(re.search(r'class_(\d+)', x).group(1)) if 'class_' in x else -1,
        int(re.search(r'belief_(\d+)', x).group(1)) if 'belief_' in x else -1
    ))

    header = base_header + extra_keys
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        # Compose la ligne
        row = [mode]
        if mode == 'val':
            row += [seed, epoch, step]
        row += [
            metrics.get('loss', ''),
            metrics.get('acc', ''),
            metrics.get('prec', ''),
            metrics.get('rec', ''),
            metrics.get('f1', ''),
            metrics.get('gm', ''),
            metrics.get('mcce', '')

        ]

        for key in extra_keys:
            row.append(metrics.get(key, ''))

        writer.writerow(row)


def create_sample(df, sr, split_seed):
    n_samples = int(len(df) * sr)
    return resample(
        df,
        n_samples=n_samples,
        replace=False,
        stratify=df["label"],
        random_state=split_seed
    )

'''def maybe_eda(df, cfg):
    if cfg.data.eda:
        return apply_eda(df, cfg)
    return df'''


def create_imbalance_ratio(df, ir, split_seed):
    class_counts = df['label'].value_counts()
    classes_sorted = class_counts.sort_values(ascending=False).index.tolist()  # tri décroissant
    n_classes = len(classes_sorted)
    n_max = class_counts.max()

    print("Original Distribution :\n", class_counts)
    print(f"Classe majoritaire conservée : {classes_sorted[0]} avec {n_max} exemples")
    print(f"Nombre de classes : {n_classes}, IR cible : {ir}")

    if ir <= 1:
        print("IR <= 1 : dataset original conservé (shuffle only)")
        return df.sample(frac=1, random_state=split_seed).reset_index(drop=True)

    # Formule log-tailed sur classes triées par fréquence décroissante
    alpha = np.log(ir) / np.log(n_classes)
    print(f"Alpha (forme du déséquilibre) = {alpha:.4f}")

    ranks = np.arange(1, n_classes + 1)  # rangs croissants
    desired_sizes = n_max / (ranks ** alpha)
    class_to_desired = dict(zip(classes_sorted, desired_sizes))

    resampled_dfs = []
    for cls in classes_sorted:
        df_cls = df[df['label'] == cls]
        desired = int(round(min(len(df_cls), class_to_desired[cls])))

        df_res = resample(
            df_cls,
            replace=False,
            n_samples=desired,
            random_state=split_seed
        )
        resampled_dfs.append(df_res)

    df_imbalanced = pd.concat(resampled_dfs).sample(frac=1, random_state=split_seed).reset_index(drop=True)

    counts_after = df_imbalanced['label'].value_counts()
    print("New distribution of classes :\n", counts_after)
    ir_actual = counts_after.max() / counts_after.min()
    print(f"IR obtenu : {ir_actual:.2f}")

    return df_imbalanced


def get_remix_y(y1, y2, lam_x, samples_per_class, K, tau, device):
        '''
        Returns mixed inputs, pairs of targets, and lambda_x, lambda_y
        *Args*
        k: hyper parameter of k-majority
        tau: hyper parameter
        where in original paper they suggested to use k = 3, and tau = 0.5
        Here, lambda_y is defined in the original paper of remix, where there
        are three cases of lambda_y as the following:
        (a). lambda_y = 0
        (b). lambda_y = 1
        (c). lambda_y = lambda_x
        '''

        cls_num_list = torch.tensor(samples_per_class)

        # check list stored pairs of image index where one mixup with the other
        check = []
        for i in range(len(y1)):
            check.append([cls_num_list[y1[i]].item(), cls_num_list[y2[i]].item()])
        check = torch.tensor(check)
        lam_y = []

        for i in range(check.size()[0]):
            # temp1 = n_i; temp2 = n_j
            temp1 = check[i][0]
            temp2 = check[i][1]

            if (temp1 / temp2) >= K and lam_x < tau:
                lam_y.append(0)
            elif (temp1 / temp2) <= (1 / K) and (1 - lam_x) < tau:
                lam_y.append(1)
            else:
                lam_y.append(lam_x)

        lam_y = torch.tensor(lam_y).to(device)

        return lam_y

def flatten_config(cfg, parent_key='', sep='/'):
    """Transforme une config imbriquée en un dict plat compatible wandb."""
    items = {}
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict) or hasattr(v, 'items'):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def t_sne_vis(embeddings, labels, seed, experiment, save_path):
    tsne = TSNE(n_components=3, random_state=seed)
    embeddings_3d = tsne.fit_transform(embeddings)

    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        color=labels,
        title=f"3D t-SNE Visualization of {experiment}"
    )

    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            zaxis_title="t-SNE Dimension 3",
            xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray"),
            yaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray"),
            zaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)", gridcolor="lightgray")
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )

    # Sauvegarde du fichier interactif HTML
    filename = os.path.join(save_path, f"tsne_seed_{seed}.html")
    pio.write_html(fig, file=filename, auto_open=False)
    
    # (Optionnel) Sauvegarde PNG (statique)
    fig.write_image(os.path.join(save_path, f"tsne_seed_{seed}.png"))

    return fig
