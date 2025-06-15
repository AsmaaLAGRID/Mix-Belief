import pandas as pd
from sklearn.model_selection import train_test_split
from .data_configs import DATASETS
from datasets import Dataset
from torch.utils.data import DataLoader
from .utils import create_imbalance_ratio, create_sample
from pathlib import Path
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder

SPLIT_SEED = 42
VALID_IRS=[10,25, 50, 80, 100]
VALID_SRS=[0.1, 0.25, 0.5, 0.8]


def read_txtfile(path:str) -> pd.DataFrame:
    labels = []
    texts = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            label, text = line.split('\t', 1)
            labels.append(label)
            texts.append(text)
    return pd.DataFrame({"text": texts, "label": labels})

def _read_file(path: str, fmt: str) -> pd.DataFrame:
    if fmt == 'csv':
        return pd.read_csv(path)
    if fmt == 'jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            return pd.read_json(f, lines=True)
    if fmt == 'parquet':
        return pd.read_parquet(path)
    if fmt == 'txt':
        return read_txtfile(path)
    raise ValueError(f"unknown format of file: {fmt}")

def preprocess_train_df(df, cfg):
    mode = cfg.dataset.mode
    if mode == 'original':
        return df
    if mode == 'imbalance':
        if cfg.dataset.ir not in VALID_IRS and cfg.dataset.ir != 1:
            raise ValueError(f"IR must be one of {VALID_IRS} (or 1 for no imbalance), got ir={cfg.dataset.ir}")

        if cfg.dataset.ir == 1:
            return df
        return create_imbalance_ratio(df, cfg.dataset.ir, SPLIT_SEED)
    if mode == 'sample':
        if cfg.dataset.sr != 1.0 and cfg.dataset.sr not in VALID_SRS:
            raise ValueError(
                f"Sampling ratio must be 1.0 (no sampling) or one of {sorted(VALID_SRS)}, "
                f"but got sr={cfg.dataset.sr}"
            )
        if cfg.dataset.sr == 1.0:
            return df
        return create_sample(df, cfg.dataset.sr, SPLIT_SEED)
    raise ValueError(f"Unknown mode: {mode}")

def tokenize_df(df, tokenizer, cfg):
    ds = Dataset.from_pandas(df)
    return ds.map(
        lambda x: tokenizer(
            x['text'],
            truncation=True,
            padding='max_length',
            max_length=cfg.train.max_length
        ),
        batched=True
    ).with_format('torch', columns=['input_ids','attention_mask','label'])

def load_raw_dfs(name: str):
    cfg = DATASETS[name]
    text_col = cfg['text_col']
    label_col = cfg['label_col']
    dfs = {}

    for split, path in cfg['files'].items():
        df = _read_file(path, cfg['format'])
        df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})

        df['text'] = df['text'].astype(str)
        #df['label'] = df['label'].astype(int)
        dfs[split] = df

    # si cfg['splits'] est défini, on splitte manuellement
    #TODO prendre val a partir du train et non pas a partir de test
    if cfg.get('need_label_encoding', False):
        print("########## Je suis la pour NG20 label Encoding ########## ")
        le = LabelEncoder()
        le.fit(dfs['train']['label'])
        for split in dfs:
            dfs[split]['label'] = le.transform(dfs[split]['label']).astype(int)
    else:
        for split in dfs:
            dfs[split]['label'] = dfs[split]['label'].astype(int)

    if 'splits' in cfg and 'train' in dfs and 'test' in dfs:
        train_p, val_p = (p for _, p in cfg['splits'])
        full_train = dfs.pop('train')
        train_df, val_df = train_test_split(
            full_train, test_size=val_p / (train_p + val_p),
            stratify=full_train.label, random_state=SPLIT_SEED
        )
        dfs = {'train': train_df, 'val': val_df, 'test': dfs['test']}

    elif 'splits' in cfg and 'all' in dfs:
        full = dfs.pop('all')
        train_p, val_p, test_p = (p for _, p in cfg['splits'])
        train_df, temp = train_test_split(full, test_size=(1 - train_p),
                                          stratify=full.label, random_state=SPLIT_SEED)
        val_df, test_df = train_test_split(temp,
                                           test_size=test_p / (val_p + test_p),
                                           stratify=temp.label, random_state=SPLIT_SEED)
        dfs = {'train': train_df, 'val': val_df, 'test': test_df}
    return dfs['train'], dfs['val'], dfs['test']

def build_dataset(cfg, tokenizer, logger):
    
    if cfg.dataset.mode=='imbalance':
        cache_dir = Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}_{cfg.dataset.ir}"
    elif cfg.dataset.mode=='sample':
        cache_dir = Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}_{cfg.dataset.sr}"
    elif cfg.dataset.mode=='original':
        cache_dir = Path("cache") / f"{cfg.dataset.name}_{cfg.dataset.mode}"
    
    if cache_dir.exists():
        train_ds = load_from_disk(cache_dir / "train")
        val_ds = load_from_disk(cache_dir / "val")
        test_ds = load_from_disk(cache_dir / "test")
    else:
        train_df, val_df, test_df = load_raw_dfs(
                name=cfg.dataset.name
        )
        logger.info(f"label values count of train_df before preprocessing: {train_df.label.value_counts()}")
        logger.info(f"label values count of val_df after preprocessing: {val_df.label.value_counts()}")
        logger.info(f"label values count of test_df after preprocessing: {test_df.label.value_counts()}")

        train_df = preprocess_train_df(train_df, cfg)
        logger.info(f"label values count of train_df after preprocessing: {train_df.label.value_counts()}")

        #Tokenization
        train_ds = tokenize_df(train_df, tokenizer, cfg)
        val_ds   = tokenize_df(val_df, tokenizer, cfg)
        test_ds  = tokenize_df(test_df, tokenizer, cfg)

        train_ds.save_to_disk(cache_dir / "train")
        val_ds.save_to_disk(cache_dir / "val")
        test_ds.save_to_disk(cache_dir / "test")
    # DataLoaders
    return (
            DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False),
            DataLoader(test_ds,  batch_size=cfg.train.batch_size, shuffle=False),
    )
