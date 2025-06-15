
import os

BASE_DIR = os.getcwd()
DATASETS = {
    'imdb': {
        'files': {
            'all': 'Datasets/IMDB/IMDB_Dataset.csv'
        },
        'format': 'csv',
        'rename': {'review': 'text', 'sentiment': 'label'},
        'need_label_encoding': True,
        'splits': [('train', 0.64), ('val', 0.16), ('test', 0.20)]
    },
    'sst2': {
        'files': {
            'train': 'datasets/SST-2/train.jsonl',
            'val':   'datasets/SST-2/dev.jsonl',
            'test':  'datasets/SST-2/test.jsonl'
        },
        'format': 'jsonl',
        'text_col': 'text',
        'label_col': 'label',
        'need_label_encoding': False,
        'num_labels': 2,
    },
    'sst5': {
        'files': {
            'train': 'datasets/SST-5/train.jsonl',
            'val':   'datasets/SST-5/dev.jsonl',
            'test':  'datasets/SST-5/test.jsonl'
        },
        'format': 'jsonl',
        'text_col': 'text',
        'label_col': 'label',
        'need_label_encoding': False,
        'num_labels': 5,
    },
    'mr': {
        'files': {
            'train': 'datasets/MR/train.parquet',
            'val':   'datasets/MR/validation.parquet',
            'test':  'datasets/MR/test.parquet'
        },
        'format': 'parquet',
        'text_col': 'text',
        'label_col': 'label',
        'need_label_encoding': False,
        'num_labels': 2,
    },
    'ng20': {
        'files': {
            'train': 'datasets/NG20/20ng-train-all-terms.txt',
            'test':  'datasets/NG20/20ng-test-all-terms.txt'
        },
        'format': 'txt',
        'text_col': 'text',
        'label_col': 'label',
        'splits': [('train', 0.8), ('val', 0.2)],
        'need_label_encoding': True,
        'num_labels': 20,
    },
    'r8': {
        'files': {
            'train': 'datasets/R8/r8-train-all-terms.txt',
            'test':  'datasets/R8/r8-test-all-terms.txt'
        },
        'format': 'txt',
        'text_col': 'text',
        'label_col': 'label',
        'splits': [('train', 0.8), ('val', 0.2)],
        'need_label_encoding': True,
        'num_labels': 8,
    },
    'ohsumed': {
        'files': {
            'train': 'datasets/Ohsumed/oh-train-stemmed.txt',
            'val': 'datasets/Ohsumed/oh-dev-stemmed.txt',
            'test':  'datasets/Ohsumed/oh-test-stemmed.txt'
        },
        'format': 'txt',
        'text_col': 'text',
        'label_col': 'label',
        'need_label_encoding': True,
        'num_labels': 23,
    },
}