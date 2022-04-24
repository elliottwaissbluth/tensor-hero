import pickle
from pathlib import Path

try:
    with open(Path.cwd() / 'tensor_hero' / 'preprocessing' / 'resources' / 'simplified_note_dict.pkl', 'rb') as f:
        simplified_note_dict = pickle.load(f)
except FileNotFoundError:
    try:
        with open(Path.cwd().parent / 'tensor_hero' / 'preprocessing' / 'resources' / 'simplified_note_dict.pkl', 'rb') as f:
            simplified_note_dict = pickle.load(f)
    except FileNotFoundError:
        with open(Path.cwd().parent.parent / 'tensor_hero' / 'preprocessing' / 'resources' / 'simplified_note_dict.pkl', 'rb') as f:
            simplified_note_dict = pickle.load(f)
        