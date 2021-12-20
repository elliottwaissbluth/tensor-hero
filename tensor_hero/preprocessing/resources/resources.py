import pickle
from pathlib import Path

with open(Path.cwd() / 'tensor_hero' / 'preprocessing' / 'resources' / 'simplified_note_dict.pkl', 'rb') as f:
    simplified_note_dict = pickle.load(f)
