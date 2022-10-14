from pathlib import Path
from .recording import DirectoryRecording, ZipRecording
from .summarize_dpoae import isodp_th_criterions


def open(path):
    path = Path(path)
    if path.suffix == '.zip':
        return ZipRecording(path)
    elif path.is_dir():
        return DirectoryRecording(path)
    else:
        raise NotImplementedError
