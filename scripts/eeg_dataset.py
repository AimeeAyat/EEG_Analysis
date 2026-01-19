import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGClipDataset(Dataset):
    """Dataset that returns (eeg: [C, T], clip_emb: [D]) per trial.

    Expects files in workspace:
      - EEGdata/sub01/process_data_1s_250Hz.npy  (N, C, T)
      - EEGdata/sub01/name_label.npy            (N,)
      - clip_feature.pth                         (dict uid->embedding) or tensor
      - description.json                         (map instance->uid)
    """

    def __init__(self, root, sub='sub01', eeg_key='process_data_1s_250Hz.npy', clip_path=None,
                 preprocess=True, baseline_window=None, zscore=True):
        self.root = root
        subj = os.path.join(root, 'EEGdata', sub)
        self.eeg = np.load(os.path.join(subj, eeg_key), allow_pickle=False)
        self.name_label = np.load(os.path.join(subj, 'name_label.npy'), allow_pickle=True)
        # load description.json mapping instance->uid
        desc_path = os.path.join(root, 'description.json')
        with open(desc_path, 'r', encoding='utf-8') as f:
            self.desc = json.load(f)
        # load clip features
        if clip_path is None:
            clip_path = os.path.join(root, 'clip_feature.pth')
        clip_data = torch.load(clip_path, map_location='cpu')
        self.clip = clip_data
        # Build mapping from trial index to clip embedding
        self.targets = []
        # preprocessing options
        self.preprocess = preprocess
        self.baseline_window = baseline_window
        self.zscore = zscore
        if self.preprocess:
            # apply per-trial baseline removal and per-channel z-score
            self.eeg = self._preprocess_eeg(self.eeg)
        for label in self.name_label:
            # label format examples: '01_airplane_06' or '19_control_00' or 'airplane_00'
            parts = label.split('_')
            if parts[0].isdigit():
                inst = '_'.join(parts[1:])
            else:
                inst = label
            emb = None
            # If clip features keyed by instance name directly
            if isinstance(self.clip, dict):
                # direct matches
                if label in self.clip:
                    emb = self.clip[label]
                elif inst in self.clip:
                    emb = self.clip[inst]
                else:
                    # maybe keys have uppercase/lowercase differences; try normalized
                    for key in (label, inst):
                        k2 = key.replace('-', '_')
                        if k2 in self.clip:
                            emb = self.clip[k2]
                            break
            # If still not found, try mapping via description.json uid -> clip (if clip keyed by uid)
            if emb is None:
                if label in self.desc:
                    uid = self.desc[label].get('uid')
                    if uid and isinstance(self.clip, dict) and uid in self.clip:
                        emb = self.clip[uid]
                elif inst in self.desc:
                    uid = self.desc[inst].get('uid')
                    if uid and isinstance(self.clip, dict) and uid in self.clip:
                        emb = self.clip[uid]
            # convert emb (which may be nested dict or tensor) into numeric numpy array
            def emb_to_array(e):
                # if dict of modalities, pick preferred
                if isinstance(e, dict):
                    for pref in ('text', 'video', 'point'):
                        if pref in e:
                            return emb_to_array(e[pref])
                    # otherwise take first value
                    first = next(iter(e.values()))
                    return emb_to_array(first)
                # torch tensor
                if 'torch' in str(type(e)):
                    try:
                        return e.detach().cpu().numpy()
                    except Exception:
                        return np.array(e)
                # numpy array
                if isinstance(e, np.ndarray):
                    return e
                # list/tuple of numbers or tensors
                if isinstance(e, (list, tuple)):
                    try:
                        return np.stack([emb_to_array(x) for x in e])
                    except Exception:
                        return np.array(e)
                # fallback scalar
                try:
                    return np.array(e)
                except Exception:
                    return None

            arr = None
            if emb is not None:
                arr = emb_to_array(emb)
            if arr is None:
                # pick a default sample from clip to infer shape
                sample_clip = None
                if isinstance(self.clip, dict):
                    sample_clip = next(iter(self.clip.values()))
                else:
                    sample_clip = self.clip
                arr_sample = emb_to_array(sample_clip)
                if arr_sample is None:
                    arr = np.zeros(512, dtype=np.float32)
                else:
                    arr = np.zeros_like(arr_sample, dtype=np.float32)

            # ensure numeric numpy array
            emb_np = np.asarray(arr, dtype=np.float32)
            # if 2D (e.g., stacked), flatten to 1D
            if emb_np.ndim > 1:
                emb_np = emb_np.reshape(-1)
            self.targets.append(torch.as_tensor(emb_np, dtype=torch.float32))

    def __len__(self):
        return int(self.eeg.shape[0])

    def __getitem__(self, idx):
        x = self.eeg[idx]  # (C, T)
        x = torch.as_tensor(x, dtype=torch.float32)
        y = self.targets[idx]
        return x, y

    def _preprocess_eeg(self, arr):
        # arr: (N, C, T)
        a = arr.astype(np.float32)
        # baseline removal: subtract mean over baseline_window if provided, else subtract mean over time
        if self.baseline_window is not None:
            start, end = self.baseline_window
            start = max(0, start)
            end = min(a.shape[-1], end)
            baseline = a[..., start:end].mean(axis=-1, keepdims=True)
        else:
            baseline = a.mean(axis=-1, keepdims=True)
        a = a - baseline
        if self.zscore:
            # per-channel zscore across time per trial
            mean = a.mean(axis=-1, keepdims=True)
            std = a.std(axis=-1, keepdims=True)
            std[std == 0] = 1.0
            a = (a - mean) / std
        return a


if __name__ == '__main__':
    # quick smoke test
    ds = EEGClipDataset(root=os.path.dirname(os.path.dirname(__file__)), sub='sub01')
    print('Loaded dataset with', len(ds), 'trials; sample shapes:', ds[0][0].shape, ds[0][1].shape)
