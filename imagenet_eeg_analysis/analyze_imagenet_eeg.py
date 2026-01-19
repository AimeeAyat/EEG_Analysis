"""Analyze ImageNet EEG data in data/ and produce image->EEG mapping CSV.

This script is defensive: it tries torch.load and pickle, prints summaries,
and attempts heuristics to map images to EEG trial indices using
`block_splits_by_image_all.pth` or `captions_with_bbox_data.pth`.

Output: `imagenet_eeg_analysis/output/image_eeg_mapping.csv` and
`imagenet_eeg_analysis/output/summary.json`.
"""
import os
import json
import csv
import pickle
from pprint import pprint

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data')
OUTDIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTDIR, exist_ok=True)

FILES = {
    'eeg': os.path.join(DATA, 'eeg_55_95_std.pth'),
    'captions': os.path.join(DATA, 'captions_with_bbox_data.pth'),
    'blocks': os.path.join(DATA, 'block_splits_by_image_all.pth'),
    'imagenet_dir': os.path.join(DATA, 'imageNet_images'),
}


def robust_load(path):
    if not os.path.exists(path):
        return None, 'MISSING'
    # try torch
    try:
        import torch
        obj = torch.load(path, map_location='cpu')
        return obj, 'torch'
    except Exception:
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return obj, 'pickle'
        except Exception as e:
            return str(e), 'ERROR'


def small_info(obj):
    t = type(obj).__name__
    info = {'type': t}
    try:
        import numpy as np
        if hasattr(obj, 'shape'):
            info['shape'] = tuple(obj.shape)
        if isinstance(obj, dict):
            info['len'] = len(obj)
            info['sample_keys'] = list(obj.keys())[:10]
            if info['sample_keys']:
                v = obj[info['sample_keys'][0]]
                info['sample_value_type'] = type(v).__name__
        elif isinstance(obj, (list, tuple)):
            info['len'] = len(obj)
            if len(obj) > 0:
                info['first_type'] = type(obj[0]).__name__
    except Exception:
        pass
    return info


def try_map(blocks, captions, eeg):
    """Attempt mapping heuristics and return dict image->list(indices)."""
    mapping = {}
    # If blocks is dict-like mapping image->indices
    if isinstance(blocks, dict):
        for k, v in blocks.items():
            try:
                mapping[str(k)] = list(v)
            except Exception:
                try:
                    mapping[str(k)] = [int(x) for x in v]
                except Exception:
                    mapping[str(k)] = []
        return mapping, 'blocks_dict'
    # If blocks is a list/tuple of pairs (image, indices)
    if isinstance(blocks, (list, tuple)):
        # guess pairs
        for item in blocks:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = item[0]
                v = item[1]
                mapping[str(k)] = list(v) if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)) else [v]
        if mapping:
            return mapping, 'blocks_list_pairs'
    # If captions holds instance lists that include ids aligning with EEG trials,
    # attempt to build mapping from captions structure
    if isinstance(captions, dict):
        # try keys as images -> values contain 'trials' or index lists
        for k, v in captions.items():
            if isinstance(v, dict) and ('trial' in v or 'indices' in v):
                idxs = v.get('trial') or v.get('indices') or v.get('trials')
                try:
                    mapping[str(k)] = list(idxs)
                except Exception:
                    pass
        if mapping:
            return mapping, 'captions_dict_trials'
    # fallback: if eeg is an array and captions is sequence of labels with same length
    if hasattr(eeg, 'shape') and isinstance(captions, (list, tuple)):
        # captions might be sequence aligned to eeg trials where each item is instance name
        if len(captions) == eeg.shape[0]:
            for idx, inst in enumerate(captions):
                mapping.setdefault(str(inst), []).append(int(idx))
            return mapping, 'captions_aligned_to_eeg'
    return mapping, 'no_mapping'


def main():
    loaded = {}
    for name, path in [('eeg', FILES['eeg']), ('captions', FILES['captions']), ('blocks', FILES['blocks'])]:
        obj, method = robust_load(path)
        loaded[name] = {'obj': obj, 'method': method}
        loaded[name]['info'] = small_info(obj)

    # list imagenet wnids
    imagenet_dirs = []
    if os.path.exists(FILES['imagenet_dir']):
        imagenet_dirs = sorted([d for d in os.listdir(FILES['imagenet_dir']) if os.path.isdir(os.path.join(FILES['imagenet_dir'], d))])

    # Try mapping
    mapping, reason = try_map(loaded['blocks']['obj'], loaded['captions']['obj'], loaded['eeg']['obj'])

    # If mapping empty, try alternative: if captions is a tuple containing (instances, labels)
    if not mapping and isinstance(loaded['captions']['obj'], (list, tuple)) and len(loaded['captions']['obj']) >= 2:
        cap = loaded['captions']['obj']
        # heuristics: common formats: (instances, captions) or (filenames, bboxes)
        # If first element is list of instance names and its length equals number of blocks entries
        try:
            if hasattr(cap[0], '__len__') and hasattr(loaded['eeg']['obj'], 'shape') and len(cap[0]) == loaded['eeg']['obj'].shape[0]:
                for idx, inst in enumerate(cap[0]):
                    mapping.setdefault(str(inst), []).append(int(idx))
                reason = 'captions_tuple_aligned'
        except Exception:
            pass

    # Prepare CSV rows
    csv_path = os.path.join(OUTDIR, 'image_eeg_mapping.csv')
    summary = {'loaded': {k: v['info'] for k, v in loaded.items()}, 'mapping_reason': reason}
    rows = []
    if mapping:
        for img, idxs in mapping.items():
            eeg_shape = None
            try:
                if hasattr(loaded['eeg']['obj'], 'shape'):
                    eeg_shape = tuple(loaded['eeg']['obj'].shape)
            except Exception:
                pass
            rows.append({'image': img, 'trial_indices': ';'.join(map(str, idxs)), 'n_trials': len(idxs), 'eeg_shape': eeg_shape})
    else:
        # no mapping found; fall back to saving captions summary
        rows.append({'image': 'NO_MAPPING_FOUND', 'trial_indices': '', 'n_trials': 0, 'eeg_shape': small_info(loaded['eeg']['obj']).get('shape')})

    # write CSV and summary
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'trial_indices', 'n_trials', 'eeg_shape'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(os.path.join(OUTDIR, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Wrote', csv_path)
    print('Summary reason:', reason)
    pprint(summary)


if __name__ == '__main__':
    main()
