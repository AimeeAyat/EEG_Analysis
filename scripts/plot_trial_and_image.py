import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(__file__))

def find_occurrences(label):
    nl = np.load(os.path.join(ROOT, 'EEGdata', 'sub01', 'name_label.npy'), allow_pickle=True)
    inds = np.where(nl == label)[0]
    return inds, nl


def find_image_for_label(label):
    # robust search: try exact matches first, then search all image subfolders for filenames
    parts = label.split('_')
    if parts[0].isdigit():
        inst = '_'.join(parts[1:])
    else:
        inst = label
    exts = ['.png', '.jpg', '.jpeg']
    # first try common class folder heuristics
    # try folder named by combining non-numeric parts up to second token
    tokens = label.split('_')
    nonnum = '_'.join([t for t in tokens if not t.isdigit()])
    candidates = []
    # possible class dirs
    candidates.append(nonnum.split('_')[0])
    candidates.append(nonnum.split('_')[0] + '_' + nonnum.split('_')[1] if len(nonnum.split('_'))>1 else nonnum)
    # check these candidate dirs
    for cand in candidates:
        class_dir = os.path.join(ROOT, 'image', cand)
        if os.path.isdir(class_dir):
            for e in exts:
                p = os.path.join(class_dir, label + e)
                if os.path.exists(p):
                    return p
                p2 = os.path.join(class_dir, inst + e)
                if os.path.exists(p2):
                    return p2
            for fn in os.listdir(class_dir):
                if inst in fn or label in fn:
                    return os.path.join(class_dir, fn)

    # fallback: walk entire image directory and find any file containing inst or label
    img_root = os.path.join(ROOT, 'image')
    for dirpath, dirnames, filenames in os.walk(img_root):
        for fn in filenames:
            name, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                if inst in name or label in name:
                    return os.path.join(dirpath, fn)
    return None

def save_montage(flat_index, image_path=None, label=None, out_dir=None, subject='sub01'):
    proc = np.load(os.path.join(ROOT, 'EEGdata', subject, 'process_data_6s_100Hz.npy'), allow_pickle=False)
    eeg = proc[flat_index]  # (64,250)
    if out_dir is None:
        out_dir = os.path.join(ROOT, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    # compute fold/role/position for informative filename
    fold = flat_index // 24
    offset = flat_index % 24
    if offset < 16:
        role = 'train'
        r = offset // 2
        c = offset % 2
    else:
        role = 'test'
        offt = offset - 16
        r = offt // 4
        c = offt % 4
    safe_label = (label or '').replace(' ', '_') if label else ''
    out_path = os.path.join(out_dir, f'montage_flat{flat_index}_{safe_label}_fold{fold}_{role}{r}{c}.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    im = ax.imshow(eeg, aspect='auto', origin='lower', cmap='RdBu_r')
    ax.set_title(f'EEG epoch flat index {flat_index} (64x250)')
    ax.set_xlabel('time samples (250Hz)')
    ax.set_ylabel('channels')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    # resolve image path if not provided
    if image_path is None and label is not None:
        image_path = find_image_for_label(label)

    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            ax2.imshow(img)
            ax2.axis('off')
            ax2.set_title(os.path.basename(image_path))
        except Exception as e:
            ax2.text(0.5, 0.5, f'Could not open image: {e}', ha='center')
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'No image file available', ha='center')
        ax2.axis('off')

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='01_airplane_00')
    parser.add_argument('--image', default=r'G:\Rabia-Salman\EEG_Analysis\image\airplane\01_airplane_00.png')
    parser.add_argument('--subject', default='sub01')  # Add this
    args = parser.parse_args()

    inds, nl = find_occurrences(args.label)
    print('Label', args.label, 'occurs', len(inds), 'times; indices:', inds.tolist())
    if len(inds) == 0:
        print('Label not found in name_label.npy')
    else:
        idx = int(inds[0])
        out = save_montage(idx, args.image, subject=args.subject)  # Pass subject
        print('Saved montage to', out)
