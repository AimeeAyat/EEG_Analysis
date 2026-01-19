import os
from plot_trial_and_image import save_montage
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SUB = 'sub02'
OUT_DIR = os.path.join(ROOT, 'outputs', 'montages_sub02_6s')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    nl = np.load(os.path.join(ROOT, 'EEGdata', SUB, 'name_label.npy'), allow_pickle=True)
    n = len(nl)
    print('Total trials to process:', n)
    missing_images = 0
    # build list of (original_idx, label) and sort by label so same-class images are processed together
    pairs = [(i, str(nl[i])) for i in range(n)]
    pairs.sort(key=lambda x: x[1])
    for seq, (idx, label) in enumerate(pairs):
        # determine output filename via same logic as save_montage; skip if exists
        fold = idx // 24
        offset = idx % 24
        if offset < 16:
            role = 'train'
            r = offset // 2
            c = offset % 2
        else:
            role = 'test'
            offt = offset - 16
            r = offt // 4
            c = offt % 4
        safe_label = label.replace(' ', '_')
        # prefix filename with the label so files of same class are adjacent in directory listings
        out_name = f'{safe_label}_montage_flat{seq}_fold{fold}_{role}{r}{c}.png'
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            if seq % 100 == 0:
                print('Skipping existing (seq)', seq, 'orig_idx', idx)
            continue
        # call save_montage with original trial index (it expects the original index)
        try:
            saved = save_montage(idx, image_path=None, label=label, out_dir=OUT_DIR, subject=SUB)
            # save_montage may write its own file; if it returns a path, and it's different
            # from our desired name, we won't try to rename here to avoid side-effects.
        except Exception as e:
            print('Error processing', idx, e)
        if seq % 100 == 0:
            print('Processed', seq, 'of', n)
    print('Done. Missing image files for', missing_images, 'trials. Montages in', OUT_DIR)

if __name__ == '__main__':
    main()
