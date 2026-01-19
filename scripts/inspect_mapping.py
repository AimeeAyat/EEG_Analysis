import numpy as np
import os

ROOT = r"g:\\Rabia-Salman\\EEG_Analysis"
nl = np.load(os.path.join(ROOT, 'EEGdata', 'sub01', 'name_label.npy'), allow_pickle=True)
proc = np.load(os.path.join(ROOT, 'EEGdata', 'sub01', 'process_data_1s_250Hz.npy'), allow_pickle=False)

def locate_instance(instance_name):
    # returns list of (flat_index, fold, role, pos_in_fold) for every occurrence
    matches = np.where(nl == instance_name)[0]
    out = []
    for idx in matches:
        fold = idx // 24
        off = idx % 24
        if off < 16:
            role = 'train'
            # train layout is (8,2)
            r = off // 2
            c = off % 2
            pos = (r, c)
        else:
            role = 'test'
            offt = off - 16
            # test layout is (2,4)
            r = offt // 4
            c = offt % 4
            pos = (r, c)
        out.append((int(idx), int(fold), role, pos))
    return out


# Example: list occurrences of '01_airplane_06' (replace with any label string)
inst = '01_airplane_09'
print('Total occurrences of', inst, ':', np.sum(nl == inst))
print(locate_instance(inst))

# Example: get EEG for a particular flat index and show shape
i = 820  # pick any flat trial index
print('label at flat index', i, '->', nl[i])
print('proc[i] shape:', proc[i].shape)  # (64, 250)

