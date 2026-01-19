import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json


def load_flat_train_test(root, sub='sub01'):
    subj = os.path.join(root, 'EEGdata', sub)
    train = np.load(os.path.join(subj, 'sub01_train_data_1s_250Hz.npy'), allow_pickle=False)
    test = np.load(os.path.join(subj, 'sub01_test_data_1s_250Hz.npy'), allow_pickle=False)
    flat_train = train.reshape(-1, train.shape[-2], train.shape[-1])
    flat_test = test.reshape(-1, test.shape[-2], test.shape[-1])
    name_label = np.load(os.path.join(subj, 'name_label.npy'), allow_pickle=True)
    train_len = flat_train.shape[0]
    test_len = flat_test.shape[0]
    labels_train = name_label[:train_len]
    labels_test = name_label[train_len:train_len + test_len]
    return flat_train, labels_train, flat_test, labels_test


def inst_from_label(label):
    parts = label.split('_')
    if parts[0].isdigit():
        return '_'.join(parts[1:])
    return label


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    Xtr, ytr, Xte, yte = load_flat_train_test(root)
    ytr_inst = [inst_from_label(l) for l in ytr]
    classes = sorted(list(set(ytr_inst)))
    cls2idx = {c:i for i,c in enumerate(classes)}
    ytr_idx = torch.tensor([cls2idx[c] for c in ytr_inst], dtype=torch.long)

    # filter test samples whose instance is in training classes
    yte_inst = [inst_from_label(l) for l in yte]
    mask = [c in cls2idx for c in yte_inst]
    coverage = np.mean(mask)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # simple EEGNet-like encoder
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
    from eegnet import get_model
    encoder = get_model(in_channels=64, out_dim=256).to(device)
    clf = nn.Linear(256, len(classes)).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(clf.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    ds = TensorDataset(Xtr_t, ytr_idx)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    encoder.train(); clf.train()
    for epoch in range(2):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            emb = encoder(xb)
            logits = clf(emb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        print('Epoch', epoch, 'done')

    # eval
    encoder.eval(); clf.eval()
    with torch.no_grad():
        emb_te = encoder(Xte_t.to(device)).cpu()
        logits = clf(emb_te.to(device)).cpu()
        preds = logits.argmax(dim=1).numpy()
    # compute accuracy only where ground truth instance in train classes
    yte_idxs = [cls2idx[c] if c in cls2idx else -1 for c in yte_inst]
    valid = [i for i,v in enumerate(yte_idxs) if v>=0]
    acc = np.mean([1 if preds[i]==yte_idxs[i] else 0 for i in valid]) if len(valid)>0 else 0.0
    print('Coverage of test instances in train classes:', coverage)
    print('Top1 instance accuracy (on covered samples):', acc)


if __name__ == '__main__':
    main()
