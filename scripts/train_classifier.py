import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.eegnet import get_model
from tqdm import tqdm


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


def label_to_category(label):
    parts = label.split('_')
    if parts[0].isdigit():
        return parts[1]
    return parts[0]


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    Xtr, ytr, Xte, yte = load_flat_train_test(root)
    cat_list = sorted(list({label_to_category(l) for l in np.concatenate([ytr, yte])}))
    cat2idx = {c:i for i,c in enumerate(cat_list)}
    ytr_idx = np.array([cat2idx[label_to_category(l)] for l in ytr])
    yte_idx = np.array([cat2idx[label_to_category(l)] for l in yte])

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    Ytr_t = torch.tensor(ytr_idx, dtype=torch.long)
    Yte_t = torch.tensor(yte_idx, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(in_channels=64, timesteps=250, out_dim=len(cat_list)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    tr_ds = TensorDataset(Xtr_t, Ytr_t)
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)

    for epoch in range(3):
        model.train()
        for xb, yb in tqdm(tr_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch', epoch, 'done')

    # evaluate
    model.eval()
    with torch.no_grad():
        preds = model(Xte_t.to(device)).cpu()
        acc = (preds.argmax(dim=1) == Yte_t).float().mean().item()
    print('Classifier test accuracy (category-level):', acc)


if __name__ == '__main__':
    main()
