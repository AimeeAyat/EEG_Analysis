import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from eegnet import get_model
import json
from tqdm import tqdm


def load_flat_train_test(root, sub='sub01', eeg_key_train='sub01_train_data_1s_250Hz.npy', eeg_key_test='sub01_test_data_1s_250Hz.npy'):
    subj = os.path.join(root, 'EEGdata', sub)
    train = np.load(os.path.join(subj, eeg_key_train), allow_pickle=False)
    test = np.load(os.path.join(subj, eeg_key_test), allow_pickle=False)
    # flatten: train shape (72,8,2,64,250) -> (-1,64,250)
    flat_train = train.reshape(-1, train.shape[-2], train.shape[-1])
    flat_test = test.reshape(-1, test.shape[-2], test.shape[-1])
    # load labels from name_label.npy: first train_len labels correspond to flat_train
    name_label = np.load(os.path.join(subj, 'name_label.npy'), allow_pickle=True)
    train_len = flat_train.shape[0]
    test_len = flat_test.shape[0]
    labels_train = name_label[:train_len]
    labels_test = name_label[train_len:train_len + test_len]
    return flat_train, labels_train, flat_test, labels_test


def build_clip_db(root):
    clip = torch.load(os.path.join(root, 'clip_feature.pth'), map_location='cpu')
    # prefer text modality
    keys = []
    mats = []
    for k, v in clip.items():
        keys.append(k)
        if isinstance(v, dict):
            if 'text' in v:
                mats.append(v['text'].detach().cpu().numpy())
            elif 'video' in v:
                mats.append(v['video'].detach().cpu().numpy())
            else:
                mats.append(next(iter(v.values())).detach().cpu().numpy())
        else:
            mats.append(np.asarray(v))
    mats = np.stack(mats, axis=0)  # (M, D)
    # normalize
    nm = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-8)
    return keys, nm


def embed_cosine(preds, gallery):
    # preds: (N, D), gallery: (M, D) normalized
    p = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(p, gallery.T)
    return sims


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    print('Loading flattened train/test for sub01')
    X_train, y_train, X_test, y_test = load_flat_train_test(root)
    print('Train/test shapes:', X_train.shape, X_test.shape)

    # build clip DB
    keys, gallery = build_clip_db(root)
    print('Clip DB size:', gallery.shape)

    # prepare tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    # For targets, map label -> clip embedding vector from clip_feature.pth (text)
    clip = torch.load(os.path.join(root, 'clip_feature.pth'), map_location='cpu')
    def get_emb_from_label(label):
        parts = label.split('_')
        inst = '_'.join(parts[1:]) if parts[0].isdigit() else label
        if label in clip:
            v = clip[label]
        elif inst in clip:
            v = clip[inst]
        else:
            # fallback: try description.json mapping
            desc = json.load(open(os.path.join(root, 'description.json'), 'r', encoding='utf-8'))
            if label in desc:
                uid = desc[label].get('uid')
                if uid and uid in clip:
                    v = clip[uid]
                else:
                    v = next(iter(clip.values()))
            else:
                v = next(iter(clip.values()))
        if isinstance(v, dict):
            if 'text' in v:
                v = v['text']
            elif 'video' in v:
                v = v['video']
            else:
                v = next(iter(v.values()))
        return torch.tensor(v.detach().cpu().numpy(), dtype=torch.float32)

    Ytr = torch.stack([get_emb_from_label(l) for l in y_train], dim=0)
    Yte = torch.stack([get_emb_from_label(l) for l in y_test], dim=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(in_channels=64, timesteps=250, out_dim=Ytr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # training loop - small number of epochs for quick test
    batch_size = 32
    tr_ds = TensorDataset(Xtr, Ytr)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(3):
        print('Epoch', epoch)
        for xb, yb in tqdm(tr_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch', epoch, 'done')

    # evaluation: predict embeddings for test set and compute retrieval accuracy
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, Xte.shape[0], batch_size):
            xb = Xte[i:i+batch_size].to(device)
            p = model(xb).cpu().numpy()
            preds.append(p)
        preds = np.concatenate(preds, axis=0)

    sims = embed_cosine(preds, gallery)  # (Ntest, M)
    top1 = np.argmax(sims, axis=1)
    # map gallery keys to instance names (keys list)
    pred_names = [keys[i] for i in top1]
    # compute top1/top5 accuracy comparing predicted instance (pred_names) to ground truth instance
    def inst_from_label(label):
        parts = label.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return label

    y_inst = [inst_from_label(l) for l in y_test]
    top1_acc = np.mean([1 if pred_names[i] == y_inst[i] else 0 for i in range(len(y_inst))])
    # top5
    top5_idx = np.argsort(-sims, axis=1)[:, :5]
    top5_acc = np.mean([1 if y_inst[i] in [keys[j] for j in top5_idx[i]] else 0 for i in range(len(y_inst))])
    print(f'Test Top1 acc: {top1_acc:.4f}  Top5 acc: {top5_acc:.4f}')

    # Save example prompts (ground-truth description and predicted description) for first 20 test samples
    desc = json.load(open(os.path.join(root, 'description.json'), 'r', encoding='utf-8'))
    examples = []
    for i in range(min(20, len(y_test))):
        gt = y_inst[i]
        pred = pred_names[i]
        gt_desc = desc.get(gt, {}).get('description', gt)
        pred_desc = desc.get(pred, {}).get('description', pred)
        examples.append({'index': i, 'gt_instance': gt, 'gt_desc': gt_desc, 'pred_instance': pred, 'pred_desc': pred_desc})

    outp = os.path.join(root, 'results_sub01_examples.json')
    json.dump({'top1': float(top1_acc), 'top5': float(top5_acc), 'examples': examples}, open(outp, 'w', encoding='utf-8'), indent=2)
    print('Saved examples to', outp)


if __name__ == '__main__':
    main()
