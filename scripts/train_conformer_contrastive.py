import os
import torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGClipDataset
from models.eeg_conformer import get_conformer
import numpy as np
from tqdm import tqdm
import torch.nn as nn


def info_nce_loss(q, k, temperature=0.07):
    qn = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    kn = k / (k.norm(dim=1, keepdim=True) + 1e-8)
    logits = torch.matmul(qn, kn.t()) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    ds = EEGClipDataset(root=root, sub='sub01')
    dl = DataLoader(ds, batch_size=48, shuffle=True)
    clip_dim = ds[0][1].numel()
    model = get_conformer(in_channels=64, timesteps=250, out_dim=clip_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)

    model.train()
    for epoch in range(2):
        for x, y in tqdm(dl):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = info_nce_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch', epoch, 'done')

    # quick eval based on train_eval helper
    from train_eval_eegnet import load_flat_train_test, build_clip_db, embed_cosine
    _, _, X_test, y_test = load_flat_train_test(root)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 64):
            xb = Xte[i:i+64].to(device)
            p = model(xb).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    keys, gallery = build_clip_db(root)
    sims = embed_cosine(preds, gallery)
    top1 = np.argmax(sims, axis=1)
    def inst_from_label(label):
        parts = label.split('_')
        return '_'.join(parts[1:]) if parts[0].isdigit() else label
    y_inst = [inst_from_label(l) for l in y_test]
    pred_names = [keys[i] for i in top1]
    top1_acc = np.mean([1 if pred_names[i] == y_inst[i] else 0 for i in range(len(y_inst))])
    top5_idx = np.argsort(-sims, axis=1)[:, :5]
    top5_acc = np.mean([1 if y_inst[i] in [keys[j] for j in top5_idx[i]] else 0 for i in range(len(y_inst))])
    print('Conformer contrastive Test Top1:', top1_acc, 'Top5:', top5_acc)


if __name__ == '__main__':
    main()
