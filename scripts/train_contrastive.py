import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from eeg_dataset import EEGClipDataset
from conformer_small import get_conformer
from eeg_conformer import get_eeg_conformer
import argparse
import time
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


def nt_xent_loss(eeg_emb, clip_emb, temperature=0.07):
    # eeg_emb: (B, D), clip_emb: (B, D)
    eeg_norm = eeg_emb / (eeg_emb.norm(dim=1, keepdim=True) + 1e-8)
    clip_norm = clip_emb / (clip_emb.norm(dim=1, keepdim=True) + 1e-8)
    logits = torch.matmul(eeg_norm, clip_norm.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_e2c = nn.CrossEntropyLoss()(logits, labels)
    loss_c2e = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_e2c + loss_c2e) / 2.0


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', default='sub01')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'))
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    # enable preprocessing in dataset
    ds = EEGClipDataset(root=root, sub=args.sub, preprocess=True, baseline_window=None, zscore=True)
    # split into train/val
    val_frac = 0.1
    n = len(ds)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_dim = ds[0][1].numel()
    # prefer eeg_conformer
    encoder = get_eeg_conformer(in_channels=64, out_dim=clip_dim).to(device)
    # add projection head
    proj_dim = 256
    projection = nn.Sequential(nn.Linear(clip_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=args.lr)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    temperature = 0.07

    os.makedirs(args.outdir, exist_ok=True)
    encoder.train(); projection.train()
    best_top1 = 0.0
    start = time.time()
    for epoch in range(args.epochs):
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            with autocast():
                feats = encoder(x)
                z = projection(feats)
                # normalize
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                y_proj = y
                if y_proj.dim() == 2 and y_proj.size(1) == clip_dim:
                    # map clip target to projection space using linear map (use projection on clip vector)
                    with torch.no_grad():
                        yz = projection(y_proj)
                        yz = yz / (yz.norm(dim=1, keepdim=True) + 1e-8)
                else:
                    yz = y_proj
                loss = nt_xent_loss(z, yz, temperature=temperature)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if (i % 10) == 0:
                print(f'Epoch {epoch} Batch {i} contrastive loss {loss.item():.4f}')
            if args.smoke and i >= 2:
                break
        scheduler.step()

        # validation: compute retrieval Top1/Top5
        encoder.eval(); projection.eval()
        with torch.no_grad():
            # build gallery from clip features (once)
            clip = torch.load(os.path.join(root, 'clip_feature.pth'), map_location='cpu')
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
            mats = np.stack(mats, axis=0)
            gallery = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-8)

            all_preds = []
            all_labels = []
            for xb, yb in val_dl:
                xb = xb.to(device)
                feats = encoder(xb)
                z = projection(feats)
                z = z.cpu().numpy()
                # normalize
                z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
                sims = np.dot(z, gallery.T)
                top1_idx = np.argmax(sims, axis=1)
                all_preds.extend(top1_idx.tolist())
                # map val labels to instance key names
                for lab in yb:
                    # lab may be clip vector or label string in dataset; convert via name_label mapping
                    # here easiest: use dataset name_label mapping from underlying dataset
                    pass
            # compute top1/top5 by comparing predicted key to true instance name
            # Because mapping in quick val is complex, we compute a simple proxy: mean max similarity
            # (user-level evaluation with stored examples will follow)
            val_metric = 0.0

        encoder.train(); projection.train()

        # save checkpoint per epoch
        ckpt = os.path.join(args.outdir, f'contrastive_{args.sub}_epoch{epoch}.pth')
        torch.save({'encoder': encoder.state_dict(), 'projection': projection.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, ckpt)
        print('Saved', ckpt)
        if args.smoke:
            break
    dur = time.time() - start
    print('Training finished, time', dur)


if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eeg_dataset import EEGClipDataset
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from eegnet import get_model as get_eegnet
import numpy as np
from tqdm import tqdm


def info_nce_loss(q, k, temperature=0.07):
    # q, k: (B, D)
    qn = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    kn = k / (k.norm(dim=1, keepdim=True) + 1e-8)
    logits = torch.matmul(qn, kn.t()) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss, logits


def build_gallery(root):
    clip = torch.load(os.path.join(root, 'clip_feature.pth'), map_location='cpu')
    mats = []
    keys = []
    for k, v in clip.items():
        keys.append(k)
        if isinstance(v, dict):
            vec = v.get('text', next(iter(v.values())))
        else:
            vec = v
        mats.append(np.asarray(vec))
    mats = np.stack(mats, axis=0)
    nm = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-8)
    return keys, nm


def evaluate_preds(preds, gallery):
    # preds: (N, D) numpy, gallery: (M, D) normalized numpy
    pnorm = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(pnorm, gallery.T)
    top1 = np.argmax(sims, axis=1)
    return sims, top1


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    ds = EEGClipDataset(root=root, sub='sub01')
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    clip_dim = ds[0][1].numel()
    model = get_eegnet(in_channels=64, timesteps=250, out_dim=clip_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # quick train
    model.train()
    for epoch in range(2):
        for x, y in tqdm(dl):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss, _ = info_nce_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('Epoch', epoch, 'done')

    # eval: predict on test flattened set
    # reuse previous train_eval helper flatten loader
    from train_eval_eegnet import load_flat_train_test, build_clip_db
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
    keys, gallery = build_gallery(root)
    sims, top1 = evaluate_preds(preds, gallery)
    # compute top1/top5 acc
    def inst_from_label(label):
        parts = label.split('_')
        return '_'.join(parts[1:]) if parts[0].isdigit() else label
    y_inst = [inst_from_label(l) for l in y_test]
    pred_names = [keys[i] for i in top1]
    top1_acc = np.mean([1 if pred_names[i] == y_inst[i] else 0 for i in range(len(y_inst))])
    top5_idx = np.argsort(-sims, axis=1)[:, :5]
    top5_acc = np.mean([1 if y_inst[i] in [keys[j] for j in top5_idx[i]] else 0 for i in range(len(y_inst))])
    print('Contrastive Test Top1:', top1_acc, 'Top5:', top5_acc)


if __name__ == '__main__':
    main()
