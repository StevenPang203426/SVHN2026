"""
Training engine with wandb, MixUp, multiple LR schedulers.
"""
import os, time, torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets.dataset import DigitsDataset, mixup_data
from losses import build_loss
from models import build_model


class Trainer:
    def __init__(self, cfg, data_dir, val=True):
        self.cfg = cfg
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("[Trainer] Device: %s" % self.device)
        print("[Trainer] Model=%s Loss=%s Aug=%s" % (cfg.model_name, cfg.loss_type, cfg.aug_level))

        pw = cfg.num_workers > 0
        self.train_set = DigitsDataset(cfg, data_dir, mode='train')
        self.train_loader = DataLoader(
            self.train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=pw, drop_last=True,
            collate_fn=self.train_set.collect_fn)

        self.val_loader = None
        if val:
            vs = DigitsDataset(cfg, data_dir, mode='val')
            self.val_loader = DataLoader(
                vs, batch_size=cfg.batch_size,
                num_workers=cfg.num_workers, pin_memory=True,
                drop_last=False, persistent_workers=pw)

        self.model = build_model(cfg).to(self.device)
        self.criterion = build_loss(cfg).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr,
                              betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        self.lr_scheduler = self._build_scheduler()
        self.best_acc = 0.0
        self.best_checkpoint_path = ""
        self.wandb_run = None
        if cfg.use_wandb:
            self._init_wandb()
        if cfg.pretrained is not None:
            self.load_model(cfg.pretrained)
            if self.val_loader is not None:
                acc, _ = self.eval()
                self.best_acc = acc
                print("[Trainer] Loaded %s, Acc: %.2f%%" % (cfg.pretrained, acc*100))

    def _build_scheduler(self):
        c = self.cfg
        if c.scheduler == "cosine_warm":
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=c.T_0, T_mult=c.T_mult, eta_min=1e-6)
        elif c.scheduler == "step":
            return MultiStepLR(self.optimizer, milestones=c.milestones, gamma=c.gamma)
        elif c.scheduler == "one_cycle":
            return OneCycleLR(self.optimizer, max_lr=c.lr, epochs=c.epochs,
                              steps_per_epoch=len(self.train_loader), pct_start=0.1)
        raise ValueError("Unknown scheduler: %s" % c.scheduler)

    def _init_wandb(self):
        try:
            import wandb
            self.wandb_run = wandb.init(project=self.cfg.wandb_project,
                name=self.cfg.experiment_name,
                config={k: v for k, v in vars(self.cfg).items() if not k.startswith('_')})
        except ImportError:
            print("[WARN] wandb not installed"); self.cfg.use_wandb = False

    def _log(self, m, step=None):
        if self.wandb_run:
            import wandb; wandb.log(m, step=step)

    def train(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            tl, ta = self.train_epoch(epoch)
            self._log({"train/loss": tl, "train/acc": ta,
                        "lr": self.optimizer.param_groups[0]['lr']}, step=epoch)
            if (epoch+1) % self.cfg.eval_interval == 0 and self.val_loader:
                va, vt = self.eval()
                self._log({"val/acc": va*100, "val/time": vt}, step=epoch)
                print("[Epoch %d] Val Acc: %.2f%% (%.1fs)" % (epoch+1, va*100, vt))
                if va > self.best_acc:
                    save_dir = os.path.join(self.cfg.checkpoints, self.cfg.experiment_name)
                    os.makedirs(save_dir, exist_ok=True)
                    fn = "ep%d-acc%.2f.pth" % (epoch+1, va*100)
                    sp = os.path.join(save_dir, fn)
                    self.save_model(sp)
                    print("  Best saved: %s" % sp)
                    self.best_acc = va
                    self.best_checkpoint_path = sp
        if self.wandb_run:
            import wandb; wandb.finish()
        print("\n[Trainer] Done. Best: %.2f%%" % (self.best_acc*100))
        return self.best_acc

    def train_epoch(self, epoch):
        self.model.train()
        tl, corr, ns = 0.0, 0, 0
        tbar = tqdm(self.train_loader, desc="Epoch %d/%d" % (epoch+1, self.cfg.epochs))
        for i, (img, label) in enumerate(tbar):
            img, label = img.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            if self.cfg.use_mixup:
                img, la, lb, lam = mixup_data(img, label, self.cfg.mixup_alpha)
                pred = self.model(img)
                loss = sum(lam*self.criterion(pred[j],la[:,j])+(1-lam)*self.criterion(pred[j],lb[:,j]) for j in range(4))
            else:
                pred = self.model(img)
                loss = sum(self.criterion(pred[j], label[:,j]) for j in range(4))
            tl += loss.item(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            if self.cfg.scheduler == "one_cycle":
                self.lr_scheduler.step()
            elif (i+1) % self.cfg.print_interval == 0:
                self.lr_scheduler.step()
            if not self.cfg.use_mixup:
                t = torch.stack([pred[j].argmax(1)==label[:,j] for j in range(4)], dim=1)
                corr += torch.all(t, dim=1).sum().item()
            ns += img.size(0)
            al = tl/(i+1)
            aa = corr/ns*100 if not self.cfg.use_mixup and ns>0 else 0
            tbar.set_postfix(loss="%.3f"%al, acc="%.2f%%"%aa)
        return tl/max(len(self.train_loader),1), corr/ns*100 if ns>0 else 0

    def eval(self):
        self.model.eval(); corr, ns = 0, 0; t0 = time.time()
        with torch.no_grad():
            for img, label in tqdm(self.val_loader, desc='Val'):
                img, label = img.to(self.device), label.to(self.device)
                pred = self.model(img)
                t = torch.stack([pred[j].argmax(1)==label[:,j] for j in range(4)], dim=1)
                corr += torch.all(t, dim=1).sum().item(); ns += img.size(0)
        self.model.train()
        return corr/ns, time.time()-t0

    def save_model(self, path, save_opt=False):
        d = {'model': self.model.state_dict(), 'best_acc': self.best_acc,
             'config': self.cfg.experiment_name}
        if save_opt: d['opt'] = self.optimizer.state_dict()
        torch.save(d, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        print("[Trainer] Loaded %s" % path)
