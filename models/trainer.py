import torch.cuda.amp as amp
from tqdm import tqdm
from collections import defaultdict
import logging
import time

def get_train_logger(log_dir='./logs'):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return (img[:,:,::-1] * 255).astype(np.uint8)

def compute_video_metrics(preds):
    """
    preds: List of (fname, uid, label, pred)
    """
    df = pd.DataFrame(preds, columns=['fname', 'uid', 'label', 'pred'])
    gdf =  df.groupby('uid')[['label', 'pred']].mean().reset_index()
    return (gdf['label'] == ( gdf['pred'] >= 0.5).astype(int)).mean()

class Trainer:
    def __init__(self, model, optimizer, criterion=None, scheduler=None, cfg=None):
        self.model = model
        self.model_name = model.__class__.__name__
        self.optim = optimizer
        self.cfg = cfg
        self.scheduler = scheduler
        self.best_score = 100
        self.criterion = criterion
        self.scaler = amp.GradScaler()
        if not isinstance(cfg.device, str):
            self.device = cfg.device
        else:
            self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
        
        print("device", self.device, type(self.device))

    def init_logger(self, log_dir):
        self.logger = get_train_logger(log_dir)

    def train_epoch(self, loader, epoch=0):
        """
        Train one epoch
        Args:
            loader: data loader
            optim: optimizer
            loss_func: loss function
            device: device
        Returns([dict]): metric score, e.g: {'f1': 0.99}
        """
        self.model.train()

        bar = tqdm(loader)
        scores = defaultdict(list)

        for batch_idx, sample in enumerate(bar):
            images = sample['image']
            labels = sample['label']
            # uids = sample['uid']

            if epoch >= self.cfg.warmup_epochs and np.random.rand() <= self.cfg.mixup:
                shuffle_indices = torch.randperm(images.size(0))
                indices = torch.arange(images.size(0))
                lam = np.clip(np.random.beta(1.0, 1.0), 0.35, 0.65)
                images = lam * images + (1 - lam) * images[shuffle_indices, :]
                labels = lam * labels + (1 - lam) * labels[shuffle_indices, :]

            images = images.to(self.device)
            labels = labels.to(self.device).float()
    
            do_update = ((batch_idx + 1) % self.cfg.gradient_accum_steps == 0) or (batch_idx + 1 == len(loader))
            
            with amp.autocast():
                pred = self.model(images)
                loss = self.criterion(pred, labels)
                self.scaler.scale(loss).backward()
                if do_update:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
        
            # Compute metric score
            loss = loss.item() * self.cfg.gradient_accum_steps
            msg_loss = f"loss: {loss:.4f}"
            bar.set_description(msg_loss)
            scores["loss"].append(loss)
        scores = {k: (np.mean(v) if isinstance(v, list) else v) for k, v in scores.items()}
        
        return scores

    def val_epoch(self, loader, epoch=0):
        """
        Train one epoch
        Args:
            loader: data loader
            optim: optimizer
            loss_func: loss function
            device: device
        Returns([dict]): metric score, e.g: {'f1': 0.99}
        """
        self.model.eval()

        bar = tqdm(loader)
        scores = defaultdict(list)
        preds = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(bar):
                images = sample['image']
                labels = sample['label']
                uids = sample['uid']
                fnames = sample['fname']
                images = images.to(self.device)
                labels = labels.to(self.device).float()
            
                with amp.autocast():
                    pred = self.model(images)
                    loss = self.criterion(pred, labels)

                # Compute metric score
                pred = torch.sigmoid(pred)
                pred = pred.cpu().numpy()
                cls_pred = (pred >= 0.5).astype(int)
                loss = loss.item() * self.cfg.gradient_accum_steps
                labels_np = labels.cpu().numpy()
                acc = (cls_pred == labels_np).mean()
                msg_loss = f"loss: {loss:.4f}, Acc: {acc:.4f}"
                bar.set_description(msg_loss)
                scores["loss"].append(loss)
                scores["acc"].append(acc)

                # Save pred
                for fname, uid, label, _pred in zip(
                        fnames, uids, labels_np, pred
                    ):
                    preds.append([fname, uid, label[0], _pred[0]])

        scores = {k: (np.mean(v) if isinstance(v, list) else v) for k, v in scores.items()}
        
        return scores, preds

    def train(self, train_loader, val_loader):
        """Train process"""
        cfg = self.cfg
        output_dir = cfg.outdir
        epochs = cfg.epochs
        weight_dir = os.path.join(output_dir, "weights")
        log_dir = os.path.join(output_dir, 'logs')
        log_example_dir = os.path.join(log_dir, 'train_examples')
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(log_example_dir, exist_ok=True)

        self.init_logger(log_dir)
        early_stop_counter = 0
        best_ckp = os.path.join(weight_dir, f'{self.model_name}_best.pth')

        # Save some example
        for sample in train_loader:
            imgs = sample['image']
            labels = sample['label']
            uids = sample['uid']
            for i in range(len(imgs)):
                img, label, uid = imgs[i], labels[i], uids[i]
                img = img.cpu().numpy().transpose(1, 2 ,0)
                img = denorm(img)
                text = f'{uid} - {label}'
                save_path = os.path.join(log_example_dir, f"example_{i}.jpg")
                put_text(img, text, save_path)
                # cv2.imwrite(save_path, img)
                if i >= 8:
                    break
            break
        
        # load pretraineds
        start_epoch = 0
        last_ckp = os.path.join(weight_dir, f'{self.model_name}_last.pth')
        if cfg.resume:
            if os.path.exists(last_ckp):
                ckp = torch.load(last_ckp, map_location='cpu')
                # self.optim.load_state_dict(ckp['optim'])
                # self.scheduler.load_state_dict(ckp['scheduler'])
                # load_my_state_dict(self.model, ckp)
                self.model.load_state_dict(ckp)
                start_epoch = 0
                self.logger.info(f"Resume training from epoch {start_epoch}")
                print(f"Resume training from epoch {start_epoch}")
            else:
                self.logger.info(f"{last_ckp} not found, train from scratch")
                print(f"{last_ckp} not found, train from scratch")

        # Train
        start = time.time()
        self.model.to(self.device)

        for epoch in range(start_epoch, epochs):
            train_scores = self.train_epoch(train_loader, epoch=epoch)

            do_valid = epoch % 1 == 0

            if do_valid:
                val_scores, preds = self.val_epoch(val_loader)
                print(preds)
                val_loss = val_scores["loss"]
                val_acc = val_scores["acc"]
                val_acc_vid = compute_video_metrics(preds)
            lr = self.optim.param_groups[0]["lr"]
            msg = f"Epoch {epoch + 1}/{epochs} (lr={lr:.5f})\nTrain "
            msg += ", ".join([f"{k}: {v:.5f}" for k, v in train_scores.items()])

            if do_valid:
                if val_loss:
                    msg += f"\nValid: loss {val_loss:.4f}, Acc {val_acc:.4f}, Acc_per_video {val_acc_vid:.4f}"

            self.logger.info(msg)
            # print(msg)
            if self.scheduler is not None:
                self.scheduler.step()

            if do_valid:
                score = val_loss
                if epoch > 1:
                    if score < self.best_score:
                        m = f"Val Loss improved from {self.best_score:.4f} -> {score:.4f}, save model"
                        self.logger.info(m)
                        # print(m)
                        self.best_score = score
                        early_stop_counter = 0
                        torch.save(self.model.state_dict(), best_ckp)
                    else:
                        early_stop_counter += 1

                # if early_stop_counter >= 3:
                #     print("Model doest not improve anymore, stop")
                #     break

            # Save last epoch
            torch.save(self.model.state_dict(), last_ckp)

        self.logger.info(f"Training is completed, elapsed: {(time.time() - start):.3f}s")
        return best_ckp, val_loss