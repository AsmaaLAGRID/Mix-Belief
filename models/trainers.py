from .data import build_dataset
import logging
from .models import TextBERT 
from .losses import build_loss, edl_loss, exp_evidence, combined_loss, combined_loss_switch
from .utils import (seed_all, 
                compute_metrics, 
                save_metrics_to_csv, 
                get_class_counts, 
                build_optimizer, 
                get_remix_y,
                create_sample,
                create_imbalance_ratio,
                display_cm,
                get_perm, 
                flatten_config,
                t_sne_vis)
from .callbacks import EarlyStopping
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import os
from torch.amp import autocast, GradScaler
import numpy as np
import random
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ["WANDB_MODE"] = "offline"

MODEL_NAME = 'bert-large-uncased'

dsdir = os.getenv('DSDIR')
if dsdir is None:
    raise EnvironmentError("DSDIR doesn't exist")

root_path = os.path.join(dsdir, 'HuggingFace_Models')
model_path = os.path.join(root_path, MODEL_NAME)

if not os.path.isdir(model_path):
    raise FileNotFoundError(f"The model doesn't exist in this folder : {model_path!r}")

class Trainer:
    # ---------------------------------------------------------------------
    # 1. INITIALISATION ----------------------------------------------------
    # ---------------------------------------------------------------------
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if cfg.train.cuda and torch.cuda.is_available() else "cpu")

        random.seed(self.cfg.train.seed)
        np.random.seed(self.cfg.train.seed)
        torch.manual_seed(self.cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.base_path = os.path.join(self.cfg.output.model_dir, self.cfg.dataset.name, f"IR_{self.cfg.dataset.ir}", self.cfg.experiment)
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.model_save_path = os.path.join(self.base_path, f"best_model_seed_{self.cfg.train.seed}.pt")
        self.log_file = os.path.join(self.base_path, f"log_metrics_seed_{self.cfg.train.seed}.csv")

        # Confusion matrix save folder
        self.cm_path = os.path.join(self.base_path, "confusion_matrices")
        os.makedirs(self.cm_path, exist_ok=True)

        self.tsne_path = os.path.join(self.base_path, "tsne_plots")
        os.makedirs(self.tsne_path, exist_ok=True)

        self.scaler = GradScaler()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
        self.logger.info(f"Tokenizer loaded : {self.tokenizer.__class__.__name__}")

        self.train_loader, self.val_loader, self.test_loader = build_dataset(cfg, self.tokenizer, self.logger)
        self.n_classes = cfg.dataset.num_cls

        torch.cuda.empty_cache()
        self.model = self._build_model()
        self.model = nn.DataParallel(self.model).to(self.device)
        self.logger.info(f"Model loaded : {self.model.__class__.__name__}")
        self.logger.info(f"model : {self.model}")

        self.n_pc = get_class_counts(self.train_loader, self.n_classes)
        self.criterion = build_loss(cfg, self.n_pc, self.device)
        self.test_criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = build_optimizer(cfg, self.model, self.train_loader)

        # for early stopping
        self.best_val_f1= 0
        #self.early_stop = False
        #self.val_patience = 0  # successive iteration when validation acc did not improve
        #self.model_save_path = os.path.join(self.cfg.output.model_dir, self.cfg.experiment + '_weights.pt')


        # Uncertainty ------------------------------------------------------
        self.use_unc = self.cfg.train.uncertainty
        self.val_steps, self.val_avg_ep_uncertainty = [], []
        self.val_correct, self.val_incorrect = [], []
        self.val_class_belief_vectors = {cls: [] for cls in range(self.n_classes)}

        # buffers epoch -> diag
        self.logged_beliefs, self.logged_uncertainties = [], []
        self.val_u_correct_ep, self.val_u_incorrect_ep, self.val_epochs = [], [], []

        self.uncertainty_vals_correct   = []
        self.uncertainty_vals_incorrect = []

        self.previous_beliefs = None  # sauvegarde des belief vectors avant switch
        self.switch_epoch = self.cfg.train.switch_epoch 

    # ------------------------------------------------------------------
    def _build_model(self):
        return TextBERT(
            pretrained_model=model_path,
            num_class=self.n_classes,
            fine_tune=self.cfg.train.finetune,
            dropout=self.cfg.model.dropout,
        )

    # ------------------------------------------------------------------
    # MIX / FORWARD STEP ------------------------------------------------
    # ------------------------------------------------------------------
    def _step(self, batch, epoch):
        y1 = batch.pop("label")
        
        if self.cfg.mix.method == "none":
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = self.model(**batch)
                #loss = self._compute_loss(outputs, y1, epoch, reduction=True)
                if self.use_unc:
                    evidences = exp_evidence(outputs)
                    alphas = evidences + 1 
                    S = torch.sum(alphas, dim=1, keepdim=True)
                    probs = alphas / S
                    preds = torch.argmax(probs, dim=1)

                    #loss = edl_loss(alphas, y1, epoch, 10, self.n_classes,reduction=True, device=self.device) # Add the case where I use_unc == False
                    loss = combined_loss(self.criterion, epoch, self.cfg.train.epochs, alphas, outputs, y1, self.n_classes, True, self.device)
                else: 
                    loss = self.criterion(outputs, y1)
                    preds = torch.argmax(outputs, dim=1)
            match = preds.eq(y1).sum()
                    
        elif self.cfg.mix.method == "remix" or self.cfg.mix.method == "mixup" :
            x1, att1 = batch["input_ids"], batch["attention_mask"]
            index = get_perm(x1)
            x2, y2, att2 = x1[index], y1[index], att1[index]
                
            lam_x = np.random.beta(self.cfg.mix.alpha, self.cfg.mix.alpha)
            lam_y = lam_x
            if self.cfg.mix.method == "remix":
                lam_y = get_remix_y(y1, y2, lam_x, self.n_pc, self.cfg.mix.k_majority, self.cfg.mix.tau, self.device)

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.module.forward_mix_encoder(x1, att1, x2, att2, lam_x)

                if self.use_unc:
                    evidences = exp_evidence(outputs)
                    alphas = evidences + 1 
                    S = torch.sum(alphas, dim=1, keepdim=True)
                    probs = alphas / S
                    preds = torch.argmax(probs, dim=1)

                    #loss1 = edl_loss(alphas, y1, epoch, 10, self.n_classes,reduction=False, device=self.device) # Add the case where I use_unc == False
                    #loss2 = edl_loss(alphas, y2, epoch, 10, self.n_classes,reduction=False, device=self.device) # Add the case where I use_unc == False
                    loss1 = combined_loss(self.criterion, epoch, self.cfg.train.epochs, alphas, outputs, y1, self.n_classes, False, self.device)
                    loss2 = combined_loss(self.criterion, epoch, self.cfg.train.epochs, alphas, outputs, y2, self.n_classes, False, self.device)

                    
                else:
                    print("criterion :", self.criterion)
                    loss1 = self.criterion(outputs, y1)
                    loss2 = self.criterion(outputs, y2)
                    preds = torch.argmax(outputs, dim=1)
            
            loss = (lam_y * loss1 + (1 - lam_y) * loss2).mean()
            match = (lam_y  * preds.eq(y1).float() + (1 - lam_y) * preds.eq(y2).float()).sum()

        return loss, match

    # ------------------------------------------------------------------
    # TRAIN ONE EPOCH ----------------------------------------------------
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        self.model.train()
        tr_loss, total, correct = 0.0, 0, 0
        outputs_buf, matches_buf = [], []

        for step, batch in enumerate(self.train_loader, 1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            y = batch['label']
            loss, match = self._step(batch, epoch)

            tr_loss += loss.item()
            correct += match.item()
            total += y.shape[0]

            # --- backward / opti ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # ---- logging périodique ----
            if step % self.cfg.train.eval_interval == 0:
                acc = correct / total
                avg_loss = tr_loss / total
                self.logger.info(f"[epoch {epoch} step {step}] Train loss:{avg_loss:.4f} | Train acc:{acc:.4f} | LR: {current_lr:.8f} ")
                val_metrics = self.evaluate(self.val_loader, epoch, step, test=False)

                if self.use_unc and epoch >= self.switch_epoch - 1:
                    self.logger.info(f"Saving belief vectors at epoch {epoch} for Mix-Belief...")
                    self.previous_beliefs = val_metrics["per_class"]


                # reset counters (train interval)
                tr_loss, total, correct = 0.0, 0, 0
                
                # ---- validation diag ----
                global_step = step + epoch * len(self.train_loader)
                log_dict =({
                    "train/loss": avg_loss,
                    "train/accuracy": acc,
                    "val/loss":    val_metrics["loss"],
                    "val/accuracy":val_metrics["acc"],
                    "val/precision":val_metrics["prec"],
                    "val/recall":val_metrics["rec"],
                    "val/f1": val_metrics["f1"],
                    "val/mcce":val_metrics["mcce"],
                    "val/step":    global_step,
                    "val/epoch":   epoch
                })

                if self.use_unc:
                    log_dict["val/mean_u"] = val_metrics.get("mean_u", 0)

                wandb.log(log_dict)

                save_metrics_to_csv(self.log_file, "val", val_metrics, epoch, step, self.cfg.train.seed)

                if val_metrics["f1"] > self.best_val_f1:
                    torch.save(self.model.module.state_dict(),  self.model_save_path)
                    print(f" ############### Where the model is saved : {self.model_save_path} ##############")
                    self.best_val_f1 = val_metrics['f1']
                '''    self.val_patience = 0
                else:
                    self.val_patience += 1
                    if self.val_patience == self.cfg.train.patience:
                        self.early_stop = True
                        return'''
                self.model.train()
    
    def evaluate(self, loader, epoch=None, step=None, test=False):
        self.model.eval()
        ts_loss, total, total_match, total_u = 0.0, 0, 0, 0.0
        all_preds, all_labels, all_uncertainties,  all_beliefs, all_embeddings, all_probs= [], [], [], [], [], []

        class_beliefs = defaultdict(list)
        class_uncertainties = defaultdict(list)

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                y = batch.pop("label")
                logits, embeddings = self.model(**batch)
                #_, preds = torch.max(outputs, dim=1)
                if self.use_unc:
                    evidences = exp_evidence(logits)
                    alphas = evidences + 1 
                    S = torch.sum(alphas, dim=1, keepdim=True)
                    probs = alphas / S
                    preds = torch.argmax(probs, dim=1)

                    #loss = edl_loss(alphas, y, epoch, 10, self.n_classes,reduction=True, device=self.device) # Add the case where I use_unc == False
                    loss = combined_loss(self.test_criterion, epoch, self.cfg.train.epochs, alphas, logits, y, self.n_classes, True, self.device)

                    
                    u = self.n_classes / S
                    total_u += u.sum().item()
                    
                    belief = evidences / S 

                    all_uncertainties.extend(u.cpu().squeeze().tolist())
                    all_beliefs.extend(belief.cpu().tolist())
                    all_probs.append(probs.cpu().numpy())

                    # Collect per-class info
                    for i in range(y.size(0)):
                        label = y[i].item()
                        class_beliefs[label].append(belief[i].cpu().numpy())
                        class_uncertainties[label].append(u[i].item())
                else:
                    loss = self.test_criterion(logits, y)
                    preds = torch.argmax(logits, dim=1)
                    all_probs.append(logits.cpu().numpy())
    
                total += y.shape[0]
                total_match+= (preds == y).sum().item()
                ts_loss += loss.item()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                if test:
                    all_embeddings.append(embeddings)
                
        all_probs = np.concatenate(all_probs, axis=0)
        metrics = compute_metrics(all_labels, all_preds, all_probs, self.n_classes)
        metrics["loss"] = ts_loss / total
        metrics["accuracy"] = total_match / total


        if self.use_unc:
            metrics["mean_u"] = total_u / total

            per_class_metrics = {}
            for c in range(self.n_classes):
                class_b = np.array(class_beliefs[c])  # shape: (N_c, C)
                class_u = np.array(class_uncertainties[c])  # shape: (N_c,)
                if len(class_b) > 0:
                    per_class_metrics[c] = {
                        "mean_belief": class_b.mean(axis=0).tolist(),  # liste de masses moyennes par classe
                        "mean_uncertainty": float(class_u.mean())
                    }
                else:
                    per_class_metrics[c] = {
                        "mean_belief": [0.0] * self.n_classes,
                        "mean_uncertainty": 0.0
                    }
            #self.val_class_belief_vectors[cls].append(class_beliefs.cpu().numpy())
            metrics["per_class"] = per_class_metrics
            metrics["all_uncertainties"] = all_uncertainties
            metrics["all_beliefs"] = all_beliefs
            metrics["all_labels"] = all_labels
            
            num_correct = total_match
            num_incorrect = total - total_match
            self.logger.info(f"Val Correct:{num_correct} | Incorrect:{num_incorrect} | Mean‑u:{metrics['mean_u']:.4f}")
    
            for c in range(self.n_classes):
                m = metrics["per_class"][c]
                belief_vec_str = ', '.join([f"{v:.4f}" for v in m["mean_belief"]])
                self.logger.info(f"Val Class {c} Mean‑u:{m['mean_uncertainty']:.4f} | Belief‑vec:[{belief_vec_str}]")
    
        self.logger.info(
                f"Val acc={metrics['accuracy']:.4f}, gm={metrics['gm']:.4f}, f1={metrics['f1']:.4f}, precision={metrics['prec']:.4f}, recall={metrics['rec']:.4f}, Calibration error={metrics['mcce']:.4f}, loss={metrics['loss']:.4f}"
            )
        display_cm(self.cm_path, wandb, metrics['cm'], self.cfg.experiment, self.cfg.train.seed, epoch, step, data='Test' if test else 'Val')
        if test : 
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_embeddings_np = all_embeddings.squeeze().cpu().numpy()
            t_sne_vis(all_embeddings_np, np.array(all_labels), self.cfg.train.seed, self.cfg.experiment, self.tsne_path)

        return metrics
    

    
    def run(self):
        wandb.init(project="uncertainty-v4", config=flatten_config(self.cfg), name=self.cfg.experiment + f"_run{self.cfg.train.seed}" , reinit=True)
        for e in range(self.cfg.train.epochs):
            self.train_one_epoch(e)
            '''if self.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {e}")
                break'''
        
        self.logger.info("Trining complete !")
        print('Best Validation accuracy: ', self.best_val_f1)
        self.model.module.load_state_dict(torch.load(self.model_save_path))
        
        test_metrics = self.evaluate(self.test_loader, test=True)
        cm_test = test_metrics.pop('cm')
        save_metrics_to_csv(self.log_file, "test", test_metrics) 
        wandb.finish()
        return test_metrics

class CurriculumTrainer(Trainer):
    def __init__(self, cfg, belief_ratio_threshold=1.8):
        super().__init__(cfg)
        self.prev_class_beliefs = {}
        self.confused_pairs = []
        self.mixup_alpha = self.cfg.mix.alpha
        self.belief_ratio_threshold = belief_ratio_threshold
        self.original_mix_method = cfg.mix.method


    def extract_confused_pairs_from_mean_belief(self, threshold=1.8):
        """
        Pour chaque classe vraie `cls`, identifie les classes `confused_with`
        selon la proximité des belief masses.
        """
        confused_pairs = []
        for cls, belief_vec in self.prev_class_beliefs.items():
            belief_vec = np.array(belief_vec)
            belief_true = belief_vec[cls] + 1e-8
            for other_cls, b in enumerate(belief_vec):
                if other_cls == cls:
                    continue
                ratio = belief_true / (b + 1e-8)
                if ratio < threshold:
                    confused_pairs.append((cls, other_cls))
        return confused_pairs

    def train_one_epoch(self, epoch):
        if epoch < self.cfg.train.switch_epoch:
            self.cfg.mix.method = 'none'
            self.logger.info(f"Epoch {epoch}: Warmup (no mixup)")
        else:
            self.cfg.mix.method = self.original_mix_method
            self.cfg.mix.alpha = self.mixup_alpha
            self.logger.info(f"Epoch {epoch}: Mix method = {self.cfg.mix.method} | alpha = {self.mixup_alpha}")

        super().train_one_epoch(epoch)

        if self.use_unc and self.previous_beliefs is not None:
            self.prev_class_beliefs = {
                int(cls): np.array(self.previous_beliefs[cls]["mean_belief"])
                for cls in self.previous_beliefs.keys()
            }
            self.confused_pairs = self.extract_confused_pairs_from_mean_belief(self.belief_ratio_threshold)
            self.logger.info(f"Confused class pairs (from belief masses): {self.confused_pairs}")

    def _guided_confusion_mixup_indices(self, y1):
        """
        Retourne un index par exemple dans le batch, pour construire les paires mixup
        en se basant sur les confused_pairs (list of tuples).
        """
        y_np = y1.cpu().numpy()
        batch_size = y1.size()[0]
        index = []
        for i, label in enumerate(y_np):
            # Cherche les classes vers lesquelles la classe `label` est confondue
            targets = [pair[1] for pair in self.confused_pairs if pair[0] == label]
            # Parmi les autres exemples du batch, cherche ceux qui ont ces classes
            possible = [j for j, other_label in enumerate(y_np) if other_label in targets]
            if possible:
                # Guided pair avec classe confondue
                j = np.random.choice(possible)
            else:
                # Cas 2: Aucun confused pair disponible ➔ on cherche dans la même classe (autre que i)
                same_class_candidates = [j for j, other_label in enumerate(y_np) if other_label == label and j != i]
    
                if same_class_candidates:
                    j = np.random.choice(same_class_candidates)
                else:
                    # Cas limite : aucun autre exemple de la même classe dans le batch
                    # On garde l'indice lui-même (pas de mixup possible ici)
                    j = i
            index.append(j)
        print(f"labels for this batch: {y_np}")
        print(f"index for this batch: {index}")
        return torch.tensor(index).to(self.device)
        
    def _step(self, batch, epoch):
        if self.cfg.mix.method == 'none':
            return super()._step(batch, epoch)
        y1 = batch.pop("label")
        x1, att1 = batch['input_ids'], batch['attention_mask']
        lam_x = np.random.beta(self.cfg.mix.alpha, self.cfg.mix.alpha)
        index = self._guided_confusion_mixup_indices(y1)
        #index = get_perm(x1)
        x2, y2, att2 = x1[index], y1[index], att1[index]

        if self.cfg.mix.method == 'mixup':
            lam_y = lam_x
        elif self.cfg.mix.method == 'remix':
            lam_y = get_remix_y(y1, y2, lam_x, self.n_pc, self.cfg.mix.k_majority, self.cfg.mix.tau, self.device)
        elif self.cfg.mix.method == 'mix-belief':
            lam_x = np.random.beta(2, 1)
            if lam_x > 0.5 :
                lam_y = 1
            else:
                lam_y = lam_x
            '''freq_tensor = torch.tensor(self.n_pc, dtype=torch.float32, device=device)
            freq_y1 = freq_tensor[y1]
            freq_y2 = freq_tensor[y2]

            is_y1_minority = freq_y1 < freq_y2
            lam_tensor = torch.tensor(lam_x, device=device)
            lam_y = torch.where(
                is_y1_minority & (lam_tensor > 0.5),
                torch.tensor(1.0, device=device),
                lam_tensor
            )'''

    
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = self.model.module.forward_mix_encoder(x1, att1, x2, att2, lam_x)

            evidences = exp_evidence(logits)
            alphas = evidences + 1 
            S = torch.sum(alphas, dim=1, keepdim=True)
            probs = alphas / S
            preds = torch.argmax(probs, dim=1)

            loss1 = combined_loss(self.criterion, epoch, self.cfg.train.epochs, alphas, logits, y1, self.n_classes, False, self.device)
            loss2 = combined_loss(self.criterion, epoch, self.cfg.train.epochs, alphas, logits, y2, self.n_classes, False, self.device)
            #loss1 = edl_loss(alphas, y1, epoch, 10, self.n_classes,reduction=False, device=self.device)
            #loss2 = edl_loss(alphas, y2, epoch, 10, self.n_classes,reduction=False, device=self.device)
            
            loss = lam_y * loss1 + (1 - lam_y) * loss2
            loss = loss.mean()
                    
            # Convert lam_y to tensor if needed
            if not isinstance(lam_y, torch.Tensor):
                lam_y = torch.full_like(y1.float(), lam_y)  # shape: [B]
            
            # Vectorisé pour batch
            match = (lam_y.data * preds.eq(y1.data ).float() + (1 - lam_y.data ) * preds.eq(y2.data ).float()).sum()

        return loss, match
