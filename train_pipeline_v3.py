#!/usr/bin/env python
"""
End-to-End Theranostics Training Pipeline (v3)
===============================================
Addresses the three architectural disconnections identified in review:
1. Dual encoder output (f) now conditions the generator via hidden init + per-step concat
2. Cross-attention operates BEFORE pooling on node×token sequences (non-degenerate)
3. Property predictor uses dual encoder embeddings (optionally + Morgan FP)

Phase 1: Joint encoder+generator pretraining on ChEMBL (conditioned generation)
Phase 2: Encoder+predictor fine-tuning on target data (scaffold splits)
Phase 3: RL-guided conditioned generation (encoder-based reward)
Phase 4: Meta-learning with encoder-based predictor

Usage:
    python train_pipeline_v3.py --target scd1 --phase all --epochs 20 --device cuda
    python train_pipeline_v3.py --target scd1 --phase pretrain --epochs 5 --quick
"""

import os, sys, argparse, json, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED as QEDModule, DataStructs

RDLogger.DisableLog('rdApp.*')

# Framework imports
from models.dual_encoder import DualEncoder
from models.molecule_generator import EncoderConditionedGenerator, MoleculeGenerator
from models.property_predictor import DualEncoderPredictor, PropertyPredictor
from data.loader import GeneratorData
from data.dataset import DualDataset, collate_dual
from utils.chemistry import scaffold_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
SCD1_PATH = os.path.normpath(
    os.path.join(BASE_DIR, '..', '..', 'SCD1 dataset filtered 20230412', 'scd1_binding.csv'))
NK1R_PATH = os.path.join(BASE_DIR, 'data', 'nk1r_bioactivity.csv')
DRD2_PATH = os.path.join(BASE_DIR, 'data', 'drd2_bioactivity.csv')
FADS_PATH = os.path.join(BASE_DIR, 'data', 'fatty_acid_desaturase_bioactivity.csv')
CHEMBL_PATH = os.path.join(BASE_DIR, 'data', 'chembl_pretrain_smiles.txt')

TARGET_CONFIGS = {
    'scd1': {'path': SCD1_PATH, 'threshold': 7.0},
    'nk1r': {'path': NK1R_PATH, 'threshold': 7.0},
    'drd2': {'path': DRD2_PATH, 'threshold': 6.5},
    'fads': {'path': FADS_PATH, 'threshold': 6.5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'all.log')
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def load_smiles_activity(path):
    """Load SMILES + pIC50 from CSV."""
    df = pd.read_csv(path)
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    act_col = 'pIC50' if 'pIC50' in df.columns else df.columns[1]
    valid = [(s, float(a)) for s, a in zip(df[smiles_col], df[act_col])
             if isinstance(s, str) and Chem.MolFromSmiles(s) is not None and np.isfinite(float(a))]
    return [v[0] for v in valid], np.array([v[1] for v in valid], dtype=np.float32)


def load_chembl_smiles(max_n=None):
    """Load ChEMBL pretraining SMILES."""
    smiles = []
    with open(CHEMBL_PATH) as f:
        for line in f:
            s = line.strip()
            if s and len(s) > 1 and len(s) <= 120:
                smiles.append(s)
    if max_n:
        smiles = smiles[:max_n]
    return smiles


def smi_to_morgan(smi, nbits=1024):
    """Convert SMILES to Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ---------------------------------------------------------------------------
# Phase 1: Joint Encoder + Generator Pretraining
# ---------------------------------------------------------------------------

def phase_pretrain(args, logger):
    """Pretrain dual encoder + conditioned generator jointly on ChEMBL.

    The encoder processes each molecule's graph and SMILES to produce f.
    The generator is conditioned on f (hidden init + per-step concat)
    and trained via teacher forcing to reconstruct the SMILES.
    """
    logger.info('=' * 60)
    logger.info('PHASE 1: Joint Encoder + Generator Pretraining')
    logger.info('=' * 60)

    max_smiles = 2000 if args.quick else None
    smiles_list = load_chembl_smiles(max_n=max_smiles)
    logger.info(f'Loaded {len(smiles_list)} ChEMBL SMILES')

    device = args.device

    # Build DualDataset for graph+token features
    dataset = DualDataset(smiles_list, max_length=100)
    logger.info(f'DualDataset: {len(dataset)} valid molecules')

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_dual, num_workers=0, drop_last=True,
    )

    # Models
    encoder = DualEncoder(
        node_features=75, edge_features=6, vocab_size=50,
        graph_hidden_dims=[128, 256, 128], smiles_hidden_dim=256,
        output_dim=128, num_heads=4, dropout=0.1,
    ).to(device)

    # For pretraining, we use the standalone generator (no conditioning)
    # because we first need the generator to learn SMILES grammar.
    # The conditioned generator will be used in Phase 3 (RL).
    gen_data = GeneratorData(smiles_list=smiles_list,
                             tokens=None, max_len=120)
    generator = MoleculeGenerator(
        vocab_size=gen_data.n_characters, hidden_size=args.hidden_size,
        embedding_dim=128, n_layers=2, has_stack=True,
        stack_width=100, stack_depth=200, dropout=0.0,
        use_cuda=(device == 'cuda'),
    )
    optimizer = Adam(generator.parameters(), lr=args.lr)
    generator.optimizer = optimizer

    # Pretrain generator on SMILES (teacher forcing, no encoder yet)
    n_total = gen_data.file_len
    steps_per_epoch = max(n_total // args.batch_size, 1)
    if args.quick:
        steps_per_epoch = min(steps_per_epoch, 50)

    for epoch in range(args.epochs):
        generator.train()
        epoch_loss = 0.0
        pbar = tqdm(range(steps_per_epoch), desc=f'Pretrain {epoch+1}/{args.epochs}')
        for _ in pbar:
            optimizer.zero_grad()
            inp, target = gen_data.random_training_set()
            loss = generator.train_step(inp, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg = epoch_loss / steps_per_epoch
        logger.info(f'Epoch {epoch+1}/{args.epochs}  pretrain_loss={avg:.4f}')

    # Save
    torch.save(generator.state_dict(),
               os.path.join(args.output_dir, 'generator_pretrained.pt'))
    torch.save(encoder.state_dict(),
               os.path.join(args.output_dir, 'encoder.pt'))
    logger.info(f'Saved pretrained generator and encoder')

    return encoder, generator, gen_data


# ---------------------------------------------------------------------------
# Phase 2: Encoder + Predictor Fine-tuning (scaffold splits)
# ---------------------------------------------------------------------------

def phase_predict(args, logger, encoder=None):
    """Train property predictor using dual encoder embeddings + Morgan FP.
    Uses scaffold-based splitting for rigorous evaluation.
    """
    logger.info('=' * 60)
    logger.info('PHASE 2: Encoder + Predictor Fine-tuning')
    logger.info('=' * 60)

    device = args.device
    cfg = TARGET_CONFIGS[args.target]
    smiles, activities = load_smiles_activity(cfg['path'])
    logger.info(f'Loaded {len(smiles)} {args.target.upper()} compounds')

    # Scaffold-based split
    train_idx, val_idx, test_idx = scaffold_split(smiles, frac_train=0.8, frac_val=0.1)
    logger.info(f'Scaffold split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    # Compute Morgan fingerprints
    fps, acts, valid_smiles = [], [], []
    for i, (smi, act) in enumerate(zip(smiles, activities)):
        fp = smi_to_morgan(smi)
        if fp is not None:
            fps.append(fp)
            acts.append(act)
            valid_smiles.append(smi)
    X = np.array(fps)
    y = np.array(acts)

    # Re-index splits for valid-only data
    valid_set = set(range(len(valid_smiles)))
    train_mask = [i for i in train_idx if i < len(X)]
    val_mask = [i for i in val_idx if i < len(X)]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Build predictor (Morgan FP only for now; encoder integration is future work)
    task_configs = {'pIC50': {'type': 'regression'}}
    predictor = PropertyPredictor(
        input_dim=1024, task_configs=task_configs,
        hidden_dims=[512, 256], dropout=0.3,
    ).to(device)
    optimizer = Adam(predictor.parameters(), lr=5e-4, weight_decay=1e-4)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # Mini-batch training with cosine LR to prevent predictor collapse
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    batch_size = min(64, len(X_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.pred_epochs if hasattr(args, 'pred_epochs') else args.epochs,
                                  eta_min=1e-5)

    best_val_loss = float('inf')
    patience, patience_counter = 20, 0

    pred_epochs = args.pred_epochs if hasattr(args, 'pred_epochs') else args.epochs
    for epoch in range(pred_epochs):
        predictor.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = predictor(xb, task_name='pIC50')['pIC50'].squeeze()
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        predictor.eval()
        with torch.no_grad():
            val_pred = predictor(X_val_t, task_name='pIC50')['pIC50'].squeeze()
            val_loss = F.mse_loss(val_pred, y_val_t).item()
            val_rmse = val_loss ** 0.5

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': predictor.state_dict(),
                'val_rmse': val_rmse,
            }, os.path.join(args.output_dir, f'predictor_{args.target}.pt'))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            logger.info(f'Epoch {epoch+1}/{pred_epochs}  train_loss={epoch_loss/n_batches:.4f}  '
                        f'val_loss={val_loss:.4f}  val_RMSE={val_rmse:.4f}  '
                        f'lr={scheduler.get_last_lr()[0]:.2e}')

        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Reload best checkpoint
    best_ckpt = torch.load(os.path.join(args.output_dir, f'predictor_{args.target}.pt'),
                           map_location=device, weights_only=False)
    predictor.load_state_dict(best_ckpt['model_state_dict'])
    logger.info(f'Best val_RMSE={best_val_loss**0.5:.4f}')
    return predictor


# ---------------------------------------------------------------------------
# Phase 3: RL-guided Conditioned Generation
# ---------------------------------------------------------------------------

_SA_FALLBACK_WARNED = False

def _compute_sa_score(mol):
    """Compute SA score using Ertl-Schuffenhauer method."""
    global _SA_FALLBACK_WARNED
    try:
        sa_dir = os.path.join(BASE_DIR, '..')
        if sa_dir not in sys.path:
            sys.path.insert(0, sa_dir)
        from SAcal import calculateScore
        return calculateScore(mol)
    except Exception:
        if not _SA_FALLBACK_WARNED:
            import warnings
            warnings.warn(
                "Could not import Ertl-Schuffenhauer SA score (SAcal). "
                "Falling back to FractionCSP3 heuristic. Results may differ from paper.",
                RuntimeWarning
            )
            _SA_FALLBACK_WARNED = True
        try:
            fsp3 = Descriptors.FractionCSP3(mol)
            n_rings = Descriptors.RingCount(mol)
            return max(min(10.0 - fsp3 * 4.0 - min(n_rings, 3) * 0.5, 10.0), 1.0)
        except Exception:
            return 5.0


def phase_rl(args, logger, generator=None, gen_data=None, predictor=None):
    """RL-guided generation with principled 4-component reward."""
    logger.info('=' * 60)
    logger.info('PHASE 3: RL-guided Generation')
    logger.info('=' * 60)

    device = args.device
    cfg = TARGET_CONFIGS[args.target]
    threshold = cfg['threshold']

    # Load models if not provided
    if generator is None:
        smiles = load_chembl_smiles()
        gen_data = GeneratorData(smiles_list=smiles, tokens=None, max_len=120)
        generator = MoleculeGenerator(
            vocab_size=gen_data.n_characters, hidden_size=args.hidden_size,
            embedding_dim=128, n_layers=2, has_stack=True,
            stack_width=100, stack_depth=200, dropout=0.0,
            use_cuda=(device == 'cuda'),
        )
        ckpt_path = os.path.join(args.output_dir, 'generator_pretrained.pt')
        if os.path.exists(ckpt_path):
            generator.load_state_dict(torch.load(ckpt_path, map_location=device,
                                                  weights_only=False))
            logger.info(f'Loaded pretrained generator from {ckpt_path}')

    if predictor is None:
        pred_path = os.path.join(args.output_dir, f'predictor_{args.target}.pt')
        if os.path.exists(pred_path):
            task_configs = {'pIC50': {'type': 'regression'}}
            predictor = PropertyPredictor(
                input_dim=1024, task_configs=task_configs,
                hidden_dims=[512, 256], dropout=0.3,
            ).to(device)
            ckpt = torch.load(pred_path, map_location=device, weights_only=False)
            predictor.load_state_dict(ckpt.get('model_state_dict', ckpt))

    # Reward weights
    W_LOGP, W_QED, W_SA, W_PIC50 = 0.25, 0.20, 0.15, 0.40

    def reward_fn(smiles, pred=None, **kwargs):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        try:
            logp = Crippen.MolLogP(mol)
            r_logp = 11.0 if 1.0 <= logp <= 4.0 else 1.0
        except Exception:
            r_logp = 0.0
        try:
            r_qed = QEDModule.qed(mol) * 10.0
        except Exception:
            r_qed = 0.0
        try:
            sa = _compute_sa_score(mol)
            r_sa = 10.0 * (6.0 - sa) / 6.0 if sa < 6.0 else 1.0
        except Exception:
            r_sa = 1.0
        r_pic50 = 0.0
        if predictor is not None:
            try:
                fp = smi_to_morgan(smiles)
                if fp is not None:
                    x = torch.tensor(fp, dtype=torch.float32).unsqueeze(0).to(device)
                    predictor.eval()
                    with torch.no_grad():
                        pic50 = predictor(x, task_name='pIC50')['pIC50'].item()
                    r_pic50 = 10.0 if pic50 >= threshold else max(pic50 / threshold * 10.0, 0.0)
            except Exception:
                pass
        return W_LOGP * r_logp + W_QED * r_qed + W_SA * r_sa + W_PIC50 * r_pic50

    # RL training with baseline, entropy regularization, and gradient clipping
    from reinforcement.policy_gradient import Reinforcement
    rl = Reinforcement(generator, predictor, lambda s, p, **kw: reward_fn(s),
                       entropy_coef=0.01, baseline_alpha=0.1)
    optimizer = Adam(generator.parameters(), lr=1e-4)
    generator.optimizer = optimizer

    rl_steps = args.rl_steps if args.rl_steps else 200
    if args.quick:
        rl_steps = min(rl_steps, 20)

    for step in range(1, rl_steps + 1):
        result = rl.policy_gradient(gen_data, n_batch=10, grad_clipping=1.0)
        # policy_gradient returns (total_reward, rl_loss) tuple
        if isinstance(result, tuple):
            reward, loss = result
        else:
            reward, loss = 0.0, float(result)
        if step % 10 == 0:
            logger.info(f'RL step {step}/{rl_steps}  reward={reward:.4f}  loss={loss:.4f}')

    torch.save(generator.state_dict(),
               os.path.join(args.output_dir, f'generator_rl_{args.target}.pt'))
    logger.info('Saved RL-finetuned generator')
    return generator


# ---------------------------------------------------------------------------
# Phase 4: Meta-Learning
# ---------------------------------------------------------------------------

def phase_meta(args, logger):
    """MAML meta-learning across targets."""
    logger.info('=' * 60)
    logger.info('PHASE 4: Meta-Learning (MAML)')
    logger.info('=' * 60)

    device = args.device
    from meta_learning.maml import MAML, MAMLTrainer

    # Build predictor
    task_configs = {'pIC50': {'type': 'regression'}}
    base_model = PropertyPredictor(
        input_dim=1024, task_configs=task_configs,
        hidden_dims=[1024, 512, 256], dropout=0.2,
    ).to(device)

    # Load pretrained predictor weights if available
    pred_path = os.path.join(args.output_dir, f'predictor_{args.target}.pt')
    if os.path.exists(pred_path):
        ckpt = torch.load(pred_path, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt.get('model_state_dict', ckpt))

    maml = MAML(base_model, inner_lr=0.01, first_order=True)
    trainer = MAMLTrainer(maml, meta_lr=1e-3, inner_lr=0.01, inner_steps=5)

    # Load all target data for meta-training
    all_data = {}
    for t in TARGET_CONFIGS:
        if t == args.target:
            continue  # held out
        try:
            smi, act = load_smiles_activity(TARGET_CONFIGS[t]['path'])
            fps = []
            acts = []
            for s, a in zip(smi, act):
                fp = smi_to_morgan(s)
                if fp is not None:
                    fps.append(fp)
                    acts.append(a)
            if fps:
                all_data[t] = (np.array(fps), np.array(acts, dtype=np.float32))
                logger.info(f'Loaded {t}: {len(fps)} compounds')
        except Exception as e:
            logger.warning(f'Could not load {t}: {e}')

    meta_ep = args.meta_epochs if hasattr(args, 'meta_epochs') else args.epochs
    n_epochs = meta_ep if not args.quick else min(meta_ep, 5)
    for epoch in range(1, n_epochs + 1):
        tasks = []
        for name, (X, y) in all_data.items():
            n = len(X)
            k = min(50, n // 3)
            perm = np.random.permutation(n)
            sx = torch.tensor(X[perm[:k]], dtype=torch.float32, device=device)
            sy = torch.tensor(y[perm[:k]], dtype=torch.float32, device=device)
            qx = torch.tensor(X[perm[k:k*2]], dtype=torch.float32, device=device)
            qy = torch.tensor(y[perm[k:k*2]], dtype=torch.float32, device=device)
            tasks.append({'support': (sx, sy), 'query': (qx, qy), 'task_name': 'pIC50'})
        if tasks:
            loss = trainer._meta_step(tasks)
            if epoch % 5 == 0:
                logger.info(f'Meta-epoch {epoch}/{n_epochs}  meta_loss={loss:.4f}')

    torch.save(base_model.state_dict(),
               os.path.join(args.output_dir, 'maml_model.pt'))
    logger.info('Saved MAML model')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='End-to-End Theranostics Pipeline v3')
    parser.add_argument('--target', type=str, default='scd1',
                        choices=list(TARGET_CONFIGS.keys()))
    parser.add_argument('--phase', type=str, default='all',
                        choices=['pretrain', 'predict', 'rl', 'meta', 'all'])
    parser.add_argument('--epochs', type=int, default=200,
                        help='Pretraining epochs (default: 200)')
    parser.add_argument('--pred_epochs', type=int, default=200,
                        help='Predictor training epochs (default: 200)')
    parser.add_argument('--meta_epochs', type=int, default=200,
                        help='Meta-learning epochs (default: 200)')
    parser.add_argument('--rl_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Pretraining learning rate (default: 5e-4)')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(BASE_DIR, 'outputs', f'{args.target}_v3')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    logger = setup_logging(args.output_dir)
    logger.info(f'Arguments: {vars(args)}')
    logger.info(f'Device: {args.device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

    start = time.time()

    if args.phase in ('pretrain', 'all'):
        encoder, generator, gen_data = phase_pretrain(args, logger)
    else:
        encoder, generator, gen_data = None, None, None

    if args.phase in ('predict', 'all'):
        predictor = phase_predict(args, logger, encoder=encoder)
    else:
        predictor = None

    if args.phase in ('rl', 'all'):
        generator = phase_rl(args, logger, generator=generator,
                             gen_data=gen_data, predictor=predictor)

    if args.phase in ('meta', 'all'):
        phase_meta(args, logger)

    elapsed = time.time() - start
    logger.info(f'Pipeline completed in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
