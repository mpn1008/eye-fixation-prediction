import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data import build_loaders
from .losses import SaliencyLoss, compute_metrics
from .models import SaliencyNet


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------


def train_epoch(
    model: SaliencyNet,
    loader,
    criterion: SaliencyLoss,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}

    for scene, fixation in tqdm(loader, desc="train", leave=False):
        scene = scene.to(device, non_blocking=True)
        fixation = fixation.to(device, non_blocking=True)

        optimiser.zero_grad()
        pred = model(scene)
        loss, breakdown = criterion(pred, fixation)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        for k, v in breakdown.items():
            totals[k] = totals.get(k, 0.0) + v

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(
    model: SaliencyNet,
    loader,
    criterion: SaliencyLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_totals: dict[str, float] = {}
    metric_totals: dict[str, float] = {}

    for scene, fixation in tqdm(loader, desc="val  ", leave=False):
        scene = scene.to(device, non_blocking=True)
        fixation = fixation.to(device, non_blocking=True)

        pred = model(scene)
        _, breakdown = criterion(pred, fixation)
        metrics = compute_metrics(pred, fixation)

        for k, v in breakdown.items():
            loss_totals[k] = loss_totals.get(k, 0.0) + v
        for k, v in metrics.items():
            metric_totals[k] = metric_totals.get(k, 0.0) + v

    n = len(loader)
    combined = {k: v / n for k, v in loss_totals.items()}
    combined.update({k: v / n for k, v in metric_totals.items()})
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.train.checkpoint_dir = cfg.train.checkpoint_dir / f"{cfg.model.backbone}_{timestamp}"
    cfg.train.log_dir = cfg.train.log_dir / f"{cfg.model.backbone}_{timestamp}"

    cfg.train.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(cfg.train.log_dir))

    # Data
    train_loader, val_loader = build_loaders(
        train_root=cfg.data.train_root,
        val_root=cfg.data.val_root,
        train_images_txt=cfg.data.train_images_txt,
        train_fixations_txt=cfg.data.train_fixations_txt,
        val_images_txt=cfg.data.val_images_txt,
        val_fixations_txt=cfg.data.val_fixations_txt,
        image_size=cfg.data.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = SaliencyNet(
        pretrained=cfg.model.pretrained,
        decoder_channels=cfg.model.decoder_channels,
    ).to(device)

    criterion = SaliencyLoss(
        w_kl=cfg.train.w_kl,
        w_cc=cfg.train.w_cc,
        w_mse=cfg.train.w_mse,
    )
    optimiser = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.train.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_auc = -float("inf")

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.train.epochs}")

        train_stats = train_epoch(model, train_loader, criterion, optimiser, device)
        val_stats = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Logging
        for k, v in train_stats.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        cc = val_stats.get("metric/cc", -1.0)
        val_loss = val_stats.get("loss/total", float("inf"))
        auc_judd = val_stats.get("metric/auc_judd", -1.0)
        print(
            f"  train loss: {train_stats['loss/total']:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val KL: {val_stats.get('loss/kl', -1.0):.4f} | "
            f"AUC-Judd: {auc_judd:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if auc_judd > best_auc:
            best_auc = auc_judd
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                cfg.train.checkpoint_dir / "best.pt",
            )
            print(f"  ✓ saved best checkpoint (AUC-Judd={auc_judd:.4f})")

    writer.close()
    print(
        f"\nTraining complete. Best val loss: {best_val_loss:.4f} | Best AUC-Judd: {best_auc:.4f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fixation prediction model")
    parser.add_argument("--train-root", type=Path, default="data/training")
    parser.add_argument("--val-root", type=Path, default="data/validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = Config(
        data=DataConfig(
            train_root=args.train_root, val_root=args.val_root, num_workers=args.num_workers
        ),
        model=ModelConfig(pretrained=not args.no_pretrained),
        train=TrainConfig(batch_size=args.batch_size, lr=args.lr),
    )
    main(cfg)
