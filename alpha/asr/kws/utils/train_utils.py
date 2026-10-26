"""训练通用工具：标签提取、精度评测、预训练权重加载。"""

import logging
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from .attribute_dict import AttributeDict
from spikingjelly.activation_based import functional


# ═══════════════════════════════════════════════════════════════
#  标签提取 / 精度评测
# ═══════════════════════════════════════════════════════════════

def extract_labels(batch: dict, device) -> torch.Tensor:
    """从 batch 的 supervisions 中提取说话人 id 作为标签（数据管线约定）。"""
    labels = [int(cut.supervisions[0].speaker) for cut in batch["supervisions"]["cut"]]
    return torch.tensor(labels, dtype=torch.long).to(device)


def batch_lengths(batch: dict, device) -> torch.Tensor:
    """每条 cut 的有效 FBank 帧数（num_frames 优先，回退 duration×100）。

    统一供离线训练（system.py）、评测（compute_test_accuracy）与
    在线适应（adakws_trainer）使用，避免三处重复实现。
    """
    return torch.tensor(
        [getattr(cut, "num_frames", None) or int(round(cut.duration * 100))
         for cut in batch["supervisions"]["cut"]],
        dtype=torch.long, device=device,
    )


def compute_test_accuracy(
    model: nn.Module,
    test_dl: torch.utils.data.DataLoader,
    device,
    world_size: int = 1,
) -> float:
    """计算测试集精度。传 lengths 做掩码池化，消除 batch 内补零帧污染。"""
    model.eval()
    total_correct = 0
    total_samples = 0
    for batch in test_dl:
        feature = batch["inputs"].to(device).permute(0, 2, 1)
        lengths = batch_lengths(batch, device)
        functional.reset_net(model)
        class_logits = model(feature, lengths)
        labels = extract_labels(batch, device)
        total_correct += (class_logits.argmax(1) == labels).sum().item()
        total_samples += feature.size(0)

    if world_size > 1:
        import torch.distributed as dist

        stats = torch.tensor([total_correct, total_samples], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_correct, total_samples = stats.cpu().tolist()

    return total_correct / total_samples if total_samples > 0 else 0.0


def _subset_of(source: str) -> str:
    """按录制源路径划分子集：真实车载 / 加噪 simu / RIR 混响。"""
    if "rec/comm_domain" in source:
        return "real_car"
    if "addsimu" in source:
        return "simu_noise"
    return "rir_reverb"


def compute_subset_accuracy(
    model: nn.Module,
    test_dl: torch.utils.data.DataLoader,
    device,
) -> dict:
    """按目标域子集统计精度（与整体精度同口径：eval + reset_net + 掩码池化）。"""
    model.eval()
    stats: dict = {}
    with torch.no_grad():
        for batch in test_dl:
            feature = batch["inputs"].to(device).permute(0, 2, 1)
            lengths = batch_lengths(batch, device)
            functional.reset_net(model)
            class_logits = model(feature, lengths)
            labels = extract_labels(batch, device)
            correct = (class_logits.argmax(1) == labels).cpu()
            for i, cut in enumerate(batch["supervisions"]["cut"]):
                key = _subset_of(cut.recording.sources[0].source)
                acc, total = stats.setdefault(key, [0, 0])
                stats[key] = [acc + int(correct[i].item()), total + 1]
    return {key: {"acc": acc / total, "n": total} for key, (acc, total) in stats.items()}


# ═══════════════════════════════════════════════════════════════
#  预训练权重加载（直接支持离线 Lightning .ckpt 与历史 {"model": ...} 格式）
# ═══════════════════════════════════════════════════════════════

def _torch_load_any(filename: Path):
    """Lightning ckpt 内含 OmegaConf 等对象需 weights_only=False；其余走安全模式。"""
    try:
        return torch.load(filename, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(filename, map_location='cpu', weights_only=False)


def _materialize_lightning_ema(ckpt: dict) -> dict:
    """把 Lightning ckpt 的 torch_ema 影子参数换入同构模型，返回 model.* 权重。

    仅当 paths.use_ema=true 且 ckpt 含 ema_state_dict 时调用。
    """
    from omegaconf import OmegaConf
    from torch_ema import ExponentialMovingAverage
    from alpha.asr.kws.system import MultiTauKwsKd  # 局部导入避免循环

    conf = OmegaConf.create(ckpt.get('hyper_parameters') or {})
    if 'model' not in conf:
        raise RuntimeError('Lightning ckpt 缺少 hyper_parameters，无法物化 EMA')
    system = MultiTauKwsKd(conf)
    system.load_state_dict(ckpt['state_dict'])
    ema = ExponentialMovingAverage(system.parameters(), decay=0.9995)
    ema.load_state_dict(ckpt['ema_state_dict'])
    ema.copy_to(system.parameters())
    return {f'model.{k}': v for k, v in system.model.state_dict().items()}


def load_pretrained_model(params: AttributeDict, model: nn.Module) -> None:
    """加载离线预训练权重到 MultiTauLIFCommanderKWS（keys: frontend/blocks/classifier）。

    支持 checkpoint（自动识别）：
      1. Lightning .ckpt（state_dict / ema_state_dict + hyper_parameters）
         - paths.use_ema=true 时物化 EMA 权重，否则用 raw state_dict
      2. 历史/导出格式 {"model": {...}}（model.* 键）
    键映射：model.* / backbone.* 前缀一律剥掉，落到本模型的裸键。
    """
    filename = Path(params.paths.pretrained_model)
    logging.info(f"尝试加载预训练模型: {filename}")
    logging.info(f"文件是否存在: {filename.exists()}")

    if not filename.exists():
        logging.warning(f"预训练模型文件不存在: {filename}")
        logging.info("将使用随机初始化的模型开始训练")
        return

    try:
        ckpt = _torch_load_any(filename)
        use_ema = bool(getattr(params.paths, 'use_ema', False))

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:      # Lightning
            if use_ema and 'ema_state_dict' in ckpt:
                src_state = _materialize_lightning_ema(ckpt)
                logging.info('已物化 EMA 权重（ema_state_dict → model）')
            else:
                src_state = {k: v for k, v in ckpt['state_dict'].items()
                             if k.startswith('model.')}
        elif isinstance(ckpt, dict) and 'model' in ckpt:          # 历史/导出格式
            src_state = ckpt['model']
        else:
            src_state = ckpt                                       # 裸 state_dict

        mapped = {}
        for key, value in src_state.items():
            if key.startswith('model.'):
                mapped[key[len('model.'):]] = value
            elif key.startswith('backbone.'):  # 兼容历史在线产物
                mapped[key[len('backbone.'):]] = value
            else:
                mapped[key] = value

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logging.warning(f"缺失的参数（将被随机初始化）: {missing[:5]}...")
        logging.info(f"成功加载 {len(mapped)} 个预训练参数（共 {len(src_state)} 个）")

    except Exception as e:
        logging.warning(f"加载预训练模型失败: {e}\n{traceback.format_exc()}")
        logging.info("将使用随机初始化的模型开始训练")

    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
