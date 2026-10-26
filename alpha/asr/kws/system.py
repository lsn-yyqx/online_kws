"""KWS 任务系统（复用 AlphaSNN 框架 Lightning 管线）。

AnnKwsTeacher   ：TdnnLstm ANN 教师（继承框架 LiteASR，仅 Lhotse 接口适配 + 原配方
                  超参数：调度器 / 每 epoch 种子 / SpecAugment）。
MultiTauKwsKd   ：MultiTauLIFCommander SNN 学生 + TdnnLstm 教师知识蒸馏，
                  继承框架 MultiTauSpikeASR（EMA / 分组优化器 / branch balance /
                  reset_net / spike 监控全部复用），仅叠加 KD、AvgPool 下采样、
                  SpecAugment 与 raw/EMA 双通道验证。

注册：alpha/asr/system.py 末尾 import 本模块，使 modules/train.py 的
get_model() 能按 system.name 查找到这两个类。

数据：Lhotse 预计算 FBank，配置见 examples/carKWS_offline/conf/*.yaml。
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lhotse.utils import fix_random_seed
from omegaconf import OmegaConf
from spikingjelly.activation_based import functional

from modules.utils.logging import logger
from alpha.asr.system import LiteASR, MultiTauSpikeASR, SpikeASR
from alpha.asr.model.tdnn_lstm import TdnnLstm
from alpha.asr.kws.dataset import build_eval_loader, build_train_loader
from alpha.asr.model.multitau_lif_commander_kws import subsampling_mask
from alpha.asr.kws.utils.train_utils import batch_lengths, extract_labels


# ═══════════════════════════════════════════════════════════════
#  公共工具
# ═══════════════════════════════════════════════════════════════

def spec_augment(features: torch.Tensor, time_mask: int, freq_mask: int,
                 num_time_masks: int, num_freq_masks: int) -> torch.Tensor:
    """批级时频遮罩（features [B, F, T]，语义与历史重建版逐位一致）。"""
    _, f_dim, t = features.shape
    for _ in range(num_time_masks):
        w = int(torch.randint(1, time_mask + 1, (1,)))
        start = int(torch.randint(0, max(1, t - w) + 1, (1,)))
        features[:, :, start:start + w] = 0.0
    for _ in range(num_freq_masks):
        w = int(torch.randint(1, freq_mask + 1, (1,)))
        start = int(torch.randint(0, max(1, f_dim - w) + 1, (1,)))
        features[:, start:start + w, :] = 0.0
    return features


class LhotseKwsMixin:
    """基于 conf.data.{train,valid,test} 块构建 Lhotse dataloader。

    每块字段：manifest_dir / cuts_file / max_duration / num_buckets /
    num_workers；train 块额外支持 spec_augment 与 on_the_fly。
    """

    def _subset_conf(self, subset: str) -> dict:
        return OmegaConf.to_container(self.conf['data'][subset], resolve=True)

    def train_dataloader(self):
        d = self._subset_conf('train')
        return build_train_loader(
            d.get('manifest_dir'), d['cuts_file'], float(d['max_duration']),
            num_buckets=int(d.get('num_buckets', 30)),
            num_workers=int(d.get('num_workers', 8)),
            spec_augment=d.get('spec_augment'),
            on_the_fly=bool(d.get('on_the_fly', False)),
        )

    def _eval_loader(self, subset: str):
        d = self._subset_conf(subset)
        return build_eval_loader(
            d.get('manifest_dir'), d['cuts_file'], float(d['max_duration']),
            num_buckets=int(d.get('num_buckets', 30)),
            num_workers=int(d.get('num_workers', 8)),
        )

    def val_dataloader(self):
        return self._eval_loader('valid')

    def test_dataloader(self):
        return self._eval_loader('test')


def build_checkpoint_callbacks(conf) -> list:
    """conf.trainer_callbacks.model_checkpoints 列表 → ModelCheckpoint 列表。

    使用 LightningModule.configure_callbacks 钩子挂载（与 Trainer 回调合并），
    不改动框架 get_callbacks。典型配置：raw/EMA 双 checkpoint（SNN）或
    逐 epoch 全量保存（ANN，供 9–17 平均导出）。
    """
    cbs_conf = conf.get('trainer_callbacks', None)
    items = OmegaConf.to_container(cbs_conf, resolve=True).get('model_checkpoints', []) \
        if cbs_conf is not None else []
    if not items:
        return []
    dirpath = Path(conf.get('root_dir', '.')) / 'checkpoints'
    return [ModelCheckpoint(dirpath=dirpath, **item) for item in items]


# ═══════════════════════════════════════════════════════════════
#  ANN 教师
# ═══════════════════════════════════════════════════════════════

class AnnKwsTeacher(LhotseKwsMixin, LiteASR):
    """TdnnLstm ANN 教师——继承框架 LiteASR，仅做接口适配。

    框架复用：模型经 alpha.asr.models 注册表构建（conf.model.name=TdnnLstm）、
    损失经 alpha.asr.loss 注册表（SpikingCELoss）、训练/验证 step 与日志键
    （train|valid/asr_loss, loss_per, acc）全部来自 BaseASR/LiteASR。
    接口适配：lhotse dict 批次（inputs [B,T,F]→[B,F,T]）、LinearWarmupCosineLR
    调度器、每 epoch fix_random_seed(seed+epoch)、configure_callbacks 逐 epoch 存档。
    """

    def on_train_epoch_start(self):
        # 复刻原版：每 epoch fix_random_seed(seed + epoch)
        fix_random_seed(int(self.conf['system'].get('seed', 42)) + self.current_epoch)

    def forward(self, batch):
        feats = batch['inputs'].to(self.device).permute(0, 2, 1).float()  # [B,F,T]
        logits = self.model(feats)
        labels = extract_labels(batch, feats.device)
        return self.loss(logits, None, labels)

    def test_step(self, batch, batch_idx):
        loss, loss_info = self.forward(batch)
        loss_stat = self.loss.log_stat(loss_info)
        bs = batch['inputs'].shape[0]
        self.log('test/asr_loss', loss, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log('test/acc', loss_stat['token_acc'], prog_bar=True,
                 on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        opt_conf = OmegaConf.to_container(self.conf['optimizer'], resolve=True)
        name = opt_conf.pop('name', 'AdamW')
        scheduler_conf = opt_conf.pop('lr_scheduler', None) or {}
        optimizer = getattr(torch.optim, name)(self.parameters(), **opt_conf)
        if not scheduler_conf:
            return optimizer
        scheduler_name = scheduler_conf.pop('name', 'LinearWarmupCosineLR')
        if scheduler_name != 'LinearWarmupCosineLR':
            raise NotImplementedError(
                f'AnnKwsTeacher only supports LinearWarmupCosineLR, got {scheduler_name}')
        from alpha.asr.kws.schedulers import LinearWarmupCosineLR
        scheduler = LinearWarmupCosineLR(optimizer, **scheduler_conf)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def configure_callbacks(self):
        return build_checkpoint_callbacks(self.conf)


# ═══════════════════════════════════════════════════════════════
#  SNN 学生（KD 蒸馏）
# ═══════════════════════════════════════════════════════════════

class MultiTauKwsKd(LhotseKwsMixin, MultiTauSpikeASR):
    """MultiTauLIFCommander + TdnnLstm 教师蒸馏（experiment4 同配方）。

    conf 约定：
      model.name: MultiTauLIFCommander（框架路由要求），model.param 内含
                  subsampling_factor（AvgPool 下采样，无参数，checkpoint 无该键）
      system.kd:  {teacher: 权重路径, alpha: 0.7, temperature: 10.0}，可省略
      system.spec_augment: {time_mask, freq_mask, num_time_masks, num_freq_masks}
      optimizer.ema.decay: 0.9995（框架 EMA，验证/测试时应用）
      trainer_callbacks.model_checkpoints: [raw, ema] 双 checkpoint

    损失：L = (1-α)·SpikingCE(ls=0.1) + α·T²·KL(teacher‖student) + w_bb·branch_balance
    验证：每 batch 先 raw 前向记 valid/acc，再换入 EMA 权重记 valid/acc_ema。
    """

    def __init__(self, conf):
        super().__init__(conf)
        param = OmegaConf.to_container(conf['model']['param'], resolve=True)

        # ---- AvgPool 下采样（无参数；与历史 AlphaSNNForKWS 前向一致）----
        self.subsample_factor = int(param.get('subsampling_factor', 2))
        self.subsample = (
            nn.AvgPool1d(self.subsample_factor, stride=self.subsample_factor)
            if self.subsample_factor > 1 else nn.Identity())

        # ---- SpecAugment（批级遮罩，仅训练）----
        sa = conf['system'].get('spec_augment', None)
        self.spec_augment_conf = OmegaConf.to_container(sa, resolve=True) \
            if sa is not None else None

        # ---- KD 教师（冻结，eval）----
        kd = conf['system'].get('kd', None)
        self.teacher = None
        if kd is not None:
            kd_conf = OmegaConf.to_container(kd, resolve=True)
            self.kd_alpha = float(kd_conf.get('alpha', 0.7))
            self.kd_temperature = float(kd_conf.get('temperature', 10.0))
            self.teacher = TdnnLstm(
                num_features=int(param['n_inputs']),
                num_classes=int(param['n_outputs']),
                subsampling_factor=3,
            )
            ckpt = torch.load(kd_conf['teacher'], map_location='cpu',
                              weights_only=True)
            state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            self.teacher.load_state_dict(state)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            logger.info(f'KD teacher loaded: {kd_conf["teacher"]} '
                        f'(alpha={self.kd_alpha}, T={self.kd_temperature})')
        self._last_kd = None

    # ---- 前向（覆盖 SpikeASR 的 tuple-batch 版本，接收 lhotse dict batch）----

    def forward(self, batch):
        feats = batch['inputs'].to(self.device).permute(0, 2, 1).float()  # [B,F,T]
        if self.training and self.spec_augment_conf is not None:
            feats = spec_augment(feats, **self.spec_augment_conf)
        lengths = batch_lengths(batch, feats.device)

        x = self.subsample(feats).transpose(1, 2)  # [B,T',F]
        mask = subsampling_mask(x.shape[1], lengths, self.subsample_factor, x.device)

        functional.reset_net(self.model)
        logits = self.model(x, mask)
        labels = extract_labels(batch, x.device)
        loss, info = self.loss(logits, None, labels)

        if self.teacher is not None:
            functional.reset_net(self.teacher)
            with torch.no_grad():
                t_logits = self.teacher(feats)  # 原始未下采样特征（教师内部 stride3）
            t = self.kd_temperature
            kd = F.kl_div(
                F.log_softmax(logits / t, dim=1),
                F.softmax(t_logits / t, dim=1),
                reduction='batchmean') * t * t
            loss = (1 - self.kd_alpha) * loss + self.kd_alpha * kd
            self._last_kd = kd.detach()
            info['loss'] = loss.detach()
        else:
            self._last_kd = None
        return loss, info

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if self._last_kd is not None:
            self.log('train/kd_loss', self._last_kd, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch['inputs'].shape[0])
        return loss

    # ---- 双通道验证：raw 与 EMA 各评一次（覆盖 BaseModule 的整体 EMA 交换）----

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def _eval_forward(self, batch):
        with torch.no_grad():
            _, info = self.forward(batch)
        return info

    def validation_step(self, batch, batch_idx):
        info = self._eval_forward(batch)
        self.valid_log('valid/asr_loss', info['loss'])
        self.valid_log('valid/acc', info['token_acc'], prog_bar=True)
        if self.ema is None:
            return
        # 换入 EMA 权重再评一次（SNN 膜电位已在前向内 reset）
        self.ema.store(self.ema_parameters())
        try:
            self.ema.copy_to(self.ema_parameters())
            info_ema = self._eval_forward(batch)
            self.valid_log('valid/acc_ema', info_ema['token_acc'], prog_bar=True)
        finally:
            self.ema.restore(self.ema_parameters())

    def configure_callbacks(self):
        return build_checkpoint_callbacks(self.conf)


class SpikeCommanderKWS(LhotseKwsMixin, SpikeASR):
    """Lhotse KWS adapter for the original SpikeCommander model."""

    def __init__(self, conf):
        if conf['model']['name'] != 'SpikeCommander':
            raise ValueError('SpikeCommanderKWS requires model.name=SpikeCommander')
        super().__init__(conf)

    def forward(self, batch):
        x = batch['inputs'].to(self.device).float()   # [B, T, F]
        lengths = batch_lengths(batch, x.device)
        labels = extract_labels(batch, x.device)
        attention_mask = self._make_attention_mask(lengths, x)

        functional.reset_net(self.model)
        model_out = self.model(x, attention_mask)

        logits, loss_mask = self._prepare_loss_input(model_out, attention_mask)
        return self.loss(logits, loss_mask, labels)

    def _classification_step(self, split, batch):
        asr_loss, loss_info = self.forward(batch)
        loss_stat = self.loss.log_stat(loss_info)
        batch_size = batch['inputs'].shape[0]
        log_kwargs = {
            'on_step': split == 'train',
            'on_epoch': True,
            'sync_dist': True,
            'batch_size': batch_size,
        }
        self.log(f'{split}/asr_loss', asr_loss, **log_kwargs)
        self.log(f'{split}/loss_per', loss_stat['loss'], **log_kwargs)
        self.log(f'{split}/acc', loss_info['token_acc'],
                 on_epoch=True, sync_dist=True,
                 batch_size=batch_size, prog_bar=True)
        return asr_loss

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_batches_seen += 1
        self.test_samples_seen += batch['inputs'].shape[0]
        functional.reset_net(self.model)


__all__ = ['AnnKwsTeacher', 'MultiTauKwsKd', 'SpikeCommanderKWS']
