import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from torch_ema import ExponentialMovingAverage
from modules.activation_based import surrogate
from modules.activation_based.compat import ensure_spikingjelly_cupy_numpy_compat
from modules.blocks.spike_fsmn.multi_tau_lif import MultiTauLIFMemory
from modules.system import BaseModule
from . import models
from . import loss as loss_module
from modules.utils.logging import logger

from spikingjelly.activation_based import functional
from spikingjelly.datasets import padded_sequence_mask
from modules.utils.spike_rate import SpikeRateMonitor

class BaseASR(BaseModule):
    def __init__(self, conf):
        super().__init__(conf)
        self.model_class = getattr(models, conf['model']['name'])
        self.resolve_model_param(conf['model']['name'])
        self.loss = self.init_loss(conf['loss'])
        self.use_spike_input = conf['data'].get('use_spike_input', False)
        if not self.use_spike_input and conf['data'].get('feats') is not None:
            assert self.feat_extractor is not None
    
    def resolve_model_param(self, name):
        model_param = self.conf['model'].get('param', None)
        if model_param is None:
            logger.info('No model param specified, using default config in model code')
            return
        if isinstance(model_param, (dict, DictConfig)):
            self.conf['model']['param'] = model_param.get(name, model_param)
            logger.info('Using inline model param config')
            return
        if not isinstance(model_param, str):
            logger.info('Using model param config with unsupported resolver type: {}', type(model_param).__name__)
            return
        if model_param.endswith('.yaml'):
            model_param = OmegaConf.load(Path('conf').joinpath(model_param))
        self.conf['model']['param'] = model_param[name]
        
    def init_loss(self, loss_conf):
        assert 'name' in loss_conf
        self.loss_name = loss_conf.pop('name')
        if hasattr(loss_module, self.loss_name):
            loss = getattr(loss_module, self.loss_name)(loss_conf)
        elif hasattr(torch.nn, self.loss_name):
            loss = getattr(torch.nn, self.loss_name)()
        else:
            raise ValueError(f'loss name error: {self.loss_name}')
        return loss
    
    def init_model(self, model_conf):
        model_path = model_conf.get('init')
        if model_path:
            logger.info(f'Model initialize with: {model_path}')
            model_dict = torch.load(model_path)
            assert 'model' in model_dict  # pytorch asr ckpt
            self.model.load_state_dict(model_dict['model'])
    
    def training_step(self, batch, batch_idx):
        asr_loss, loss_info = self.forward(batch)
        loss_stat = self.loss.log_stat(loss_info)
        self.log_dict({'train/asr_loss': asr_loss, 'train/loss_per': loss_stat['loss']})
        self.log('train/acc', loss_stat['token_acc'], prog_bar=True)
        return asr_loss
        
    def validation_step(self, batch, batch_idx):
        asr_loss, loss_info = self.forward(batch)
        loss_stat = self.loss.log_stat(loss_info)
        self.valid_log('valid/asr_loss', asr_loss)
        self.valid_log('valid/loss_per', loss_stat['loss'])
        self.valid_log('valid/acc', loss_stat['token_acc'], prog_bar=True)

    def on_validation_epoch_end(self):
        pass
        
        
class LiteASR(BaseASR):
    def __init__(self, conf):
        super().__init__(conf)
        self.model = self.model_class(**self.conf['model']['param'])
        
    def forward(self, batch):
        if 'feats' in batch:
            feats = batch['feats']
        else:
            feats = self.feat_extractor.get_feat_batch(batch['noisy_wav'], batch['wav_lengths'])
        model_out = self.model(feats)
        loss, loss_info = self.loss(model_out, feats, batch['label'])
        return loss, loss_info


class SpikeASR(BaseASR):
    def __init__(self, conf):
        super().__init__(conf)
        self.best_acc = {'train': 0.0, 'valid': 0.0, 'test': 0.0}
        self.test_batches_seen = 0
        self.test_samples_seen = 0
        self.time_reduce = self.conf['loss'].get('time_reduce', self.conf['loss'].get('reduce', 'sum'))
        model_param = self._prepare_spike_param(self.conf['model']['param'])
        self.model = self.model_class(model_param)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Loss Function: {self.loss_name}')
        logger.info(f'Model parameters (trainable): {trainable_params:,}')
        _monitor = SpikeRateMonitor(self.model)
        self._spike_monitor = _monitor if _monitor.is_active else None

        _ema_conf = (conf.get('optimizer') or {}).get('ema')
        if _ema_conf is not None:
            self.ema_decay = _ema_conf.get('decay')
            self.ema_stop_epoch = _ema_conf.get('stop_epoch')
        else:
            self.ema_stop_epoch = None

    def _prepare_spike_param(self, param):
        if isinstance(param, DictConfig):
            param = OmegaConf.to_container(param, resolve=True)

        if param.get('backend') == 'cupy':
            ensure_spikingjelly_cupy_numpy_compat()

        if param.get('surrogate_function') is not None:
            return param
        surrogate_name = param.get('surrogate', 'atan')
        if surrogate_name == 'atan':
            alpha = param.get('alpha', 2.0)
            param['surrogate_function'] = surrogate.ATan(alpha=alpha)
            return param
        raise ValueError(f'Unsupported surrogate: {surrogate_name}')

    def _reduce_time(self, outputs, attention_mask=None):
        if outputs.dim() == 2:
            return outputs
        if outputs.dim() != 3:
            raise ValueError(f'Expected 2-D or 3-D model outputs, got {tuple(outputs.shape)}')

        outputs, time_mask = self._to_time_major(outputs, attention_mask)
        reduce = self.time_reduce
        if reduce == 'mean':
            if time_mask is None:
                return outputs.mean(dim=0)
            mask = time_mask.to(dtype=outputs.dtype)
            denom = mask.sum(dim=0).clamp_min(1.0)
            return (outputs * mask).sum(dim=0) / denom
        if reduce == 'max':
            if time_mask is not None:
                outputs = outputs.masked_fill(~time_mask, torch.finfo(outputs.dtype).min)
            return outputs.max(dim=0).values
        if reduce == 'spike_count':
            if time_mask is not None:
                outputs = outputs * time_mask.to(dtype=outputs.dtype)
            return outputs.sum(dim=0)
        if reduce == 'sum':
            probs = torch.softmax(outputs, dim=2)
            if time_mask is not None:
                probs = probs * time_mask.to(dtype=probs.dtype)
            return probs.sum(dim=0)
        raise ValueError(f'Unsupported time_reduce: {reduce}')

    def _to_time_major(self, outputs, attention_mask=None):
        if attention_mask is None:
            # Historical SpikeASR/SpikeCommander contract: [T, B, C].
            return outputs, None
        if attention_mask.dim() != 2:
            raise ValueError(f'Expected attention_mask shape [B, T], got {tuple(attention_mask.shape)}')

        batch, time = attention_mask.shape
        if outputs.shape[0] == time and outputs.shape[1] == batch:
            time_major = outputs
        elif outputs.shape[0] == batch and outputs.shape[1] == time:
            time_major = outputs.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                'Model output shape is incompatible with attention_mask: '
                f'outputs={tuple(outputs.shape)}, attention_mask={tuple(attention_mask.shape)}'
            )
        time_mask = attention_mask.transpose(0, 1).unsqueeze(-1).to(device=outputs.device, dtype=torch.bool)
        return time_major, time_mask

    def _make_attention_mask(self, lengths, x):
        return padded_sequence_mask(lengths).transpose(0, 1).to(device=x.device, dtype=torch.bool)

    def _prepare_loss_input(self, model_out, attention_mask):
        if getattr(self.loss, 'sequence_loss', False):
            logits, _ = self._to_time_major(model_out, attention_mask)
            return logits.transpose(0, 1).contiguous(), attention_mask
        return self._reduce_time(model_out, attention_mask), None

    def forward(self, batch):
        x, labels, lengths = batch
        x = x.float()
        labels = labels.long()
        attention_mask = self._make_attention_mask(lengths, x)
        model_out = self.model(x, attention_mask)
        logits, loss_mask = self._prepare_loss_input(model_out, attention_mask)
        loss, loss_info = self.loss(logits, loss_mask, labels)
        return loss, loss_info

    def training_step(self, batch, batch_idx):
        return self._classification_step('train', batch)

    def validation_step(self, batch, batch_idx):
        self._classification_step('valid', batch)

    def test_step(self, batch, batch_idx):
        return self._classification_step('test', batch)

    def _classification_step(self, split, batch):
        asr_loss, loss_info = self.forward(batch)
        loss_stat = self.loss.log_stat(loss_info)
        batch_size = batch[1].numel()
        loss_log_kwargs = {
            'on_step': split == 'train',
            'on_epoch': True,
            'sync_dist': True,
            'batch_size': batch_size,
        }
        metric_log_kwargs = {
            'on_step': False,
            'on_epoch': True,
            'sync_dist': True,
            'batch_size': batch_size,
        }
        self.log(f'{split}/asr_loss', asr_loss, **loss_log_kwargs)
        self.log(f'{split}/loss_per', loss_stat['loss'], **loss_log_kwargs)
        self.log(f'{split}/acc', loss_info['token_acc'], prog_bar=True, **metric_log_kwargs)
        return asr_loss

    def _update_best_acc(self, split):
        metric = self.trainer.callback_metrics.get(f'{split}/acc')
        if metric is None:
            return
        current_acc = float(metric.detach().cpu()) if torch.is_tensor(metric) else float(metric)
        self.best_acc[split] = max(self.best_acc[split], current_acc)
        self.log(f'{split}/best_acc', self.best_acc[split], prog_bar=split != 'train',
                 sync_dist=True)

    def on_train_epoch_start(self):
        if self._spike_monitor is not None:
            self._spike_monitor.reset()

    def on_validation_epoch_start(self):
        if self._spike_monitor is not None:
            self._spike_monitor.reset()

    def setup_ema(self):
        self.ema_parameters = self.parameters
        ema_decay = getattr(self, 'ema_decay', None)
        if ema_decay is not None and self.ema is None:
            self.ema = ExponentialMovingAverage(self.ema_parameters(), decay=ema_decay)

    def on_train_epoch_end(self):
        if (self.ema is not None
                and self.ema_stop_epoch is not None
                and self.current_epoch == self.trainer.max_epochs - self.ema_stop_epoch - 1):
            self.ema.copy_to(self.ema_parameters())
            self.ema = None
            logger.info(f'EMA disabled: weights promoted after epoch {self.current_epoch + 1}.')
        self._update_best_acc('train')
        if self._spike_monitor is not None:
            rate = self._spike_monitor.get_mean_rate()
            if rate is not None:
                self.log('train/spike_rate', rate, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self._update_best_acc('valid')
            if self._spike_monitor is not None:
                rate = self._spike_monitor.get_mean_rate()
                if rate is not None:
                    self.log('valid/spike_rate', rate, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self):
        self._update_best_acc('test')
        if self._spike_monitor is not None:
            rate = self._spike_monitor.get_mean_rate()
            if rate is not None:
                self.log('test/spike_rate', rate, on_epoch=True, sync_dist=True)
        metrics = self.trainer.callback_metrics
        test_acc = metrics.get('test/acc')
        test_loss = metrics.get('test/asr_loss')
        if test_acc is None or test_loss is None:
            return

        count_tensor = torch.tensor(
            [self.test_batches_seen, self.test_samples_seen],
            device=self.device,
            dtype=torch.long,
        )
        if self.trainer.world_size > 1:
            count_tensor = self.all_gather(count_tensor).sum(dim=0)

        if not self.trainer.is_global_zero:
            return

        test_acc = float(test_acc.detach().cpu()) if torch.is_tensor(test_acc) else float(test_acc)
        test_loss = float(test_loss.detach().cpu()) if torch.is_tensor(test_loss) else float(test_loss)
        logger.info(
            'Test finished | batches: {}, samples: {}, test loss: {:.4f}, '
            'test acc: {:.2f}%, best test acc: {:.2f}%'.format(
                int(count_tensor[0].item()),
                int(count_tensor[1].item()),
                test_loss,
                test_acc * 100.0,
                self.best_acc['test'] * 100.0,
            )
        )

    def _best_acc_summary(self):
        return (
            'Training finished | Best train accuracy: '
            f'{self.best_acc["train"]:.2%}, Best valid accuracy: '
            f'{self.best_acc["valid"]:.2%}'
        )

    def on_train_end(self):
        if not self.trainer.is_global_zero:
            return
        logger.info(self._best_acc_summary())

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        functional.reset_net(self.model)
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        functional.reset_net(self.model)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.test_batches_seen += 1
        self.test_samples_seen += batch[1].numel()
        functional.reset_net(self.model)

    def on_test_start(self):
        super().on_test_start()
        self.test_batches_seen = 0
        self.test_samples_seen = 0
        if self._spike_monitor is not None:
            self._spike_monitor.reset()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint['best_acc'] = self.best_acc

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.best_acc.update(checkpoint.get('best_acc', {}))


class MultiTauSpikeASR(SpikeASR):
    """SpikeASR specialization for MultiTauLIFCommander training and analysis."""

    def __init__(self, conf):
        if conf['model']['name'] != 'MultiTauLIFCommander':
            raise ValueError('MultiTauSpikeASR requires model.name=MultiTauLIFCommander')
        super().__init__(conf)
        self.record_branch_weights = bool(self.conf['system'].get('record_branch_weights', False))
        self.record_every_epoch = int(self.conf['system'].get('record_every_epoch', 1))
        if self.record_every_epoch < 1:
            raise ValueError('system.record_every_epoch must be >= 1')
        self._last_branch_weight_snapshot_epoch = None
        self.branch_balance_weight = float(self.conf['loss'].get('branch_balance_weight', 0.0))
        if self.branch_balance_weight < 0:
            raise ValueError('loss.branch_balance_weight must be non-negative')
        self._set_branch_activity_tracking(False)

    def _prepare_spike_param(self, param):
        """Keep MultiTau surrogate configuration declarative and model-owned."""
        if isinstance(param, DictConfig):
            param = OmegaConf.to_container(param, resolve=True)
        if param.get('backend') == 'cupy':
            ensure_spikingjelly_cupy_numpy_compat()
        return param

    def _classification_step(self, split, batch):
        asr_loss = super()._classification_step(split, batch)
        if split != 'train':
            return asr_loss
        branch_balance_loss = self.model.branch_balance_loss()
        self.log(
            'train/branch_balance_loss',
            branch_balance_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[1].numel(),
        )
        return asr_loss + self.branch_balance_weight * branch_balance_loss

    def _optimizer_parameter_groups(self, base_weight_decay, tau_weight_decay, branch_weight_decay):
        tau_parameters, branch_parameters = self.model.multi_tau_parameters()
        tau_ids = {id(parameter) for parameter in tau_parameters}
        branch_ids = {id(parameter) for parameter in branch_parameters}
        if tau_ids & branch_ids:
            raise RuntimeError('tau and branch fusion optimizer parameter groups overlap')

        main_parameters = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
            and id(parameter) not in tau_ids
            and id(parameter) not in branch_ids
        ]
        groups = []
        if main_parameters:
            groups.append({'params': main_parameters, 'weight_decay': base_weight_decay})
        if tau_parameters:
            groups.append({'params': tau_parameters, 'weight_decay': tau_weight_decay})
        if branch_parameters:
            groups.append({'params': branch_parameters, 'weight_decay': branch_weight_decay})
        return groups

    def configure_optimizers(self):
        opt_conf = OmegaConf.to_container(self.conf['optimizer'], resolve=True)
        assert isinstance(opt_conf, dict) and 'name' in opt_conf
        base_weight_decay = float(opt_conf.get('weight_decay', 0.0))
        tau_weight_decay = float(opt_conf.pop('tau_weight_decay', 0.0))
        branch_weight_decay = float(opt_conf.pop('branch_weight_decay', 0.0))
        if tau_weight_decay < 0 or branch_weight_decay < 0:
            raise ValueError('optimizer tau/branch weight decay must be non-negative')
        parameter_groups = self._optimizer_parameter_groups(
            base_weight_decay,
            tau_weight_decay,
            branch_weight_decay,
        )
        return self.get_optimizer(opt_conf, parameter_groups)

    def _save_branch_weight_snapshot(self, stage):
        if not self.trainer.is_global_zero or not self._should_record_branch_snapshot(stage):
            return None
        if stage == 'valid' and self._last_branch_weight_snapshot_epoch == self.current_epoch:
            return None

        module_names = []
        arrays = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, MultiTauLIFMemory):
                continue
            snapshot = module.get_branch_weight_snapshot()
            if snapshot is None:
                continue
            index = len(module_names)
            module_names.append(name)
            for key, value in snapshot.items():
                arrays[f'module_{index}_{key}'] = value.cpu().numpy()

        if not module_names:
            return None

        arrays.update({
            'format_version': np.asarray(2, dtype=np.int64),
            'stage': np.asarray(stage, dtype=np.str_),
            'epoch': np.asarray(self.current_epoch, dtype=np.int64),
            'global_step': np.asarray(self.global_step, dtype=np.int64),
            'module_names': np.asarray(module_names, dtype=np.str_),
        })
        root_dir = Path(self.conf.get('root_dir', self.trainer.default_root_dir))
        snapshot_dir = root_dir / 'branch_weights'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / (
            f'{stage}-epoch={self.current_epoch}-step={self.global_step}.npz'
        )
        np.savez_compressed(snapshot_path, **arrays)
        if stage == 'valid':
            self._last_branch_weight_snapshot_epoch = self.current_epoch
        logger.info(f'Branch-weight snapshot saved: {snapshot_path}')
        return snapshot_path

    def _should_record_branch_snapshot(self, stage):
        if not self.record_branch_weights:
            return False
        if stage == 'test':
            return True
        if stage != 'valid':
            return False
        epoch_number = self.current_epoch + 1
        max_epochs = getattr(self.trainer, 'max_epochs', None)
        is_final_epoch = max_epochs is not None and max_epochs > 0 and epoch_number >= max_epochs
        return epoch_number % self.record_every_epoch == 0 or is_final_epoch

    def _set_branch_activity_tracking(self, enabled):
        for module in self.model.modules():
            if isinstance(module, MultiTauLIFMemory):
                module.set_branch_activity_tracking(enabled)

    def on_train_epoch_start(self):
        self._set_branch_activity_tracking(False)
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        should_track = (
            self.trainer.is_global_zero
            and self._should_record_branch_snapshot('valid')
        )
        self._set_branch_activity_tracking(should_track)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if not self.trainer.sanity_checking:
            self._save_branch_weight_snapshot('valid')

    def on_test_start(self):
        super().on_test_start()
        self._set_branch_activity_tracking(
            self.trainer.is_global_zero and self._should_record_branch_snapshot('test')
        )

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._save_branch_weight_snapshot('test')

# KWS 任务系统注册（中文车载 29 类指令，详见 alpha/asr/kws/system.py）。
# 放在文件末尾：本模块先完成定义，kws.system 再引用上面的类，避免循环导入。
try:
    from .kws.system import AnnKwsTeacher, MultiTauKwsKd, SpikeCommanderKWS  # noqa: F401,E402
except ImportError as e:  # lhotse 等可选依赖缺失时跳过，不影响其他任务
    logger.warning(f'KWS systems not registered: {e}')
