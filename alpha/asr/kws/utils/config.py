"""在线/评测/导出的 YAML 配置加载与路径解析。

产物目录与离线训练（Lightning）对齐：
  exp/logs/<exp_name>/<version>/{log.log, hparams.yaml, tensorboard/, checkpoints/}
"""

import logging
from pathlib import Path

from .attribute_dict import AttributeDict
from omegaconf import OmegaConf

# 项目根目录 = AlphaSNN（框架根；utils → kws → asr → alpha → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def load_config(config_path) -> AttributeDict:
    """加载 YAML 配置，返回 AttributeDict。

    - root_dir/exp_dir = exp/logs/<system.exp_name>/<system.version>（与离线对齐）
    - pretrained_model / manifest_dir 等相对路径以项目根 AlphaSNN 为基准
    """
    conf = OmegaConf.load(config_path)
    params = _to_attr_dict(OmegaConf.to_container(conf, resolve=True))
    resolve_paths(params)
    return params


def _to_attr_dict(obj):
    """递归将嵌套 dict 转换为 AttributeDict。"""
    if isinstance(obj, dict):
        return AttributeDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj


def resolve_paths(params: AttributeDict) -> None:
    """把配置中的相对路径解析为绝对路径，并确定产物目录。"""
    sys = params.system

    exp_name = getattr(sys, 'exp_name', 'carKWS_online')
    version = str(getattr(sys, 'version', 'online01'))
    root_dir = PROJECT_ROOT / 'exp' / 'logs' / exp_name / version
    sys.root_dir = root_dir
    sys.exp_dir = root_dir  # 兼容既有引用（adakws_trainer 存档等）
    sys.log_file = root_dir / getattr(sys, 'log_filename', 'log.log')

    params.paths.pretrained_model = _abs(params.paths.pretrained_model)

    node = params.data.target
    for key in ('fbank_dir', 'manifest_dir'):
        if key in node:
            node[key] = _abs(node[key])


def _abs(p, base=PROJECT_ROOT):
    """Path 转绝对路径：已绝对则原样，相对则基于 base 解析。"""
    p = Path(p)
    return p if p.is_absolute() else base / p


def log_config(params: AttributeDict) -> None:
    """打印配置。icefall 的 AttributeDict.__str__ 内部用 json.dumps，
    遇到 Path 会报错，所以这里手动转字符串再打。"""
    import json

    logging.info(json.dumps(params, default=str, indent=2, ensure_ascii=False))


def dump_hparams(params: AttributeDict, out_path: Path) -> None:
    """把运行配置快照写入 hparams.yaml（与离线 Lightning 产物结构对齐）。"""
    import json

    import yaml

    plain = json.loads(json.dumps(params, default=str))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(plain, f, allow_unicode=True, sort_keys=False)
