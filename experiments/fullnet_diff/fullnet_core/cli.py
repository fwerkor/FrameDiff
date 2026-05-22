from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import build_run_config, load_config, write_config
from .models import available_models
from .paths import CONFIG_PATH, OUTPUT_ROOT, PROJECT_ROOT
from .runner import run_fullnet


def _parse_models(value: list[str] | None) -> list[str] | None:
    if value is None:
        return None
    items: list[str] = []
    for item in value:
        items.extend(part.strip() for part in item.split(",") if part.strip())
    return items


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", help="模型名或逗号分隔列表，如 qwen2 glm4")
    parser.add_argument("--iters", type=int, help="任务迭代次数")
    parser.add_argument("--pta-path", help="PTA/MindSpeed 代码路径")
    parser.add_argument("--msa-path", help="MSA 代码路径")
    parser.add_argument("--pta-env", help="PTA conda 环境名")
    parser.add_argument("--msa-env", help="MSA conda 环境名")
    parser.add_argument("--save-steps", type=int, help="PTA-SAVE 步数")
    parser.add_argument("--load-steps", type=int, help="PTA/MSA LOAD 步数")
    parser.add_argument("--mutnm", type=int, help="每轮变异配置数")
    parser.add_argument("--base-seed", type=int, help="基础随机种子")
    parser.add_argument("--trace", action="store_true", help="开启论文实验用调试/张量摘要环境")
    parser.add_argument("--debug-compare", action="store_true", help="输出更详细的层级张量摘要")


def _config_from_args(args: argparse.Namespace) -> dict:
    return build_run_config(
        load_config(CONFIG_PATH),
        models=_parse_models(getattr(args, "models", None)),
        total_iter=getattr(args, "iters", None),
        pta_path=getattr(args, "pta_path", None),
        msa_path=getattr(args, "msa_path", None),
        pta_env=getattr(args, "pta_env", None),
        msa_env=getattr(args, "msa_env", None),
        save_steps=getattr(args, "save_steps", None),
        load_steps=getattr(args, "load_steps", None),
        mutnm=getattr(args, "mutnm", None),
        base_seed=getattr(args, "base_seed", None),
        trace=True if getattr(args, "trace", False) else None,
        debug_compare=True if getattr(args, "debug_compare", False) else None,
    )


def cmd_models(_args: argparse.Namespace) -> int:
    for model in available_models():
        print(model)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    target = Path(args.output).resolve() if args.output else CONFIG_PATH
    write_config(config, target)
    print(f"配置已写入: {target}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    fullnet = config["fullnet"]
    print(f"project: {PROJECT_ROOT}")
    print(f"models: {', '.join(fullnet['MODELS'])}")
    print("compare: pta_msa")
    print(f"iters: {fullnet['TOTAL_ITER']}")
    for key in ("PTA_PATH", "MSA_PATH"):
        value = str(config.get(key, ""))
        exists = Path(value).expanduser().exists() if value and not value.startswith("<") else False
        print(f"{key}: {value} ({'ok' if exists else 'check'})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if args.write_config:
        write_config(config, CONFIG_PATH)
    return run_fullnet(config)


def _latest_output() -> Path:
    candidates = [path for path in OUTPUT_ROOT.glob("*") if (path / "iters").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"未找到可分析 output: {OUTPUT_ROOT}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def cmd_analyze(args: argparse.Namespace) -> int:
    target = _latest_output() if args.latest or not args.output else Path(args.output).expanduser().resolve()
    run_dir = target / "iters"
    config = load_config(target / "config.json")
    fullnet = config.get("fullnet")
    if not isinstance(fullnet, dict):
        raise ValueError(f"配置缺少 fullnet: {target / 'config.json'}")
    models = fullnet.get("MODELS") or []
    planned = int(fullnet.get("TOTAL_ITER", 0) or 0)
    from utils.analyze.fullnet_result import analyze_fullnet_run

    result = analyze_fullnet_run(
        output_root=target,
        run_dir=run_dir,
        model_name=",".join(models),
        planned_iterations=planned,
    )
    payload = {
        "analysis_dir": str(result.analysis_dir),
        "report_html": str(result.report_html),
        "summary_json": str(result.summary_json),
        "executed_iterations": result.executed_iterations,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FrameDiff language-model full-network CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="列出可选语言模型").set_defaults(func=cmd_models)

    init = sub.add_parser("init", help="生成整网配置")
    _add_common_run_args(init)
    init.add_argument("-o", "--output", help="输出配置路径，默认 config.json")
    init.set_defaults(func=cmd_init)

    doctor = sub.add_parser("doctor", help="检查模型选择和关键路径")
    _add_common_run_args(doctor)
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run", help="进入整网链路")
    _add_common_run_args(run)
    run.add_argument("--dry-run", action="store_true", help="只打印最终配置，不启动训练链路")
    run.add_argument("--write-config", action="store_true", help="运行前同步写回 config.json")
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze", help="重建整网分析报告")
    analyze.add_argument("output", nargs="?", default="", help="output/<id> 或绝对路径")
    analyze.add_argument("--latest", action="store_true", help="分析最近一次 output")
    analyze.set_defaults(func=cmd_analyze)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
