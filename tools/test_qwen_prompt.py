from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Priors import qwen_vl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Qwen prompt with random trajectory + map.")
    parser.add_argument("--provider", type=str, default="siliconflow", choices=["local", "siliconflow"])
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--prompt-text", type=str, default="")
    parser.add_argument("--prompt-file", type=str, default="")
    parser.add_argument("--image", type=str, default="", help="Optional image path (fallback)")
    parser.add_argument("--cache", type=str, default="", help="Trajectory cache .pth")
    parser.add_argument("--map", type=str, default="", help="Map image path")
    parser.add_argument("--norm-stats", type=str, default="Dataset/Indoor_norm_stats.json")
    parser.add_argument("--erase-rate", type=float, default=0.5)
    parser.add_argument("--map-extent", type=str, default="0,16,0,30", help="x_min,x_max,y_min,y_max")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--test-len", type=int, default=0, help="Use only first N points (0 means full length).")
    parser.add_argument("--test-len-mode", type=str, default="head", choices=["head", "random"])
    parser.add_argument("--save-prompt", type=str, default="", help="Write composed prompt to file")
    parser.add_argument("--save-image", type=str, default="", help="Write rendered image to file")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-path", type=str, default="/chat/completions")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--save-output", type=str, default="", help="Write raw output to file")
    return parser.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_text:
        return args.prompt_text
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return input("Prompt: ").strip()


def _parse_extent(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("map-extent must be x_min,x_max,y_min,y_max")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _load_trajs(cache_path: str) -> list[torch.Tensor]:
    dataset_part = torch.load(cache_path, map_location="cpu")
    trajs: list[torch.Tensor] = []
    for sample in dataset_part:
        if isinstance(sample, (list, tuple)) and len(sample) > 0:
            traj = sample[0]
        else:
            traj = sample
        if not isinstance(traj, torch.Tensor):
            continue
        trajs.append(traj.to(torch.float32))
    return trajs


def _build_erase_mask(length: int, erase_rate: float, generator: torch.Generator) -> Optional[torch.Tensor]:
    if length < 2:
        return None
    n_remain = length - int(length * erase_rate)
    n_remain = max(2, min(length, n_remain))
    if length <= 2:
        remain_indices = torch.tensor([0, length - 1], dtype=torch.long)
    else:
        n_middle = max(0, n_remain - 2)
        perm = torch.randperm(length - 2, generator=generator)[:n_middle] + 1
        perm = torch.sort(perm)[0]
        remain_indices = torch.cat([torch.tensor([0]), perm, torch.tensor([length - 1])])
    mask = torch.ones(length, dtype=torch.float32)
    mask[remain_indices] = 0.0
    return mask


def _encode_image_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _run_siliconflow(
    prompt: str,
    *,
    image_path: Optional[str],
    model_id: str,
    max_new_tokens: int,
    api_base: Optional[str],
    api_key: Optional[str],
    api_path: str,
    timeout: float,
) -> str:
    import urllib.request

    api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    api_base = api_base or os.getenv("SILICONFLOW_API_BASE")
    if not api_key:
        raise RuntimeError("Missing SILICONFLOW_API_KEY (or pass --api-key).")
    if not api_base:
        raise RuntimeError("Missing SILICONFLOW_API_BASE (or pass --api-base).")

    api_base = api_base.rstrip("/")
    api_path = api_path if api_path.startswith("/") else f"/{api_path}"
    url = f"{api_base}{api_path}"

    content = []
    if image_path:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        image_b64 = _encode_image_base64(image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(max_new_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resp_text = response.read().decode("utf-8")
    data = json.loads(resp_text)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty response: {resp_text[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
        return "".join(texts).strip()
    return str(content).strip()


def _resolve_vision_model_class():
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except Exception:
        pass
    try:
        from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except Exception as exc:
        raise RuntimeError(
            "AutoModelForVision2Seq is not available. "
            "Please install transformers>=4.41 and ensure torchvision matches torch."
        ) from exc


def _run_local(
    prompt: str,
    *,
    image_path: Optional[str],
    model_id: str,
    max_new_tokens: int,
    force_download: bool,
    trust_remote_code: bool,
) -> str:
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "AutoProcessor import failed. Please ensure torchvision is installed "
            "and matches your torch version."
        ) from exc

    AutoModelForVision2Seq = _resolve_vision_model_class()

    processor = AutoProcessor.from_pretrained(
        model_id,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        force_download=force_download,
        trust_remote_code=trust_remote_code,
    )

    image = None
    if image_path:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": prompt})

    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt_text], images=[image] if image is not None else None, return_tensors="pt")
    else:
        inputs = processor(text=[prompt], images=[image] if image is not None else None, return_tensors="pt")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output.strip()


def main() -> int:
    args = parse_args()
    prompt_header = _load_prompt(args)
    if not prompt_header:
        print("Empty prompt.")
        return 1

    image_path = args.image.strip() or None
    if args.cache and args.map:
        trajs = _load_trajs(args.cache)
        if not trajs:
            print("Empty cache.")
            return 1
        rng = random.Random(int(args.seed))
        if args.sample_index >= 0:
            traj = trajs[args.sample_index % len(trajs)]
        else:
            traj = rng.choice(trajs)
        if args.test_len and int(args.test_len) > 0 and traj.shape[1] > int(args.test_len):
            target_len = int(args.test_len)
            if args.test_len_mode == "random":
                start = rng.randint(0, int(traj.shape[1]) - target_len)
            else:
                start = 0
            traj = traj[:, start : start + target_len]

        torch_gen = torch.Generator().manual_seed(int(args.seed))
        mask = _build_erase_mask(int(traj.shape[1]), float(args.erase_rate), torch_gen)
        if mask is None:
            print("Trajectory too short.")
            return 1

        mean, std = qwen_vl.load_norm_stats(args.norm_stats)
        xy_denorm = qwen_vl.denorm_xy(traj, mean, std)
        time_denorm = traj[2] * std[2] + mean[2]

        map_extent = _parse_extent(args.map_extent)
        prompt_table = qwen_vl._format_traj_for_prompt(xy_denorm, time_denorm, mask)
        prompt = f"{prompt_header}\n\n轨迹表：\n{prompt_table}\n"

        save_image = args.save_image.strip()
        if not save_image:
            save_image = "/tmp/qwen_prompt_render.png"
        qwen_vl.render_traj_on_map(
            xy_denorm,
            mask,
            args.map,
            map_extent=map_extent,
            render_out_path=save_image,
        )
        image_path = save_image
    else:
        prompt = prompt_header

    if args.save_prompt:
        with open(args.save_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)

    if args.provider == "siliconflow":
        text = _run_siliconflow(
            prompt,
            image_path=image_path,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            api_base=args.api_base or None,
            api_key=args.api_key or None,
            api_path=args.api_path,
            timeout=float(args.api_timeout),
        )
    else:
        text = _run_local(
            prompt,
            image_path=image_path,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
            force_download=bool(args.force_download),
            trust_remote_code=bool(args.trust_remote_code),
        )

    if args.save_output:
        with open(args.save_output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
