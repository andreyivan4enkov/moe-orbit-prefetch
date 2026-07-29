"""
DeepSeek-V2-Lite-Chat: sparse runtime + modeled prefetch.

Оптимизации под Chat:
  - короткий system / chat_template
  - modeled prefetch horizon=2
  - device-pack экспертов
  - progress throttle
  - cancel / unload / reload
"""
from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .sparse_moe_runtime import MID, SparseDeepseekRuntime

_runtime: SparseDeepseekRuntime | None = None
_tokenizer: Any = None
_load_error: str | None = None
_cancel = threading.Event()

ProgressCb = Callable[[dict[str, Any]], None]


def get_cancel_event() -> threading.Event:
    return _cancel


def request_cancel() -> None:
    """Прервать текущую генерацию (между слоями/токенами)."""
    _cancel.set()
    if _runtime is not None:
        _runtime.request_cancel()


def clear_cancel() -> None:
    _cancel.clear()
    if _runtime is not None:
        _runtime.cancel_flag = _cancel


def unload_engine() -> dict[str, Any]:
    """Выгрузить spine/экспертов из RAM (для перезапуска)."""
    global _runtime, _tokenizer, _load_error
    request_cancel()
    info: dict[str, Any] = {"unloaded": False}
    if _runtime is not None:
        st = _runtime.stats()
        info["before"] = {
            "spine_bytes": st.get("spine_bytes"),
            "n_hot": st.get("n_hot"),
            "resident_bytes": st.get("resident_bytes"),
        }
        _runtime.spine.clear()
        _runtime._pack_dev.clear()
        if _runtime.experts is not None:
            _runtime.experts.hot.clear()
            _runtime.experts.s_env.clear()
        _runtime = None
    _tokenizer = None
    _load_error = None
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    info["unloaded"] = True
    return info


def get_engine(*, offload_dir: Path | None = None, force_reload: bool = False):
    global _runtime, _tokenizer, _load_error
    del offload_dir
    if force_reload:
        unload_engine()
        clear_cancel()
    if _runtime is not None and _tokenizer is not None and not force_reload:
        _runtime.cancel_flag = _cancel
        return _runtime, _tokenizer
    if _load_error and not force_reload:
        raise RuntimeError(_load_error)
    try:
        clear_cancel()
        tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
        print(
            "Loading SPARSE+MODELED runtime for DeepSeek-V2-Lite-Chat "
            "(NOT full from_pretrained)..."
        )
        rt = SparseDeepseekRuntime.load(
            device="cpu",
            dtype=__import__("torch").float16,
            use_modeled_prefetch=True,
            prefetch_horizon=2,
        )
        rt.progress_every_n_layers = 3
        rt.cancel_flag = _cancel
        st = rt.stats()
        print(
            f"spine_bytes={st['spine_bytes']/1e9:.2f}GB "
            f"full_model_loaded={st['full_model_loaded']} "
            f"modeled_prefetch={st['use_modeled_prefetch']} horizon={st['prefetch_horizon']}"
        )
        _runtime, _tokenizer, _load_error = rt, tok, None
        return rt, tok
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise


def chat_generate(
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.3,
    on_progress: ProgressCb | None = None,
) -> str:
    clear_cancel()
    rt, tok = get_engine()
    rt.cancel_flag = _cancel
    # DeepSeek-V2-Lite-Chat: без длинного system — меньше prompt → быстрее prefill
    clean = [m for m in messages if m.get("role") != "system" or (m.get("content") or "").strip()]
    if len(clean) >= 2 and clean[0].get("role") == "system" and len(clean[0].get("content") or "") > 120:
        clean = clean[1:]

    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
    else:
        prompt = ""
        for m in clean:
            prompt += f"{m['role']}: {m['content']}\n"
        prompt += "assistant:"

    if on_progress:
        on_progress(
            {
                "phase": "prompt",
                "msg": "DeepSeek chat_template → эмбеддинги",
                "chars": len(prompt),
            }
        )

    input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    eos = getattr(tok, "eos_token_id", None)
    prev_cb = rt.progress_cb
    cancelled = False

    def _cb(ev: dict) -> None:
        if ev.get("phase") == "token_done" and ev.get("token_id") is not None:
            try:
                piece = tok.decode([int(ev["token_id"])], skip_special_tokens=True)
                ev = {**ev, "token_text": piece}
            except Exception:
                pass
        if on_progress:
            on_progress(ev)

    rt.progress_cb = _cb if on_progress else None
    rt.use_modeled_prefetch = True
    rt.prefetch_horizon = 2
    try:
        gen = rt.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos,
        )
    except RuntimeError as e:
        if "GENERATION_CANCELLED" in str(e):
            cancelled = True
            gen = __import__("torch").zeros((1, 0), dtype=__import__("torch").long)
            if on_progress:
                on_progress({"phase": "cancelled", "msg": "генерация прервана"})
        else:
            raise
    finally:
        rt.progress_cb = prev_cb

    if rt.experts is not None and not cancelled:
        dropped = rt.experts.evict_below_mean()
        dropped += rt.experts.trim_to_bytes(1_500_000_000)
        rt._drop_pack_dev_missing()
        st = rt.stats()
        if on_progress:
            on_progress(
                {
                    "phase": "sleep",
                    "msg": "sleep: cold-эксперты выгружены",
                    "evicted": len(dropped),
                    "n_hot": st["n_hot"],
                    "resident_mb": round(st["resident_bytes"] / 1e6, 1),
                    "modeled_hit": st.get("n_modeled_hit", 0),
                }
            )
        print(
            f"[sparse] after_gen n_hot={st['n_hot']} "
            f"resident={st['resident_bytes']/1e6:.1f}MB "
            f"loads={st['n_loads']} hits={st['n_hits']} misses={st['n_misses']} "
            f"modeled_hit={st.get('n_modeled_hit')} evicted={len(dropped)}"
        )
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True).strip() if gen.numel() else ""
    if on_progress:
        on_progress(
            {
                "phase": "done" if not cancelled else "cancelled",
                "msg": "ответ готов" if not cancelled else "прервано",
                "reply_chars": len(text),
                "cancelled": cancelled,
            }
        )
    return text


def ask(question: str, *, system: str | None = None, max_new_tokens: int = 64) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return chat_generate(messages, max_new_tokens=max_new_tokens, temperature=0.2)
