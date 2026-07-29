"""
Sparse MoE runtime: L1 spine resident; L2 experts only via DynamicExpertStore.

Forbidden: AutoModelForCausalLM.from_pretrained of the full MoE.
Allowed: safetensors by tensor name + get_expert(layer, eid) along the gate/orbit.

Full source — edit freely. Math: docs/MATH.md. Design: docs/DESIGN_DYNAMIC_WEIGHTS.md.
License: Apache-2.0 — LICENSE, NOTICE, ATTRIBUTION.md
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open

from .dynamic_expert_store import DynamicExpertStore
from .orbit_predictor import OrbitPredictor

MID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
GIGACHAT_MID = "ai-sage/GigaChat3-10B-A1.8B-bf16"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(min_v, max_v, dim):
    if min_v == max_v:
        max_v += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(linear_func, 0, 1)


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class YarnRoPE:
    """YaRN RoPE как в modeling_deepseek (без загрузки всей модели)."""

    def __init__(self, cfg: dict, device: torch.device):
        self.dim = int(cfg["qk_rope_head_dim"])
        self.base = float(cfg["rope_theta"])
        rs = cfg["rope_scaling"]
        self.scaling_factor = float(rs["factor"])
        self.original_max_position_embeddings = int(rs["original_max_position_embeddings"])
        self.beta_fast = float(rs["beta_fast"])
        self.beta_slow = float(rs["beta_slow"])
        self.mscale = float(rs["mscale"])
        self.mscale_all_dim = float(rs["mscale_all_dim"])
        self.device = device
        self.max_seq_len_cached = 0
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _set_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        dim = self.dim
        device = self.device
        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        _mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.mscale)
            / _yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        self.cos_cached = (emb.cos() * _mscale).to(dtype)
        self.sin_cached = (emb.sin() * _mscale).to(dtype)

    def __call__(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached or self.cos_cached is None:
            self._set_cache(max(seq_len, 32), dtype)
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return (weight.float() * x32).to(input_dtype)


def _swiglu(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    g = F.linear(x, gate_w)
    u = F.linear(x, up_w)
    return F.linear(F.silu(g) * u, down_w)


@dataclass
class SparseDeepseekRuntime:
    """Генерация без полного from_pretrained."""

    model_id: str = MID
    device: str = "cpu"
    dtype: torch.dtype = torch.float16
    cfg: dict = field(default_factory=dict)
    spine: dict[str, torch.Tensor] = field(default_factory=dict)
    experts: DynamicExpertStore | None = None
    rope: YarnRoPE | None = None
    spine_bytes: int = 0
    _tok: Any = None
    # последний сигнал «мышления» из residual/орбиты (для UI)
    last_orbit: list[int] = field(default_factory=list)
    last_modeled_orbit: list[int] = field(default_factory=list)
    last_h_norm: float = 0.0
    last_layer: int = 0
    last_h_vec: Any = None  # np residual для token-horizon
    progress_cb: Any = None  # Callable[[dict], None] | None
    cancel_flag: Any = None  # threading.Event | None — прерывание generate
    # ускорение / modeled prefetch
    use_modeled_prefetch: bool = True
    prefetch_horizon: int = 2  # слои вперёд (alias layer_horizon)
    token_horizon: int = 4  # токены t…t+H
    keep_hot_during_generate: bool = True
    progress_every_n_layers: int = 1
    predictors: dict[int, OrbitPredictor] = field(default_factory=dict)
    _pack_dev: OrderedDict = field(default_factory=OrderedDict)  # LRU (layer,eid)→pack
    _cur_tok_id: int | None = None
    _state_lock: Any = field(default_factory=threading.Lock)
    _prefetch_lock: Any = field(default_factory=threading.Lock)
    _prefetch_q: Any = field(default=None, repr=False)
    _prefetch_worker_started: bool = False
    _prefetch_epoch: int = 0
    _prefetch_paused: bool = False
    pack_dev_cap: int = 512  # ≥~2 токена × слои × top_k, иначе LRU → 0 hits
    n_modeled_prefetch: int = 0
    n_prefetch_failures: int = 0
    last_prefetch_error: str = ""
    fail_fast_prefetch: bool = False  # debug: re-raise worker errors
    n_modeled_hit: int = 0  # live gate ∩ modeled перед загрузкой
    n_gate_pack_hit: int = 0  # live expert уже в pack_dev
    n_gate_pack_miss: int = 0  # пришлось грузить sync на gate
    gate_miss_seconds: float = 0.0
    _in_generate: bool = False
    # захват h на входе MoE (post-attn LN) — для residual-стендов
    capture_moe_h: bool = False
    captured_moe_h: dict = field(default_factory=dict)  # layer -> np.ndarray [T, H]

    def _ensure_prefetch_worker(self) -> None:
        if self._prefetch_worker_started:
            return
        self._prefetch_q = queue.Queue(maxsize=128)
        self._prefetch_worker_started = True

        def _worker() -> None:
            while True:
                try:
                    item = self._prefetch_q.get()
                except Exception:
                    return
                if item is None:
                    return
                L, eid, epoch = item
                try:
                    if self._cancelled():
                        continue
                    if self._prefetch_paused or epoch != self._prefetch_epoch:
                        continue
                    self._expert_pack_dev(int(L), int(eid))
                    self.n_modeled_prefetch += 1
                except Exception as exc:
                    self.n_prefetch_failures += 1
                    self.last_prefetch_error = f"{type(exc).__name__}: {exc}"
                    self._emit(
                        {
                            "phase": "prefetch_error",
                            "layer": int(L),
                            "expert": int(eid),
                            "error": self.last_prefetch_error,
                        }
                    )
                    if self.fail_fast_prefetch:
                        raise
                finally:
                    try:
                        self._prefetch_q.task_done()
                    except Exception:
                        pass

        threading.Thread(target=_worker, daemon=True, name="expert-prefetch").start()

    def _drain_prefetch_q(self) -> None:
        if self._prefetch_q is None:
            return
        try:
            while True:
                self._prefetch_q.get_nowait()
                try:
                    self._prefetch_q.task_done()
                except Exception:
                    pass
        except queue.Empty:
            pass

    def _trim_pack_dev(self) -> None:
        """LRU cap — не раздувать RAM во время generate."""
        cap = max(16, int(self.pack_dev_cap))
        with self._prefetch_lock:
            while len(self._pack_dev) > cap:
                self._pack_dev.popitem(last=False)

    def _set_prefetch_state(
        self,
        *,
        tok_id: int | None = None,
        h_np: np.ndarray | None = None,
    ) -> None:
        """Atomic update of shared modeled-prefetch state."""
        with self._state_lock:
            if tok_id is not None:
                self._cur_tok_id = int(tok_id)
            if h_np is not None:
                self.last_h_vec = h_np

    def _get_prefetch_state(self) -> tuple[np.ndarray | None, int | None]:
        """Atomic snapshot of shared modeled-prefetch state."""
        with self._state_lock:
            return self.last_h_vec, self._cur_tok_id

    def request_cancel(self) -> None:
        if self.cancel_flag is not None:
            self.cancel_flag.set()
        # stop accepting / executing stale prefetch work
        self._prefetch_paused = True
        self._prefetch_epoch += 1
        self._drain_prefetch_q()

    def _cancelled(self) -> bool:
        return bool(self.cancel_flag is not None and self.cancel_flag.is_set())

    def _check_cancel(self, where: str = "") -> None:
        if self._cancelled():
            raise RuntimeError(f"GENERATION_CANCELLED{(':' + where) if where else ''}")

    @classmethod
    def load(
        cls,
        *,
        model_id: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        use_modeled_prefetch: bool = True,
        prefetch_horizon: int = 2,
        token_horizon: int = 4,
    ) -> "SparseDeepseekRuntime":
        mid = model_id or MID
        idx_path = Path(try_to_load_from_cache(mid, "model.safetensors.index.json"))
        if not idx_path or not Path(idx_path).exists():
            raise FileNotFoundError(f"нет index для {mid}")
        cfg = json.loads(Path(try_to_load_from_cache(mid, "config.json")).read_text(encoding="utf-8"))
        store = DynamicExpertStore.from_index_file(mid, idx_path, allow_hub_download=False)
        weight_map = store.weight_map

        # spine = всё кроме routed experts (и их scale_inv)
        spine_names = [
            n
            for n in weight_map
            if ".mlp.experts." not in n
        ]
        by_shard: dict[str, list[str]] = {}
        for n in spine_names:
            by_shard.setdefault(weight_map[n], []).append(n)

        spine: dict[str, torch.Tensor] = {}
        nbytes = 0
        for shard, names in by_shard.items():
            path = store.resolve_shard_path(shard)
            if path is None:
                raise FileNotFoundError(f"spine shard missing: {shard}")
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for name in names:
                    t = f.get_tensor(name)
                    # bias и т.п. оставляем native; веса — в dtype
                    if t.ndim >= 2:
                        t = t.to(dtype=dtype)
                    spine[name] = t
                    nbytes += int(t.numel() * t.element_size())

        n_routed = int(cfg.get("n_routed_experts", 64))
        top_k = int(cfg.get("num_experts_per_tok", 6))
        n_layers = int(cfg["num_hidden_layers"])
        first_dense = int(cfg.get("first_k_dense_replace", 1))
        predictors = {
            L: OrbitPredictor(n_experts=n_routed, top_k=top_k)
            for L in range(first_dense, n_layers)
        }

        rt = cls(
            model_id=mid,
            cfg=cfg,
            spine=spine,
            experts=store,
            device=device,
            dtype=dtype,
            spine_bytes=nbytes,
            use_modeled_prefetch=use_modeled_prefetch,
            prefetch_horizon=prefetch_horizon,
            token_horizon=token_horizon,
            predictors=predictors,
        )
        rt.rope = YarnRoPE(cfg, torch.device(device))
        return rt

    def w(self, name: str) -> torch.Tensor:
        t = self.spine[name]
        if t.device.type != self.device:
            t = t.to(self.device)
            self.spine[name] = t
        return t

    def stats(self) -> dict[str, Any]:
        assert self.experts is not None
        st = self.experts.stats()
        st["spine_bytes"] = self.spine_bytes
        st["mode"] = "sparse_spine_plus_orbit_experts"
        st["full_model_loaded"] = False
        st["model_id"] = self.model_id
        st["use_modeled_prefetch"] = self.use_modeled_prefetch
        st["prefetch_horizon"] = self.prefetch_horizon
        st["token_horizon"] = self.token_horizon
        st["n_modeled_prefetch"] = self.n_modeled_prefetch
        st["n_prefetch_failures"] = self.n_prefetch_failures
        st["last_prefetch_error"] = self.last_prefetch_error
        st["n_modeled_hit"] = self.n_modeled_hit
        st["n_gate_pack_hit"] = self.n_gate_pack_hit
        st["n_gate_pack_miss"] = self.n_gate_pack_miss
        st["gate_miss_seconds"] = round(self.gate_miss_seconds, 4)
        st["n_pack_dev"] = len(self._pack_dev)
        return st

    def clear_expert_residency(self) -> None:
        """Сброс hot/pack для честного cold-сравнения (предикторы не трогаем)."""
        self._prefetch_paused = True
        self._prefetch_epoch += 1
        self._drain_prefetch_q()
        if self.experts is not None:
            self.experts.drop_all_hot()
        with self._prefetch_lock:
            self._pack_dev.clear()
        self.n_gate_pack_hit = 0
        self.n_gate_pack_miss = 0
        self.gate_miss_seconds = 0.0
        self.n_modeled_prefetch = 0
        self.n_prefetch_failures = 0
        self.last_prefetch_error = ""
        self.n_modeled_hit = 0
        self.captured_moe_h = {}
        self._prefetch_paused = False

    def install_predictors(self, factory) -> None:
        """
        Поставить независимые предикторы на каждый MoE-слой (без моста между слоями).
        factory: () -> predictor с API OrbitPredictor.
        """
        n_layers = int(self.cfg["num_hidden_layers"])
        first_dense = int(self.cfg.get("first_k_dense_replace", 1))
        self.predictors = {L: factory() for L in range(first_dense, n_layers)}

    def encode_moe_h(
        self, input_ids: torch.Tensor, *, layer: int = 1
    ) -> np.ndarray:
        """
        Дойти до post-attn LN выбранного MoE-слоя и вернуть h [T, H] БЕЗ загрузки экспертов.
        Нужно для residual-стенда: тот же сигнал, что видит gate, без miss-wait I/O.
        """
        cfg = self.cfg
        first_dense = int(cfg.get("first_k_dense_replace", 1))
        eps = float(cfg["rms_norm_eps"])
        device = torch.device(self.device)
        target = int(layer)
        if target < first_dense:
            raise ValueError(f"layer {target} is dense, not MoE (first_dense={first_dense})")

        ids = input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        bsz, q_len = ids.shape
        self._set_prefetch_state(tok_id=int(ids[0, -1].item()))
        embed = self.w("model.embed_tokens.weight")
        hidden = F.embedding(ids, embed)
        position_ids = torch.arange(0, q_len, device=device, dtype=torch.long).unsqueeze(0)
        past_kvs: list = [None] * (target + 1)

        for L in range(target + 1):
            residual = hidden
            h = _rms_norm(hidden, self.w(f"model.layers.{L}.input_layernorm.weight"), eps)
            attn_out, _kv = self._attn(L, h, position_ids, past_kvs[L])
            hidden = residual + attn_out
            residual = hidden
            h = _rms_norm(
                hidden, self.w(f"model.layers.{L}.post_attention_layernorm.weight"), eps
            )
            if L == target:
                out = h[0].detach().float().cpu().numpy().copy()
                self.captured_moe_h[target] = out
                return out
            if L >= first_dense:
                # промежуточные MoE до target — без них residual выше неточен;
                # для L < target и L MoE всё же нужен forward экспертов.
                hidden = residual + self._moe(L, h)
            else:
                hidden = residual + self._dense_mlp(L, h)
        raise RuntimeError("encode_moe_h: unreachable")
    def _emit(self, event: dict[str, Any]) -> None:
        cb = self.progress_cb
        if cb is not None:
            try:
                cb(event)
            except Exception:
                pass

    def _expert_pack_dev(self, layer: int, eid: int, *, count_gate: bool = False) -> dict[str, torch.Tensor]:
        """Кэш экспертов уже в device/dtype — без повторного .to() на каждый hit."""
        assert self.experts is not None
        key = (layer, eid)
        with self._prefetch_lock:
            if key in self._pack_dev:
                self._pack_dev.move_to_end(key)
                if count_gate:
                    self.n_gate_pack_hit += 1
                return self._pack_dev[key]
        t0 = time.perf_counter() if count_gate else None
        pack = self.experts.get_expert(layer, eid, device="cpu")
        out = {}
        for k, v in pack.items():
            if "scale" in k:
                continue  # fp8 aux — bf16/fp16 pack без scale
            out[k] = v.to(device=self.device, dtype=self.dtype, non_blocking=False)
        with self._prefetch_lock:
            if key not in self._pack_dev:
                self._pack_dev[key] = out
                self._pack_dev.move_to_end(key)
            else:
                self._pack_dev.move_to_end(key)
                out = self._pack_dev[key]
            if count_gate and t0 is not None:
                self.n_gate_pack_miss += 1
                self.gate_miss_seconds += time.perf_counter() - t0
        self._trim_pack_dev()
        return out

    def _drop_pack_dev_missing(self) -> None:
        assert self.experts is not None
        hot = set(self.experts.hot.keys())
        with self._prefetch_lock:
            for key in list(self._pack_dev.keys()):
                if key not in hot:
                    self._pack_dev.pop(key, None)

    def _prefetch_jobs_async(self, jobs: list[tuple[int, int]]) -> None:
        """Одна очередь / один worker — без шторма потоков."""
        if not jobs or self._prefetch_paused:
            return
        self._ensure_prefetch_worker()
        assert self._prefetch_q is not None
        with self._prefetch_lock:
            packed = set(self._pack_dev.keys())
        seen: set[tuple[int, int]] = set()
        epoch = self._prefetch_epoch
        for j in jobs:
            if j in seen or j in packed:
                continue
            seen.add(j)
            try:
                self._prefetch_q.put_nowait((j[0], j[1], epoch))
            except queue.Full:
                break

    def _wait_prefetch_idle(self, timeout_s: float = 120.0) -> None:
        """Дождаться очереди prefetch (после prefill — до decode)."""
        if self._prefetch_q is None:
            return
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if self._prefetch_q.unfinished_tasks == 0:
                return
            time.sleep(0.01)

    def prefetch_token_horizon(self, h_vec: torch.Tensor | None = None, tok_id: int | None = None) -> int:
        """
        h → орбиты на t…t+H. Async через одну очередь.
        Приоритет: ранние MoE-слои × ближайшие шаги горизонта.
        """
        if not self.use_modeled_prefetch or not self.predictors:
            return 0
        if h_vec is None:
            h_np, snap_tid = self._get_prefetch_state()
            if h_np is None:
                return 0
        else:
            h_np = h_vec.detach().float().cpu().numpy()
            self._set_prefetch_state(h_np=h_np)
            _, snap_tid = self._get_prefetch_state()
        tid = snap_tid if tok_id is None else tok_id
        H = max(1, int(self.token_horizon))
        jobs: list[tuple[int, int]] = []
        # ближайшие слои важнее (очередь FIFO — кладём их первыми)
        for layer in sorted(self.predictors.keys()):
            orbits = self.predictors[layer].predict_token_horizon(h_np, tid, H)
            for orb in orbits[: max(1, min(H, 2))]:  # не дальше 2 шагов в sync-приоритете очереди
                for e in orb:
                    jobs.append((layer, int(e)))
        self._prefetch_jobs_async(jobs)
        self._emit(
            {
                "phase": "token_prefetch",
                "msg": f"token-horizon H={H}: queued {len(jobs)}",
                "token_horizon": H,
                "n_jobs": len(jobs),
            }
        )
        return len(jobs)

    def _modeled_prefetch(self, layer: int, h_vec: torch.Tensor) -> list[int]:
        """Только оценка орбиты для метрик; I/O — sticky live + token_horizon."""
        if not self.use_modeled_prefetch or layer not in self.predictors:
            return []
        h_np = h_vec.detach().float().cpu().numpy()
        self._set_prefetch_state(h_np=h_np)
        _, snap_tid = self._get_prefetch_state()
        pred = self.predictors[layer].prefetch_set(
            h_np, snap_tid, horizon_copies=1
        )
        self.last_modeled_orbit = list(pred)
        return pred

    def _sticky_prefetch_live(self, layer: int, live: list[int]) -> None:
        """Live орбита → очередь тех же eid на L+1…L+horizon (высокий приоритет)."""
        if not self.use_modeled_prefetch or not live:
            return
        n_layers = int(self.cfg["num_hidden_layers"])
        jobs: list[tuple[int, int]] = []
        for dl in range(1, max(1, self.prefetch_horizon) + 1):
            L2 = layer + dl
            if L2 >= n_layers or L2 not in self.predictors:
                break
            for e in live:
                jobs.append((L2, int(e)))
        self._prefetch_jobs_async(jobs)

    def _moe_gate(self, layer: int, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [T, H] → topk_idx [T,K], topk_weight [T,K]."""
        assert self.experts is not None
        if self.capture_moe_h:
            self.captured_moe_h[int(layer)] = h.detach().float().cpu().numpy().copy()
        gate = self.w(f"model.layers.{layer}.mlp.gate.weight")
        self.last_h_norm = float(h.float().norm(dim=-1).mean().item())
        href = h[-1] if h.shape[0] == 1 else h.mean(dim=0)
        modeled = self._modeled_prefetch(layer, href)

        logits = F.linear(h.float(), gate.float())
        scoring = self.cfg.get("scoring_func", "softmax")
        if scoring == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1)
        bias_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        if bias_name in self.spine:
            scores = scores + self.w(bias_name).float().view(1, -1)

        top_k = int(self.cfg["num_experts_per_tok"])
        topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
        if top_k > 1 and self.cfg.get("norm_topk_prob"):
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * float(self.cfg.get("routed_scaling_factor", 1.0))
        self.last_orbit = [int(x) for x in topk_idx[-1].tolist()]
        self.last_layer = layer
        if modeled:
            inter = set(modeled) & set(self.last_orbit)
            self.n_modeled_hit += len(inter)
        if layer in self.predictors:
            _, snap_tid = self._get_prefetch_state()
            self.predictors[layer].deposit(
                href.detach().float().cpu().numpy(),
                self.last_orbit,
                snap_tid,
            )
        return topk_idx, topk_weight.to(h.dtype)

    def _shared_ffn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp.shared_experts"
        return _swiglu(
            x,
            self.w(f"{p}.gate_proj.weight"),
            self.w(f"{p}.up_proj.weight"),
            self.w(f"{p}.down_proj.weight"),
        )

    def _dense_mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{layer}.mlp"
        return _swiglu(
            x,
            self.w(f"{p}.gate_proj.weight"),
            self.w(f"{p}.up_proj.weight"),
            self.w(f"{p}.down_proj.weight"),
        )

    def _moe(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        assert self.experts is not None
        bsz, seq, hdim = hidden.shape
        flat = hidden.reshape(-1, hdim)
        topk_idx, topk_weight = self._moe_gate(layer, flat)
        y = torch.zeros_like(flat)
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for t in range(flat.shape[0]):
            for k in range(topk_idx.shape[1]):
                buckets[int(topk_idx[t, k].item())].append((t, k))
        eids = list(buckets.keys())
        packs: dict[int, dict[str, torch.Tensor]] = {}
        # 1) sync load текущего слоя
        for eid in eids:
            packs[eid] = self._expert_pack_dev(layer, eid, count_gate=True)
        # 2) sticky L+1 в очередь — 3) FFN текущего слоя пересекается с I/O
        self._sticky_prefetch_live(layer, list(eids))
        for eid, pairs in buckets.items():
            pack = packs[eid]
            tok_ix = torch.tensor([t for t, _ in pairs], device=flat.device, dtype=torch.long)
            ks = torch.tensor([k for _, k in pairs], device=flat.device, dtype=torch.long)
            x = flat.index_select(0, tok_ix)
            out = _swiglu(x, pack["gate_proj.weight"], pack["up_proj.weight"], pack["down_proj.weight"])
            w = topk_weight[tok_ix, ks].to(out.dtype).unsqueeze(-1)
            y.index_add_(0, tok_ix, out * w)
        y = y.view(bsz, seq, hdim) + self._shared_ffn(layer, hidden)
        return y

    def _attn(
        self,
        layer: int,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.rope is not None
        cfg = self.cfg
        bsz, q_len, _ = hidden.shape
        n_heads = int(cfg["num_attention_heads"])
        q_head_dim = int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"])
        qk_nope = int(cfg["qk_nope_head_dim"])
        qk_rope = int(cfg["qk_rope_head_dim"])
        v_head = int(cfg["v_head_dim"])
        kv_lora = int(cfg["kv_lora_rank"])

        q = F.linear(hidden, self.w(f"model.layers.{layer}.self_attn.q_proj.weight"))
        q = q.view(bsz, q_len, n_heads, q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [qk_nope, qk_rope], dim=-1)

        compressed = F.linear(
            hidden, self.w(f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight")
        )
        compressed_kv, k_pe = torch.split(compressed, [kv_lora, qk_rope], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, qk_rope).transpose(1, 2)
        kv = F.linear(
            _rms_norm(
                compressed_kv,
                self.w(f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"),
                float(cfg["rms_norm_eps"]),
            ),
            self.w(f"model.layers.{layer}.self_attn.kv_b_proj.weight"),
        )
        kv = kv.view(bsz, q_len, n_heads, qk_nope + v_head).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [qk_nope, v_head], dim=-1)

        kv_seq_len = value_states.shape[-2]
        if past_kv is not None:
            kv_seq_len += past_kv[0].shape[-2]

        cos, sin = self.rope(kv_seq_len, hidden.dtype)
        q_pe, k_pe = _apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, n_heads, q_len, q_head_dim)
        query_states[:, :, :, :qk_nope] = q_nope
        query_states[:, :, :, qk_nope:] = q_pe

        key_states = k_pe.new_empty(bsz, n_heads, q_len, q_head_dim)
        key_states[:, :, :, :qk_nope] = k_nope
        key_states[:, :, :, qk_nope:] = k_pe

        if past_kv is not None:
            key_states = torch.cat([past_kv[0], key_states], dim=2)
            value_states = torch.cat([past_kv[1], value_states], dim=2)

        softmax_scale = q_head_dim ** (-0.5)
        if cfg.get("rope_scaling"):
            mscale_all_dim = cfg["rope_scaling"].get("mscale_all_dim", 0)
            if mscale_all_dim:
                mscale = _yarn_get_mscale(cfg["rope_scaling"]["factor"], mscale_all_dim)
                softmax_scale = softmax_scale * mscale * mscale

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * softmax_scale
        # causal mask
        kv_len = key_states.shape[-2]
        mask = torch.full(
            (q_len, kv_len),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
            dtype=attn_weights.dtype,
        )
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
        attn_weights = attn_weights + mask[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_out = torch.matmul(attn_weights, value_states)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, n_heads * v_head)
        attn_out = F.linear(attn_out, self.w(f"model.layers.{layer}.self_attn.o_proj.weight"))
        return attn_out, (key_states, value_states)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        cfg = self.cfg
        n_layers = int(cfg["num_hidden_layers"])
        first_dense = int(cfg.get("first_k_dense_replace", 1))
        eps = float(cfg["rms_norm_eps"])
        device = torch.device(self.device)

        if past_kvs is None:
            past_kvs = [None] * n_layers
            past_len = 0
        else:
            past_len = 0 if past_kvs[0] is None else past_kvs[0][0].shape[-2]

        ids = input_ids.to(device)
        bsz, q_len = ids.shape
        # токен для induction-предиктора (последний входного окна)
        self._set_prefetch_state(tok_id=int(ids[0, -1].item()))
        embed = self.w("model.embed_tokens.weight")
        hidden = F.embedding(ids, embed)

        position_ids = torch.arange(
            past_len, past_len + q_len, device=device, dtype=torch.long
        ).unsqueeze(0)

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = []
        # норма входа (эмбеддинг) — стартовый пульс
        self.last_h_norm = float(hidden.float().norm(dim=-1).mean().item())
        self._emit(
            {
                "phase": "embed",
                "msg": "эмбеддинг токенов",
                "h_norm": round(self.last_h_norm, 3),
                "q_len": int(q_len),
            }
        )
        for layer in range(n_layers):
            self._check_cancel(f"layer_{layer}")
            residual = hidden
            h = _rms_norm(hidden, self.w(f"model.layers.{layer}.input_layernorm.weight"), eps)
            attn_out, kv = self._attn(layer, h, position_ids, past_kvs[layer])
            new_past.append(kv)
            hidden = residual + attn_out
            residual = hidden
            h = _rms_norm(
                hidden, self.w(f"model.layers.{layer}.post_attention_layernorm.weight"), eps
            )
            if layer >= first_dense:
                hidden = residual + self._moe(layer, h)
                st = self.experts.stats() if self.experts is not None else {}
                every = max(1, int(self.progress_every_n_layers))
                if layer % every == 0 or layer == n_layers - 1:
                    self._emit(
                        {
                            "phase": "moe",
                            "msg": f"слой {layer}/{n_layers - 1} · live+modeled орбита",
                            "layer": layer,
                            "n_layers": n_layers,
                            "orbit": list(self.last_orbit),
                            "modeled_orbit": list(self.last_modeled_orbit),
                            "h_norm": round(self.last_h_norm, 3),
                            "n_hot": st.get("n_hot", 0),
                            "resident_mb": round(st.get("resident_bytes", 0) / 1e6, 1),
                            "loads": st.get("n_loads", 0),
                            "hits": st.get("n_hits", 0),
                            "misses": st.get("n_misses", 0),
                            "modeled_hit": self.n_modeled_hit,
                        }
                    )
            else:
                hidden = residual + self._dense_mlp(layer, h)
                self._emit(
                    {
                        "phase": "dense",
                        "msg": f"слой {layer}/{n_layers - 1} · dense MLP",
                        "layer": layer,
                        "n_layers": n_layers,
                        "h_norm": round(float(h.float().norm(dim=-1).mean().item()), 3),
                    }
                )
            # sleep mid-forward только вне generate (иначе убиваем prefetch)
            if (
                not self._in_generate
                and not self.keep_hot_during_generate
                and layer >= first_dense
                and self.experts is not None
                and self.experts.n_hot() > 32
            ):
                self.experts.evict_below_mean()
                self._drop_pack_dev_missing()

        hidden = _rms_norm(hidden, self.w("model.norm.weight"), eps)
        # lm_head часто считают в fp32 для стабильности
        logits = F.linear(hidden.float(), self.w("lm_head.weight").float())
        return logits, new_past

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        import time

        past = None
        generated = input_ids.to(self.device)
        prompt_len = generated.shape[1]
        t_gen0 = time.perf_counter()
        t_prefill_end: float | None = None
        self._in_generate = True
        try:
            for step in range(max_new_tokens):
                self._check_cancel(f"token_{step}")
                self._emit(
                    {
                        "phase": "token",
                        "msg": f"токен {step + 1}/{max_new_tokens}",
                        "step": step + 1,
                        "max_new_tokens": max_new_tokens,
                    }
                )
                step_in = generated if past is None else generated[:, -1:]
                self._set_prefetch_state(tok_id=int(step_in[0, -1].item()))
                t_step0 = time.perf_counter()
                logits, past = self.forward_logits(step_in, past)
                if step == 0:
                    # прогрев t…t+H ДО decode; worker пауза на время decode (не жрёт bandwidth)
                    if self.use_modeled_prefetch:
                        self._prefetch_paused = False
                        _, snap_tid = self._get_prefetch_state()
                        self.prefetch_token_horizon(tok_id=snap_tid)
                        self._wait_prefetch_idle(timeout_s=180.0)
                        self._prefetch_paused = True
                    t_prefill_end = time.perf_counter()
                next_logits = logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = torch.softmax(next_logits / max(temperature, 1e-5), dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                tid = int(next_id.item())
                generated = torch.cat([generated, next_id.to(generated.device)], dim=-1)
                self._set_prefetch_state(tok_id=tid)
                n_out = step + 1
                elapsed = time.perf_counter() - t_gen0
                prefill_s = None if t_prefill_end is None else (t_prefill_end - t_gen0)
                decode_s = None
                decode_tps = None
                if t_prefill_end is not None and n_out >= 1:
                    decode_s = max(1e-9, time.perf_counter() - t_prefill_end)
                    decode_tps = n_out / decode_s
                step_s = time.perf_counter() - t_step0
                self._emit(
                    {
                        "phase": "token_done",
                        "msg": f"token_id={tid} · {n_out} tok · {elapsed:.1f}s",
                        "token_id": tid,
                        "step": n_out,
                        "orbit": list(self.last_orbit),
                        "modeled_orbit": list(self.last_modeled_orbit),
                        "h_norm": round(self.last_h_norm, 3),
                        "layer": self.last_layer,
                        "elapsed_s": round(elapsed, 2),
                        "prefill_s": None if prefill_s is None else round(prefill_s, 2),
                        "decode_s": None if decode_s is None else round(decode_s, 2),
                        "tokens_out": n_out,
                        "tok_per_s": None if decode_tps is None else round(decode_tps, 4),
                        "step_s": round(step_s, 2),
                        "gate_pack_hit": self.n_gate_pack_hit,
                        "gate_pack_miss": self.n_gate_pack_miss,
                        "gate_miss_s": round(self.gate_miss_seconds, 3),
                        "speed_note": "tok/s = tokens_out / decode_wall; miss_s = sync expert load",
                    }
                )
                if eos_token_id is not None and tid == eos_token_id:
                    break
            return generated[:, prompt_len:]
        finally:
            self._in_generate = False
            self._prefetch_paused = False
