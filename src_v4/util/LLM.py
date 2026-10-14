# -*- coding: utf-8 -*-
"""统一 LLM 调用（从 src/util/LLM.py 复刻）。

- 全走云端：openai SDK + OpenAI 兼容端点（DeepSeek/GPT/Qwen/云端 llama 切换只改 config.LLM）。
- 多 agent：按角色取 config.LLM.roles[role] 合并 default。
- 职责：统一 chat / chat_json（结构化输出）、重试与限流、缓存、调用计数与明细日志。
- json_mode 只缓存可解析 JSON（截断/坏 JSON 不入缓存，解析重试真实调 API）。
"""
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, Optional

DEFAULT_ROLE = "default"

_DISK_CACHE_LOCK = threading.Lock()


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 输出中提取 JSON 对象（容忍代码围栏/前后缀文本/截断）。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"```\s*$", "", t).strip()
    start = t.find("{")
    if start < 0:
        return None
    end = t.rfind("}")
    raw = t[start:end + 1] if end >= start else t[start:]
    for candidate in (raw, _repair_json_lax(raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    for repaired in (_repair_truncated_json(raw),
                     _repair_truncated_json(_repair_json_lax(raw))):
        if repaired is not None:
            return repaired
    return None


def _repair_truncated_json(raw: str) -> Optional[Dict]:
    for tail in ('"}]}', '}]}', '"]}', ']}', '"}', "}"):
        try:
            return json.loads(raw + tail)
        except json.JSONDecodeError:
            continue
    return None


def _repair_json_lax(s: str) -> str:
    """单遍宽松修复：字符串值内裸换行→\\n、未转义英文双引号→\\\"。"""
    out = []
    i, n = 0, len(s)
    in_str = False
    while i < n:
        ch = s[i]
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(s[i + 1])
                i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                j = i + 1
                while j < n and s[j] in " \t\r\n":
                    j += 1
                if j >= n or s[j] in ",}]:":
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue
        if in_str and ch in "\r\n":
            if ch == "\r" and i + 1 < n and s[i + 1] == "\n":
                i += 1
            out.append("\\n")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class LLM:
    """云端 LLM 统一封装（openai 兼容端点）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, role: str = DEFAULT_ROLE) -> None:
        self.config = config or {}
        self.role = role
        self._client: Any = None
        self._semaphore = threading.Semaphore(int(self._call_cfg().get("max_concurrent", 1)))
        self._lock = threading.Lock()
        self._call_count: int = 0
        self._tokens_prompt: int = 0          # resp.usage.prompt_tokens (real API calls only)
        self._tokens_completion: int = 0      # resp.usage.completion_tokens
        self._cache: Dict[str, str] = {}
        self._cache_dirty: bool = False
        self._load_disk_cache()

    # ---------- 配置 ----------
    def _call_cfg(self) -> Dict[str, Any]:
        return self.config.get("LLM_CALL", {})

    def _resolve(self, role: Optional[str] = None) -> Dict[str, Any]:
        llm_cfg = self.config.get("LLM", {})
        merged = dict(llm_cfg.get("default", {}))
        role = role or self.role or DEFAULT_ROLE
        if role != DEFAULT_ROLE:
            merged.update(llm_cfg.get("roles", {}).get(role, {}))
        providers = llm_cfg.get("providers", {}) or {}
        active = (llm_cfg.get("active_provider") or "").strip()
        if active and active in providers:
            merged.update(providers[active])
        return merged

    def _build_client(self, cfg: Dict[str, Any]):
        provider = cfg.get("provider", "openai")
        if provider != "openai":
            raise NotImplementedError(f"暂只支持 provider=openai，收到: {provider}")
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get(cfg.get("api_key_env") or "", "")
            if not api_key:
                raise ValueError(f"环境变量 {cfg.get('api_key_env')} 未设置（LLM 全走云端，密钥从环境变量读取）")
            self._client = OpenAI(base_url=cfg.get("base_url") or None, api_key=api_key)
        return self._client

    # ---------- 缓存 ----------
    def _cache_key(self, cfg: Dict[str, Any], prompt: str, system_prompt: Optional[str],
                   json_mode: bool = False) -> str:
        raw = json.dumps([cfg.get("model"), cfg.get("temperature"), json_mode, prompt, system_prompt],
                         ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_enabled(self, cfg: Dict[str, Any]) -> bool:
        return bool(self._call_cfg().get("cache_enabled", False))

    def _load_disk_cache(self) -> None:
        cache_dir = self._call_cfg().get("cache_dir") or ""
        if not cache_dir:
            return
        path = os.path.join(cache_dir, "llm_cache.json")
        with _DISK_CACHE_LOCK:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                except (json.JSONDecodeError, OSError):
                    self._cache = {}

    def _save_disk_cache(self) -> None:
        cache_dir = self._call_cfg().get("cache_dir") or ""
        if not cache_dir:
            return
        with _DISK_CACHE_LOCK:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "llm_cache.json"), "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)

    # ---------- 计数与日志 ----------
    def _count(self) -> None:
        with self._lock:
            self._call_count += 1

    def _count_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._tokens_prompt += int(prompt_tokens or 0)
            self._tokens_completion += int(completion_tokens or 0)

    def _log_call(self, cfg: Dict[str, Any], prompt: str, text: str, attempt: int,
                  role: Optional[str] = None, prompt_tokens: int = 0,
                  completion_tokens: int = 0) -> None:
        cc = self._call_cfg()
        if cc.get("log_calls", True):
            try:
                from util.log import get
                get("util.LLM").info(
                    "LLM role=%s model=%s attempt=%d prompt_len=%d resp_len=%d",
                    role or self.role or DEFAULT_ROLE, cfg.get("model"),
                    attempt + 1, len(prompt), len(text))
            except Exception:
                pass
        # Optional full prompt/response logging for per-fold audit trails.
        pdir = cc.get("prompt_log_dir") or ""
        if pdir:
            try:
                os.makedirs(pdir, exist_ok=True)
                rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "role": role or self.role or DEFAULT_ROLE,
                       "model": cfg.get("model"),
                       "attempt": attempt + 1,
                       "prompt_tokens": int(prompt_tokens or 0),
                       "completion_tokens": int(completion_tokens or 0),
                       "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
                       "prompt": prompt, "response": text}
                with open(os.path.join(pdir, "prompts.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # ---------- 主接口 ----------
    def chat(self, prompt: str, system_prompt: Optional[str] = None,
             role: Optional[str] = None, json_mode: bool = False) -> str:
        """统一调用入口：重试/限流/并发限制，返回文本；空响应按失败重试（最多 max_retries 次）。"""
        cfg = self._resolve(role)
        cache_key = self._cache_key(cfg, prompt, system_prompt, json_mode)
        if self._cache_enabled(cfg) and cache_key in self._cache:
            return self._cache[cache_key]

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        max_retries = int(self._call_cfg().get("max_retries", 3))
        timeout = float(self._call_cfg().get("timeout_seconds", 60))
        interval = float(self._call_cfg().get("request_interval", 0.0))
        with self._semaphore:
            for attempt in range(max_retries + 1):
                try:
                    client = self._build_client(cfg)
                    extra_body: Dict[str, Any] = {}
                    if cfg.get("thinking") is not None:
                        extra_body["thinking"] = cfg["thinking"]
                    if cfg.get("reasoning_effort") is not None:
                        extra_body["reasoning_effort"] = cfg["reasoning_effort"]
                    resp = client.chat.completions.create(
                        model=cfg.get("model") or "",
                        messages=messages,
                        temperature=cfg.get("temperature", 0.0),
                        max_tokens=cfg.get("max_tokens", 2048),
                        timeout=timeout,
                        **({"response_format": {"type": "json_object"}} if json_mode else {}),
                        **({"extra_body": extra_body} if extra_body else {}),
                    )
                    text = resp.choices[0].message.content or ""
                    usage = getattr(resp, "usage", None) or {}
                    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                    ct = int(getattr(usage, "completion_tokens", 0) or 0)
                    self._count()
                    if pt or ct:
                        self._count_tokens(pt, ct)
                    self._log_call(cfg, prompt, text, attempt, role, pt, ct)
                    if text:
                        if self._cache_enabled(cfg):
                            if not json_mode or _extract_json(text) is not None:
                                self._cache[cache_key] = text
                            self._save_disk_cache()
                        return text
                    if attempt >= max_retries:
                        return text
                    time.sleep(interval or min(2 ** attempt, 10))
                except Exception as exc:
                    if attempt >= max_retries:
                        raise
                    delay = interval or min(2 ** attempt, 10)
                    if getattr(exc, "status_code", None) == 429:
                        delay = max(delay, 20.0)
                    time.sleep(delay)

    def chat_json(self, prompt: str, system_prompt: Optional[str] = None,
                  role: Optional[str] = None, max_parse_tries: int = 4) -> Dict:
        """结构化 JSON 输出：解析失败自动重试；空响应由 chat 内部重试，最终仍空则如实抛出。"""
        last = None
        for attempt in range(max_parse_tries):
            # 重试时附加格式修正提示（v4.25 修：温度 0 下重复同一 prompt 基本注定同样坏输出，
            # 纯浪费 API 调用；提示只约束格式、不引导语义，不违反 prompt 中立原则）
            p = prompt if attempt == 0 else (
                prompt + "\n（注意：上次输出不是合法 JSON。请只输出符合格式要求的 JSON，不要包含任何多余文本。）")
            text = self.chat(p, system_prompt=system_prompt, role=role, json_mode=True)
            if not text:
                break
            last = text
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
            if attempt < max_parse_tries - 1:
                time.sleep(1.5)
        raise ValueError(f"LLM 输出无法解析为 JSON，最后输出: {str(last)[:200]}")

    @property
    def call_count(self) -> int:
        """已发生 LLM 调用次数（成本指标）。"""
        return self._call_count

    @property
    def token_counts(self) -> Dict[str, int]:
        """已消耗 token 数（resp.usage 累计；缓存命中/失败重试不计入）。"""
        return {
            "prompt_tokens": self._tokens_prompt,
            "completion_tokens": self._tokens_completion,
            "total_tokens": self._tokens_prompt + self._tokens_completion,
        }

    def reset_cache(self) -> None:
        self._cache.clear()

