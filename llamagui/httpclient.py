"""Minimal dependency-free client for the llama-server HTTP API.

Only what the GUI itself needs: liveness, server properties, and a small
streaming chat call used by the self-test.  ``urllib`` and ``http.client`` are
used directly rather than pulling in ``requests``, which keeps the "zero
dependencies" promise intact.

Errors are returned as a :class:`HttpResult` with ``error`` set instead of
being raised: a server that is still loading, or one that closed the connection
as it replaced its model, are normal states for this GUI rather than bugs.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

#: Health states reported by ``GET /health``.
HEALTH_LOADING = "loading"
HEALTH_READY = "ready"
HEALTH_UNREACHABLE = "unreachable"
HEALTH_ERROR = "error"


@dataclass
class HttpResult:
    """Outcome of one request."""

    ok: bool = False
    status: int = 0
    data: Any = None
    error: str = ""
    body: str = ""
    elapsed_ms: float = 0.0

    @property
    def message(self) -> str:
        return self.error or self.body[:200]


@dataclass
class Health:
    """Parsed ``GET /health`` result."""

    state: str = HEALTH_UNREACHABLE
    status: int = 0
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state == HEALTH_READY

    @property
    def loading(self) -> bool:
        return self.state == HEALTH_LOADING

    def describe(self) -> str:
        if self.state == HEALTH_READY:
            return "就绪"
        if self.state == HEALTH_LOADING:
            return "正在加载模型"
        if self.state == HEALTH_UNREACHABLE:
            return "未响应"
        return f"异常：{self.detail}" if self.detail else "异常"


class ServerClient:
    """Small synchronous client.  Callers run it off the UI thread."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        api_key: str = "",
        api_prefix: str = "",
        use_ssl: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self.host = host or "127.0.0.1"
        self.port = int(port or 8080)
        self.api_key = api_key or ""
        self.api_prefix = "/" + api_prefix.strip("/") if api_prefix.strip("/") else ""
        self.use_ssl = bool(use_ssl)
        self.timeout = timeout

    # ----------------------------------------------------------------- addressing

    @property
    def scheme(self) -> str:
        return "https" if self.use_ssl else "http"

    @property
    def connect_host(self) -> str:
        """The address a *client* should use to reach the server.

        A wildcard listen address (``0.0.0.0`` / ``::``) means "every local
        interface"; it is not something you can connect to - on Windows the
        attempt fails outright - so loopback is used instead.
        """
        host = self.host.strip()
        bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
        if bare in ("", "0.0.0.0", "*"):
            return "127.0.0.1"
        if bare == "::":
            return "::1"
        return bare

    @property
    def authority(self) -> str:
        host = self.connect_host
        if ":" in host:
            host = f"[{host}]"  # IPv6 literals must be bracketed in a URL
        return f"{host}:{self.port}"

    @property
    def base_url(self) -> str:
        """Base URL including any API prefix, without a trailing slash."""
        return f"{self.scheme}://{self.authority}{self.api_prefix}"

    @property
    def web_ui_url(self) -> str:
        """Where llama-server's own web UI is served (never includes a prefix)."""
        return f"{self.scheme}://{self.authority}{self.api_prefix or ''}/"

    @property
    def openai_base(self) -> str:
        """``base_url`` suitable for OpenAI-compatible SDKs."""
        return f"{self.base_url}/v1"

    def url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------ requests

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "llamagui"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> HttpResult:
        """Perform one request.  Never raises for network-level failures."""
        url = self.url(path)
        data = body
        headers = self._headers({"Content-Type": content_type} if (payload is not None or body)
                                else None)
        if payload is not None:
            try:
                data = json.dumps(payload).encode("utf-8")
            except (TypeError, ValueError) as exc:
                return HttpResult(ok=False, error=f"请求体序列化失败：{exc}")
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
                elapsed = (time.perf_counter() - started) * 1000.0
                text = raw.decode("utf-8", errors="replace")
                parsed: Any = None
                if text.strip():
                    try:
                        parsed = json.loads(text)
                    except ValueError:
                        parsed = None
                return HttpResult(
                    ok=200 <= resp.status < 300,
                    status=resp.status,
                    data=parsed,
                    body=text,
                    elapsed_ms=elapsed,
                )
        except urllib.error.HTTPError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            parsed = None
            if text.strip():
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = None
            return HttpResult(
                ok=False,
                status=exc.code,
                data=parsed,
                body=text,
                error=f"HTTP {exc.code}",
                elapsed_ms=elapsed,
            )
        except urllib.error.URLError as exc:
            return HttpResult(ok=False, error=f"连接失败：{exc.reason}")
        except (socket.timeout, TimeoutError):
            return HttpResult(ok=False, error="请求超时")
        except OSError as exc:
            return HttpResult(ok=False, error=f"网络错误：{exc}")

    # ------------------------------------------------------------------- health

    def health(self, timeout: Optional[float] = 2.0) -> Health:
        """Interpret ``GET /health``.

        The documented contract is 503 while loading and 200 once ready; the
        endpoint is public and never requires the API key.  Anything else
        (connection refused, timeout) means the process is not listening yet.
        """
        if not self.host or not self.port:
            return Health(state=HEALTH_UNREACHABLE, detail="未配置地址")
        url = self.url("/health")
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if resp.status == 200:
                    if '"ok"' in body or not body.strip():
                        return Health(state=HEALTH_READY, status=200, detail=body[:200])
                    return Health(state=HEALTH_READY, status=200, detail=body[:200])
                return Health(state=HEALTH_ERROR, status=resp.status, detail=body[:200])
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                pass
            if exc.code == 503:
                return Health(state=HEALTH_LOADING, status=503, detail=_error_message(body))
            return Health(state=HEALTH_ERROR, status=exc.code, detail=_error_message(body) or body[:200])
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
            return Health(state=HEALTH_UNREACHABLE, detail=str(getattr(exc, "reason", exc)))

    def wait_ready(
        self,
        *,
        timeout_s: float = 600.0,
        interval_s: float = 0.75,
        should_stop=None,  # noqa: ANN001
        on_tick=None,  # noqa: ANN001
        retarget=None,  # noqa: ANN001
    ) -> Health:
        """Poll until ready, unrecoverable, or *should_stop* returns True.

        *retarget*, when given, is called before every poll and returns the
        client to poll with.  It lets the caller switch to the address the
        server actually reported in its log, should that differ from the one
        this client was built for.
        """
        deadline = time.monotonic() + max(1.0, timeout_s)
        last = Health(state=HEALTH_UNREACHABLE, detail="尚未开始")
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return last
            client = self
            if retarget is not None:
                try:
                    client = retarget() or self
                except Exception:  # noqa: BLE001 - never break the poll loop
                    client = self
            last = client.health(timeout=min(2.0, interval_s + 1.5))
            if on_tick is not None:
                on_tick(last)
            if last.ready:
                return last
            time.sleep(interval_s)
        if not last.detail:
            last.detail = f"等待就绪超过 {timeout_s:.0f} 秒"
        return last

    # --------------------------------------------------------------- properties

    def props(self, timeout: Optional[float] = 3.0) -> HttpResult:
        return self.request("/props", timeout=timeout)

    def models(self, timeout: Optional[float] = 3.0) -> HttpResult:
        return self.request("/v1/models", timeout=timeout)

    def slots(self, timeout: Optional[float] = 3.0) -> HttpResult:
        return self.request("/slots", timeout=timeout)

    def metrics(self, timeout: Optional[float] = 3.0) -> Tuple[Dict[str, float], str]:
        """Fetch and parse Prometheus metrics into ``{metric_name: value}``."""
        result = self.request("/metrics", timeout=timeout)
        if not result.ok or not result.body:
            return {}, result.error or "无法获取 /metrics"
        numbers: Dict[str, float] = {}
        for line in result.body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "{" in line:
                continue  # labelled series; the GUI only needs the simple ones
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, value = parts[0].strip(), parts[1].strip()
            try:
                numbers[name] = float(value)
            except ValueError:
                continue
        return numbers, ""

    # -------------------------------------------------------------------- chat

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 64,
        model: str = "",
        temperature: Optional[float] = None,
        timeout: float = 60.0,
        extra: Optional[dict] = None,
    ) -> Iterator[dict]:
        """Yield raw SSE chunk dictionaries from ``/v1/chat/completions``.

        Raises ``OSError``/``ValueError`` on transport failure so the caller can
        present a precise message; individual chunk payloads are yielded as-is
        because the reasoning/content split varies by template.
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "max_tokens": int(max_tokens),
        }
        if model:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if extra:
            payload.update(extra)

        req = urllib.request.Request(
            self.url("/v1/chat/completions"),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=self._headers({"Content-Type": "application/json"}),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                if isinstance(chunk, dict):
                    yield chunk

    def self_test(self, model: str = "", timeout: float = 60.0) -> HttpResult:
        """Send a one-token chat request to prove the pipeline works."""
        started = time.perf_counter()
        tokens = 0
        text = ""
        try:
            for chunk in self.stream_chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                model=model,
                timeout=timeout,
            ):
                tokens += 1
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta") or {}
                    text += str(delta.get("content") or "")
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout,
                TimeoutError, OSError) as exc:
            return HttpResult(ok=False, error=f"自检失败：{exc}")
        elapsed = (time.perf_counter() - started) * 1000.0
        return HttpResult(
            ok=True,
            status=200,
            data={"chunks": tokens, "preview": text},
            body=text,
            elapsed_ms=elapsed,
        )


def parse_listen_url(url: str) -> Optional[Tuple[str, int, bool]]:
    """``(host, port, use_ssl)`` from a "listening on" URL, or ``None``."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname or not port:
        return None
    return parts.hostname, port, parts.scheme == "https"


def _error_message(body: str) -> str:
    """Pull ``error.message`` out of an OpenAI-shaped error body."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or "")
        if isinstance(err, str):
            return err
        if "message" in data:
            return str(data["message"])
    return body.strip()[:200]


#: Endpoint catalogue rendered by the API tab.  ``(name, method, path, note)``.
ENDPOINTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("健康检查", "GET", "/health", "503 = 正在加载，200 = 就绪。无需 API 密钥。"),
    ("服务属性", "GET", "/props", "返回 n_ctx、采样默认值、对话模板、模态能力。"),
    ("模型列表", "GET", "/v1/models", "OpenAI 兼容；id 即 --alias 或模型路径。"),
    ("对话补全", "POST", "/v1/chat/completions", "OpenAI 兼容，支持流式、工具调用、结构化输出。"),
    ("文本补全", "POST", "/v1/completions", "OpenAI 兼容补全端点。"),
    ("Responses", "POST", "/v1/responses", "OpenAI Responses API 兼容。"),
    ("嵌入", "POST", "/v1/embeddings", "需要 pooling 不为 none 的模型。"),
    ("分词", "POST", "/tokenize", "文本转 token，用于估算 prompt 长度。"),
    ("反分词", "POST", "/detokenize", "token 转文本。"),
    ("应用模板", "POST", "/apply-template", "查看模型对话模板渲染结果，排错利器。"),
    ("槽位监控", "GET", "/slots", "每个槽位的处理状态与速度。"),
    ("指标", "GET", "/metrics", "Prometheus 格式，需要 --metrics。"),
    ("重排序", "POST", "/reranking", "需要 --reranking 与 rerank 模型。"),
    ("代码填充", "POST", "/infill", "FIM 补全，适合编辑器插件。"),
    ("Anthropic 消息", "POST", "/v1/messages", "Anthropic Messages API 兼容。"),
    ("令牌计数", "POST", "/v1/messages/count_tokens", "不生成，只计 token。"),
    ("模型路由列表", "GET", "/models", "仅路由模式：列出可用模型与加载状态。"),
)


def curl_example(client: ServerClient, model: str = "") -> str:
    """A copy-pasteable curl invocation for the current server."""
    model = model or "local-model"
    auth = f' -H "Authorization: Bearer {client.api_key}"' if client.api_key else ""
    return (
        f'curl {client.url("/v1/chat/completions")}'
        f' -H "Content-Type: application/json"{auth}'
        f' -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "你好"}}],'
        f' "stream": false}}\''
    )


def python_example(client: ServerClient, model: str = "") -> str:
    """A copy-pasteable ``openai`` SDK snippet."""
    model = model or "local-model"
    key = client.api_key or "sk-no-key-required"
    return (
        "from openai import OpenAI\n"
        f'client = OpenAI(base_url="{client.openai_base}", api_key="{key}")\n'
        "resp = client.chat.completions.create(\n"
        f'    model="{model}",\n'
        '    messages=[{"role": "user", "content": "你好"}],\n'
        ")\n"
        "print(resp.choices[0].message.content)"
    )


def env_example(client: ServerClient, model: str = "") -> str:
    """Environment variables for tools that read OpenAI-style settings."""
    model = model or "local-model"
    key = client.api_key or "sk-no-key-required"
    return "\n".join(
        [
            f"OPENAI_BASE_URL={client.openai_base}",
            f"OPENAI_API_KEY={key}",
            f"OPENAI_MODEL={model}",
            f"# 内置 Web UI: {client.web_ui_url}",
        ]
    )
