"""SDK-independent request auditing, offline replay, budgets and pacing."""

from __future__ import annotations

import json
import logging
import ssl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import unquote, unquote_plus

import httpx

from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings


@dataclass(frozen=True)
class CallState:
    """Raw evidence needed across SDK parsing, scoped to one logical call."""

    # False is only a placeholder until Sayari sends a valid partial-results flag.
    partial_results: bool = False
    explored_count: int | None = None
    # The original risk scalars, kept immutable and left out of repr so they never show up in logs.
    risk_values: tuple[tuple[str, bool | int | float | str | None], ...] = field(
        default=(), repr=False
    )


def _is_upstream(path: str) -> bool:
    """Recognize upstream paths without matching similar prefixes."""
    return path == "/v1/supply_chain/upstream" or path.startswith("/v1/supply_chain/upstream/")


def _is_entity_profile(path: str) -> bool:
    """Recognize profile endpoints needing exact raw scalar types."""
    # Match both entity and entity_summary: each carries a risk mapping whose scalar types must
    # survive the SDK's RiskValue coercion. Checking only "/v1/entity/" would quietly miss
    # entity_summary, because "/v1/entity_summary/" doesn't share that prefix.
    return path.startswith("/v1/entity/") or path.startswith("/v1/entity_summary/")


class SayariError(Exception):
    """Safe boundary error; contains no request, response or credential."""

    def __init__(self, message: str, *, call_state: CallState | None = None) -> None:
        """Attach safe text and metadata without retaining HTTP objects."""
        super().__init__(message)
        self.call_state = call_state or CallState()


class BudgetExceeded(SayariError):
    """The next data attempt would exceed the configured budget."""


class SayariAuthError(SayariError):
    """Authentication or endpoint permissions failed."""


class SayariRateLimitError(SayariError):
    """The SDK exhausted its rate-limit retry allowance."""


class SayariNotFound(SayariError):
    """The requested resource was not found."""


class OfflineCacheMiss(SayariError):
    """A logical call has no stored response."""


class SayariValidationError(SayariError):
    """Response validation failed; raw errors remain private."""


@dataclass(frozen=True)
class HttpAttempt:
    """Payload-free attempt evidence for the run manifest."""

    # "auth" or "data", so credential requests are counted apart from chargeable data attempts.
    kind: str
    tier: int
    status: int | None
    # Only whether the server sent a Retry-After header; we never keep the headers themselves.
    retry_after: bool
    retry: bool = False


@dataclass
class AuditCounters:
    """Execution telemetry separate from deterministic evidence.

    Attributes:
        auth_http_attempts: Credential requests, outside the data-call budget.
        data_http_attempts: Actual endpoint attempts, including retries.
        cache_lookups: Logical calls consulting the cache, not HTTP attempts.
        cache_hits: Logical calls satisfied by cached evidence.
        attempts: Payload-free records in observed execution order.
        pacing_waits: Number of sleeps imposed by local rate-limit spacing.
        pacing_wait_seconds: Total requested sleep duration.
    """

    auth_http_attempts: int = 0
    data_http_attempts: int = 0
    # Logical calls that checked the cache. A --refresh run skips the cache, so it doesn't count.
    cache_lookups: int = 0
    cache_hits: int = 0
    attempts: list[HttpAttempt] = field(default_factory=list)
    pacing_waits: int = 0
    pacing_wait_seconds: float = 0.0


class RateLimitPacer:
    """Space actual attempts independently within each documented rate tier."""

    def __init__(
        self,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Inject monotonic time and sleep for reproducible pacing tests."""
        self._sleep = sleep
        self._monotonic = monotonic
        self._next: dict[int, float] = {}
        self.wait_count = 0
        self.wait_seconds = 0.0

    @staticmethod
    def tier(path: str) -> int:
        """Return Tier 2 for upstream requests, otherwise Tier 1."""
        return 2 if _is_upstream(path) else 1

    def wait(self, path: str) -> int:
        """Pace one attempt and reserve its next tier slot."""
        tier = self.tier(path)
        # Space attempts locally: Tier 2 allows 15 per 10 seconds, Tier 1 allows 200 per minute.
        interval = 10.0 / 15 if tier == 2 else 60.0 / 200
        now = self._monotonic()
        ready = max(now, self._next.get(tier, now))
        if ready > now:
            self._sleep(ready - now)
            self.wait_count += 1
            self.wait_seconds += ready - now
        # Book the next slot from whichever is later, the planned time or when we actually woke, so
        # an oversleep can't squeeze the following attempts closer together.
        self._next[tier] = max(ready, self._monotonic()) + interval
        return tier


@contextmanager
def _quiet_http_logs() -> Iterator[None]:
    """Suppress sensitive HTTP diagnostics and restore caller loggers."""
    # HTTPX logs full supplier query URLs and HTTPCore can log wire details. Silence just these two
    # libraries while a logical call is running, then put them back.
    names = sorted(
        {"httpx", "httpcore"}
        | {
            name
            for name in logging.Logger.manager.loggerDict
            if name.startswith(("httpx.", "httpcore."))
        }
    )
    saved = [
        (logging.getLogger(name), logging.getLogger(name).disabled, logging.getLogger(name).level)
        for name in names
    ]
    for logger, _, _ in saved:
        logger.disabled = True
        logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        for logger, disabled, level in saved:
            logger.disabled = disabled
            logger.setLevel(level)


class AuditedTransport(httpx.BaseTransport):
    """Sequential replay and safety checks before socket access."""

    def __init__(
        self,
        settings: Settings,
        cache: ResponseCache,
        *,
        offline: bool = False,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
        pacer: RateLimitPacer | None = None,
    ) -> None:
        """Configure replay-first auditing with lazy socket access.

        Injected transports and pacers support network-free verification. Each instance owns its
        counters and call state; offline and refresh cannot both be enabled.
        """
        if offline and refresh:
            raise ValueError("Refresh is incompatible with offline mode")
        self.settings = settings
        self.cache = cache
        self.offline = offline
        self.refresh = refresh
        self.audit = AuditCounters()
        self._attempt_start = 0
        self._transport = transport
        self._pacer = pacer or RateLimitPacer()
        self._secrets = [settings.sayari_client_id or ""]
        if settings.sayari_client_secret is not None:
            self._secrets.append(settings.sayari_client_secret.get_secret_value())
        # Start in replay mode so nothing can reach the network until live_attempts() is entered.
        self._replaying = True
        self._active = False
        self._key: str | None = None
        self._data_path = ""
        self._record_state: Callable[[CallState], None] = lambda state: None

    @contextmanager
    def logical_call(self, record_state: Callable[[CallState], None]) -> Iterator[None]:
        """Yield isolated endpoint state with guaranteed cleanup.

        The callback receives only metadata validated for this call. Nested calls are rejected
        because cache identity and retry accounting are sequential.
        """
        # This transport is stateful and strictly sequential, so refuse nested or overlapping calls.
        if self._active:
            raise SayariError("Concurrent or nested Sayari calls are prohibited")
        self._active = True
        self._attempt_start = len(self.audit.attempts)
        self._key = None
        self._data_path = ""
        self._record_state = record_state
        try:
            with _quiet_http_logs():
                yield
        finally:
            self._active = False
            self._replaying = True
            # Drop the finished call's state receiver so later metadata can't leak into it.
            self._record_state = lambda state: None

    @contextmanager
    def live_attempts(self) -> Iterator[None]:
        """Permit live attempts temporarily under all transport guards."""
        self._replaying = False
        try:
            yield
        finally:
            self._replaying = True

    def _reject_credentials(self, text: str) -> None:
        """Reject raw, URL-encoded, or JSON-encoded credentials."""
        # Check the original and URL-decoded forms so simple encoding can't hide a credential.
        forms = [text, unquote(text), unquote_plus(text)]
        try:
            # Normalise JSON string escaping for this check only; what we store is left untouched.
            forms.append(json.dumps(json.loads(text), ensure_ascii=False))
        except ValueError:
            pass
        if any(
            secret and encoded in value
            for secret in self._secrets
            for encoded in (
                secret,
                json.dumps(secret, ensure_ascii=True)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
            )
            for value in forms
        ):
            # Refuse the data without saying which credential or payload tripped the check.
            raise SayariError("Data request or response contains an authentication credential")

    def _record_response_state(self, content: bytes) -> None:
        """Retain approved raw scalars before SDK coercion."""
        if not _is_entity_profile(self._data_path):
            self._validate_coverage(content)
            return
        # Verified in sayari/shared_types/types/risk_value.py: RiskValue omits int. Keep the scalar
        # types as received, because once parsed as a float the original type is gone.
        try:
            payload = json.loads(content)
        except ValueError:
            raise SayariValidationError("Malformed entity profile") from None
        risk = payload.get("risk") if isinstance(payload, dict) else None
        if not isinstance(risk, dict):
            # If the risk mapping is missing, this side channel must not invent one. Verified in
            # sayari/entity/client.py: EntityClient.entity_summary parses EntitySummaryResponse.
            return
        values: list[tuple[str, bool | int | float | str | None]] = []
        for name, entry in risk.items():
            if not isinstance(entry, dict) or "value" not in entry:
                # Leave a malformed entry for the full response validation; don't make up a value.
                continue
            value = entry["value"]
            if type(value) not in (bool, int, float, str, type(None)):
                raise SayariValidationError("Malformed entity risk value")
            values.append((name, value))
        self._record_state(CallState(risk_values=tuple(values)))

    def _validate_coverage(self, content: bytes) -> None:
        """Reject coercions that could misstate coverage or tiers."""
        if not _is_upstream(self._data_path):
            return
        try:
            payload = json.loads(content)
        except ValueError:
            raise SayariValidationError("Malformed upstream response") from None
        if not isinstance(payload, dict):
            raise SayariValidationError("Malformed upstream response")
        # Verified in sayari/supply_chain/types/upstream_trade_traversal_response.py:
        # UpstreamTradeTraversalResponse uses coercing bool/int fields. For example, True would
        # become an explored count of 1 unless we check before the SDK parses it.
        partial, explored = payload.get("partial_results"), payload.get("explored_count")
        # Record whichever fields are valid even if the check below fails, as diagnostics. The
        # placeholders must never be read as measured coverage.
        self._record_state(
            CallState(
                partial_results=partial if type(partial) is bool else False,
                explored_count=explored if type(explored) is int and explored >= 0 else None,
            )
        )
        if type(partial) is not bool or type(explored) is not int or explored < 0:
            raise SayariValidationError("Malformed upstream coverage fields")
        # Verified in sayari/supply_chain/types/trade_traversal_path_segment.py:
        # TradeTraversalPathSegment.tier is a coercing int, so reject strings and booleans while we
        # can still see their original types. Validating the containers is left to the SDK.
        data = payload.get("data")
        paths = data.get("paths") if isinstance(data, dict) else None
        if isinstance(paths, list):
            for path in paths:
                hops = path.get("path") if isinstance(path, dict) else None
                if isinstance(hops, list):
                    for hop in hops:
                        if isinstance(hop, dict) and type(hop.get("tier")) is not int:
                            raise SayariValidationError("Malformed upstream hop tier")

    def _send(self, request: httpx.Request, *, auth: bool) -> httpx.Response:
        """Audit one paced HTTP attempt within the transport budget.

        Authentication uses a separate counter. This method records retry events; it never decides
        whether to retry, which remains the SDK's responsibility.
        """
        # Check the budget on every attempt, retries included, before any socket is opened. Token
        # requests have their own counter and don't use up the data allowance.
        if not auth and self.audit.data_http_attempts >= self.settings.call_budget:
            raise BudgetExceeded("Sayari data call budget exhausted")
        if self._transport is None:
            # With a custom transport in place, Client(verify=...) has no effect, so TLS has to be
            # set on the real socket transport, which we create lazily here.
            context = ssl.create_default_context(cafile=self.settings.sayari_ca_bundle)
            self._transport = httpx.HTTPTransport(verify=context, retries=0, trust_env=False)
        tier = self._pacer.wait(request.url.path)
        self.audit.pacing_waits = self._pacer.wait_count
        self.audit.pacing_wait_seconds = self._pacer.wait_seconds
        kind = "auth" if auth else "data"
        # Verified in sayari/core/http_client.py: _should_retry. We only label attempts we observe;
        # the SDK still decides whether to retry and how long to wait. Redirect hops and later
        # logical calls are not retries. An attempt counts as a retry when the previous attempt of
        # the same kind in this logical call got a retryable status.
        previous = next(
            (
                attempt
                for attempt in reversed(self.audit.attempts[self._attempt_start :])
                if attempt.kind == kind
            ),
            None,
        )
        retry = (
            previous is not None
            and previous.status is not None
            and (previous.status >= 500 or previous.status in {408, 409, 429})
        )
        if auth:
            self.audit.auth_http_attempts += 1
        else:
            self.audit.data_http_attempts += 1
        response: httpx.Response | None = None
        try:
            response = self._transport.handle_request(request)
            # Read the body into memory so endpoint parsing can use it after the stream closes.
            response.read()
            return response
        finally:
            self.audit.attempts.append(
                HttpAttempt(
                    kind=kind,
                    retry=retry,
                    tier=tier,
                    status=response.status_code if response is not None else None,
                    retry_after=response is not None and "retry-after" in response.headers,
                )
            )
            if response is not None:
                response.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Replay raw bytes or send one guarded SDK-built request.

        Origin and credential checks run before anything is cached or sent. Only successful data
        responses enter the cache; authentication is never cached.
        """
        if not self._active:
            raise SayariError("A request requires a logical call boundary")
        origin = httpx.URL(self.settings.sayari_api_base)
        # Refuse any request or redirect that leaves the configured Sayari origin.
        if (request.url.scheme, request.url.host, request.url.port) != (
            origin.scheme,
            origin.host,
            origin.port,
        ):
            raise SayariError("Cross-origin Sayari redirect refused")
        # Tell the token endpoint apart from data endpoints: it is replayed and counted separately.
        # Verified in sayari/auth/client.py: AuthClient.get_token builds oauth/token requests.
        auth = request.url.path == "/oauth/token"
        self._reject_credentials(str(request.url))
        # Token requests bypass both the response cache and the data-attempt budget.
        if auth:
            if self._replaying or self.offline:
                # Hand back a fixed replay-only token; nothing is sent and nothing is cached.
                return httpx.Response(
                    200,
                    json={
                        "access_token": "offline-replay",
                        "expires_in": 86400,
                        "token_type": "Bearer",
                    },
                )
            if not all(value.strip() for value in self._secrets[:2]) or len(self._secrets) < 2:
                raise SayariAuthError("Missing Sayari credentials")
            response = self._send(request, auth=True)
            if response.is_success:
                try:
                    token = response.json().get("access_token")
                except (ValueError, AttributeError):
                    token = None
                if not isinstance(token, str) or not token.strip():
                    raise SayariAuthError("Malformed Sayari authentication response")
                # Remember this token so we can catch it if it ever appears in data we log or cache.
                self._secrets.append(token)
            return response

        # Check the data request body for credentials before touching the cache or the network.
        self._reject_credentials(request.read().decode("utf-8"))
        if self._key is None:
            self._key = self.cache.key(request)
            self._data_path = request.url.path
            if not self.refresh:
                # Count the cache lookup once per logical call, however many real retries follow.
                self.audit.cache_lookups += 1
                cached = self.cache.get(self._key)
                if cached is not None:
                    # Refuse a cached body that contains a credential before the SDK parses it.
                    self._reject_credentials(cached.decode("utf-8"))
                    self.audit.cache_hits += 1
                    self._record_response_state(cached)
                    return httpx.Response(
                        200, content=cached, headers={"Content-Type": "application/json"}
                    )
        if self._replaying or self.offline:
            raise OfflineCacheMiss("No cached Sayari response")
        response = self._send(request, auth=False)
        # Check the response body for credentials before it is cached or parsed.
        self._reject_credentials(response.content.decode("utf-8", errors="replace"))
        if response.is_success:
            # An upstream body can be HTTP 200 yet report success: false; never cache that one.
            if _is_upstream(self._data_path):
                payload = response.json()
                if isinstance(payload, dict) and payload.get("success") is False:
                    self._record_response_state(response.content)
                    raise SayariValidationError("Upstream response reports failure")
            # Cache the raw body before model validation, so if parsing fails a fixed parser can
            # replay it without spending another data request.
            self.cache.put(self._key, response.content)
            self._record_response_state(response.content)
        return response

    def close(self) -> None:
        """Close existing transport without creating a socket client."""
        # Don't create a socket transport just to close an adapter that only ever replayed.
        if self._transport is not None:
            self._transport.close()
