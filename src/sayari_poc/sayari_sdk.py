"""The sole SDK boundary: audited I/O, raw replay and domain normalization."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Self, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError
from sayari.base_client import BaseClient
from sayari.core.api_error import ApiError
from sayari.core.client_wrapper import SyncClientWrapper
from sayari.core.oauth_token_provider import OAuthTokenProvider
from sayari.core.request_options import RequestOptions
from sayari.entity import EntitySummaryResponse
from sayari.ontology.types.get_ontology_risk_factors_response import GetOntologyRiskFactorsResponse
from sayari.resolution import ResolutionResponse as SdkResolutionResponse
from sayari.supply_chain import UpstreamTradeTraversalResponse

from sayari_poc.cache import ResponseCache
from sayari_poc.config import Settings
from sayari_poc.models import (
    EntityProfile,
    EntityProfileResponse,
    InputEntity,
    ResolutionCandidate,
    ResolvedEntity,
    RiskFactor,
    SupplyPath,
    UpstreamEntity,
    UpstreamResult,
)
from sayari_poc.transport import (
    AuditCounters,
    AuditedTransport,
    CallState,
    OfflineCacheMiss,
    RateLimitPacer,
    SayariAuthError,
    SayariError,
    SayariNotFound,
    SayariRateLimitError,
    SayariValidationError,
)

_Response = TypeVar("_Response")


class SayariClient:
    """Synchronous SDK facade for audited retrieval and normalization."""

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
        """Create a sequential SDK facade with audited replay first.

        The service must be a credential-free HTTPS origin. The live SDK is only built on a
        permitted cache miss or refresh; offline calls need no credentials. Invalid origins or
        conflicting modes raise ValueError.
        """
        base = urlsplit(settings.sayari_api_base)
        # Insist on a bare HTTPS origin: no credentials, query, fragment or endpoint path.
        if (
            base.scheme != "https"
            or not base.hostname
            or base.username
            or base.password
            or base.query
            or base.fragment
            or base.path not in ("", "/")
        ):
            raise ValueError("SAYARI_API_BASE must be an HTTPS origin without credentials")
        self._settings = settings
        self._transport = AuditedTransport(
            settings, cache, offline=offline, refresh=refresh, transport=transport, pacer=pacer
        )
        self._http = httpx.Client(
            transport=self._transport,
            trust_env=False,
            timeout=settings.sdk_timeout_seconds,
            follow_redirects=True,
        )
        # Verified in sayari/core/client_wrapper.py: BaseClientWrapper.get_headers calls the token
        # provider before sending data. Two separate SDK clients keep the replay token out of live
        # requests, and cache hits don't spend any auth calls.
        self._replay_sdk = self._make_sdk()
        self._live_sdk: BaseClient | None = None
        # Hand the configured retry count and timeout to the SDK as request options. Verified in
        # sayari/core/request_options.py: RequestOptions; sayari/core/http_client.py:
        # HttpClient.request. RequestOptions types the timeout as int, but HttpClient.request
        # passes it straight to httpx, which takes a float, so a fractional timeout works.
        self._options = RequestOptions(
            max_retries=settings.sdk_max_retries,
            timeout_in_seconds=settings.sdk_timeout_seconds,  # pyright: ignore[reportArgumentType]
        )

    def _make_sdk(self) -> BaseClient:
        """Share audited transport between SDK auth and endpoints."""
        settings = self._settings
        secret = settings.sayari_client_secret
        # Let the SDK fetch and refresh its own token, but through our audited client. Verified in
        # sayari/core/oauth_token_provider.py: OAuthTokenProvider.get_token and _refresh.
        provider = OAuthTokenProvider(
            client_id=settings.sayari_client_id or "",
            client_secret=secret.get_secret_value() if secret is not None else "",
            client_wrapper=SyncClientWrapper(
                base_url=settings.sayari_api_base,
                httpx_client=self._http,
                timeout=settings.sdk_timeout_seconds,
            ),
        )
        # Verified in sayari/client.py: Sayari.__init__ omits the override accepted by
        # sayari/base_client.py: BaseClient.__init__. We use the generated client so the SDK token
        # provider shares our audited transport and TLS configuration.
        return BaseClient(
            base_url=settings.sayari_api_base,
            client_id=settings.sayari_client_id or "",
            client_secret=secret.get_secret_value() if secret is not None else "",
            httpx_client=self._http,
            timeout=settings.sdk_timeout_seconds,
            _token_getter_override=provider.get_token,
        )

    @property
    def audit(self) -> AuditCounters:
        """Transport-owned execution counters for the run manifest."""
        return self._transport.audit

    @property
    def calls_made(self) -> int:
        """Data HTTP attempts, including retries but excluding auth."""
        return self.audit.data_http_attempts

    @property
    def cache_lookups(self) -> int:
        """Logical cache lookups, excluding refresh bypasses."""
        return self.audit.cache_lookups

    @property
    def cache_hits(self) -> int:
        """Logical calls satisfied by raw cached responses."""
        return self.audit.cache_hits

    def __enter__(self) -> Self:
        """Return this facade for deterministic context-managed cleanup."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the HTTP client without suppressing caller exceptions."""
        self.close()

    def close(self) -> None:
        """Release the shared HTTP client and its transport."""
        self._http.close()

    def _call(self, operation: Callable[[BaseClient], _Response]) -> tuple[_Response, CallState]:
        """Run one SDK operation with isolated state and safe errors.

        Replay uses an independent token provider. Only a genuine cache miss can fall back to live
        retrieval, and offline mode forbids that fallback.
        """
        state = CallState()

        def record_state(received: CallState) -> None:
            """Capture only this logical call's transport-validated metadata."""
            nonlocal state
            state = received

        try:
            with self._transport.logical_call(record_state):
                if not self._transport.refresh:
                    # Only a real cache miss may fall through to a live request.
                    try:
                        response = operation(self._replay_sdk)
                        return response, state
                    except OfflineCacheMiss:
                        if self._transport.offline:
                            raise
                with self._transport.live_attempts():
                    if self._live_sdk is None:
                        # A fresh token provider keeps the replay placeholder token from ever being
                        # used as a live credential.
                        self._live_sdk = self._make_sdk()
                    response = operation(self._live_sdk)
                    return response, state
        except ApiError as exc:
            # Read only the status code; the error's string form includes the response body.
            # Verified in sayari/core/api_error.py: ApiError.__init__ and __str__.
            status = exc.status_code
            if status in (401, 403):
                raise SayariAuthError(
                    "Sayari authentication or access rejected", call_state=state
                ) from None
            if status == 404:
                raise SayariNotFound("Sayari resource not found", call_state=state) from None
            if status == 429:
                raise SayariRateLimitError(
                    "Sayari rate limit retries exhausted", call_state=state
                ) from None
            raise SayariError("Sayari request failed", call_state=state) from None
        except SayariError as exc:
            # Already a safe domain error: keep its type and attach this call's state.
            raise type(exc)(str(exc), call_state=state) from None
        except ValidationError:
            raise SayariValidationError("Malformed Sayari response", call_state=state) from None
        except (httpx.HTTPError, OSError):
            raise SayariError(
                "Sayari HTTP transport or cache access failed", call_state=state
            ) from None
        except ValueError:
            raise SayariValidationError(
                "Malformed Sayari response or cache", call_state=state
            ) from None

    @staticmethod
    def _validate_id(entity_id: str) -> None:
        """Reject path delimiters before the SDK interpolates an entity ID."""
        # Verified in sayari/entity/client.py: EntityClient.get_entity interpolates
        # jsonable_encoder(id) into the path without escaping, so path delimiters must be rejected.
        if re.fullmatch(r"[A-Za-z0-9_-]+", entity_id) is None:
            raise SayariValidationError("Invalid Sayari entity ID")

    def resolve(self, entity: InputEntity) -> ResolvedEntity:
        """Resolve one row while retaining candidate evidence and failures.

        Only the original primary candidate can establish canonical identity, and only with explicit
        strong match strength. Malformed alternates keep their response positions; scores remain raw
        relevance values.
        """
        result = ResolvedEntity(
            row_number=entity.row_number,
            sheet=entity.sheet,
            input_name=entity.name,
            status="error",
        )
        try:
            response: SdkResolutionResponse
            response, _ = self._call(
                lambda sdk: sdk.resolution.resolution(
                    name=entity.name,
                    address=entity.address,
                    country=entity.country,
                    request_options=self._options,
                )
            )
            result.candidate_count = len(response.data)
            for index, raw in enumerate(response.data):
                # Set aside a malformed alternate without changing which entry is primary.
                try:
                    candidate = ResolutionCandidate.model_validate(raw.model_dump())
                except ValidationError:
                    # If the first candidate is unusable, the whole resolution fails.
                    if index == 0:
                        raise
                    result.candidate_errors[index] = "Malformed resolution candidate"
                else:
                    result.candidates.append(candidate)
            if result.candidates:
                primary = result.candidates[0]
                result.entity_id = primary.entity_id
                result.label = primary.label
                result.translated_label = primary.translated_label
                result.match_strength = primary.match_strength.value
                # Copy the raw relevance score as is; don't scale it or treat it as confidence.
                result.score = primary.score
                # Only an explicit "strong" match is accepted; weak matches stay for review.
                result.status = "resolved" if primary.match_strength.value == "strong" else "weak"
            else:
                result.status = "no_match"
        except (ValidationError, SayariValidationError):
            result.error = "Malformed resolution response"
            result.error_type = "ValidationError"
        except SayariError as exc:
            result.error, result.error_type = "Resolution request failed", type(exc).__name__
        except Exception as exc:
            result.error, result.error_type = "Unexpected resolution failure", type(exc).__name__
        return result

    def get_entity(self, entity_id: str) -> EntityProfile:
        """Fetch and validate the consumed entity-summary fields.

        Preserve original risk scalar types before domain validation. Reject a profile belonging to
        another ID; failures cross as safe SayariError types.
        """
        self._validate_id(entity_id)
        response: EntitySummaryResponse
        # Fetch the condensed profile through the SDK entity endpoint, inside the audited call.
        # entity_summary leaves out relationships, which nothing here reads, and returns the same
        # identity, country, PSA and degree evidence. Verified in sayari/entity/client.py:
        # EntityClient.entity_summary.
        response, state = self._call(
            lambda sdk: sdk.entity.entity_summary(entity_id, request_options=self._options)
        )
        try:
            payload = response.model_dump()
            for factor, value in state.risk_values:
                # Put back the scalar types exactly as received before strict validation. Verified
                # in sayari/shared_types/types/risk_value.py: RiskValue omits int from its union.
                payload["risk"][factor]["value"] = value
            profile = EntityProfileResponse.model_validate(payload)
            if profile.id != entity_id:
                raise SayariValidationError("Entity profile ID does not match requested entity")
            # Drop only a literal False; a numeric zero is still something Sayari observed.
            factors = [
                RiskFactor(
                    factor=name, value=value.value, level=value.level, metadata=value.metadata
                )
                for name, value in profile.risk.items()
                if value.value is not False
            ]
            levels = {factor.level for factor in factors}
            return EntityProfile(
                entity_id=profile.id,
                label=profile.label,
                translated_label=profile.translated_label,
                countries=profile.countries,
                psa_count=profile.psa_count,
                degree=profile.degree,
                risk_factors=factors,
                max_level=next(
                    (x for x in ("critical", "high", "elevated", "relevant") if x in levels), None
                ),
            )
        except ValidationError:
            raise SayariValidationError("Malformed entity profile") from None

    def get_risk_factors(self) -> GetOntologyRiskFactorsResponse:
        """Fetch unfiltered ontology through the audited SDK boundary."""
        # The pinned SDK ontology model requires do_not_render_metadata, which some committed
        # records omit, so a successful HTTP response can still fail to parse. We keep the raw bytes
        # for replay rather than writing a replacement parser. See
        # sayari/ontology/types/ontology_risk_factor.py: OntologyRiskFactor.
        response, _ = self._call(
            lambda sdk: sdk.ontology.get_risk_factors(request_options=self._options)
        )
        return response

    def upstream(
        self, supplier_id: str, *, max_depth: int | None = None, limit: int | None = None
    ) -> UpstreamResult:
        """Return bounded trade evidence or an explicit per-supplier error.

        Received tiers and path order stay intact. Partial flags outrank empty success; invalid
        coverage or identity evidence cannot become no_data. Failure metadata is diagnostic, not
        proof of a completed retrieval.
        """
        result = UpstreamResult(
            supplier_id=supplier_id,
            entities={},
            partial_results=False,
            explored_count=None,
            status="error",
        )
        state = CallState()
        try:
            self._validate_id(supplier_id)
            response: UpstreamTradeTraversalResponse
            response, state = self._call(
                lambda sdk: sdk.supply_chain.upstream_trade_traversal(
                    supplier_id,
                    max_depth=self._settings.max_upstream_depth if max_depth is None else max_depth,
                    limit=self._settings.upstream_limit if limit is None else limit,
                    request_options=self._options,
                )
            )
            if response.success is False:
                raise SayariValidationError("Upstream response reports failure", call_state=state)
            entities: dict[str, UpstreamEntity] = {}
            for entity_id, raw in response.data.entities.items():
                if raw.id != entity_id:
                    raise SayariValidationError(
                        "Upstream entity ID does not match dictionary key", call_state=state
                    )
                # Keep identity, labels, countries and factors. translated_label is an extra field:
                # use it when present and never invent one. TradeTraversalEntity permits extras;
                # verified in sayari/supply_chain/types/trade_traversal_entity.py.
                entities[entity_id] = UpstreamEntity(
                    entity_id=entity_id,
                    label=raw.label,
                    translated_label=(raw.model_extra or {}).get("translated_label"),
                    countries=raw.countries,
                    country_count=len(raw.countries),
                    risk_factors=raw.risk_factors,
                )
            paths = [
                SupplyPath.model_validate(
                    {
                        "source_entity_id": raw.source_entity_id,
                        "path_index": index,
                        "hops": [hop.model_dump() for hop in raw.path],
                    }
                )
                for index, raw in enumerate(response.data.paths or [])
            ]
            # Every hop we keep must have a matching entity in this same response.
            if any(hop.entity_id not in entities for path in paths for hop in path.hops):
                raise SayariValidationError(
                    "Upstream path hop is absent from response entities", call_state=state
                )
            # Set entities and paths together, and only once everything above has validated.
            result.entities, result.paths = entities, paths
            # Partial coverage wins; then tell bounded non-empty evidence from an empty success.
            result.status = (
                "partial" if response.partial_results else ("assessed" if entities else "no_data")
            )
        except ValidationError:
            result.error_type = "ValidationError"
        except SayariError as exc:
            state = exc.call_state
            result.error_type = (
                "ValidationError" if isinstance(exc, SayariValidationError) else type(exc).__name__
            )
        except Exception as exc:
            result.error_type = type(exc).__name__
        else:
            result.error_type = None
        finally:
            # Keep the raw-validated partial flag even on error, where it is only a diagnostic.
            result.partial_results = state.partial_results
            result.explored_count = state.explored_count
        return result
