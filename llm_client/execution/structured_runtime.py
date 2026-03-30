"""Internal runtimes for structured-output entrypoints.

This module owns the implementation behind the public
``call_llm_structured`` and ``acall_llm_structured`` facades. The public API
remains in ``client.py``; this module holds the structured-call control flow so
runtime logic can be grouped by workload family without changing
caller-facing signatures.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, NoReturn, TypeVar, cast

from llm_client.core.client import AsyncCachePolicy, CachePolicy, Hooks, LLMCallResult, RetryPolicy
from llm_client.core.config import ClientConfig
from llm_client.core.errors import LLMCapabilityError
from llm_client.langfuse_callbacks import inject_metadata as _inject_langfuse_metadata
from pydantic import BaseModel, ValidationError

import hashlib as _hashlib
import json as _json
import logging as _logging

from llm_client.parsing_utils import safe_json_loads as _safe_json_loads

T = TypeVar("T", bound=BaseModel)

_client: Any = import_module("llm_client.core.client")
_structured_logger = _logging.getLogger("llm_client.structured_runtime")


def _robust_validate_json(response_model: type[T], raw_content: str) -> T:
    """Parse and validate JSON from LLM output with best-effort extraction.

    Tries ``model_validate_json`` first (fast path).  If that fails due to
    JSON decoding issues (control characters, fenced markdown, etc.), falls
    back to ``safe_json_loads`` + ``model_validate`` which strips control
    chars, extracts JSON from fences/prose, and uses ``strict=False``.

    Pydantic ``ValidationError`` (schema mismatch) is never swallowed -- only
    JSON-level failures trigger the fallback.
    """
    try:
        return response_model.model_validate_json(raw_content)
    except ValidationError:
        # Schema validation error -- don't mask with fallback
        raise
    except Exception:
        # JSON decoding or other parse failure -- try robust extraction
        _structured_logger.debug(
            "model_validate_json failed on raw content (%d chars), "
            "falling back to safe_json_loads",
            len(raw_content),
        )
        parsed_data = _safe_json_loads(raw_content)
        return response_model.model_validate(parsed_data)


class _StructuredValidationRetry(Exception):
    """Retryable validation error from model_validate_json.

    Raised when the LLM provider returns syntactically valid JSON that passes
    the provider's schema check but fails Pydantic validation (e.g., the
    provider didn't enforce ``minProperties``).  Carries the raw content and
    formatted error so a repair message can be appended on retry.
    """

    def __init__(self, raw_content: str, validation_error: ValidationError) -> None:
        self.raw_content = raw_content
        self.validation_error = validation_error
        super().__init__(
            f"Pydantic validation failed on provider-accepted response: "
            f"{validation_error.error_count()} error(s). "
            f"First: {validation_error.errors()[0]['msg'] if validation_error.errors() else 'unknown'}"
        )


def _build_validation_repair_message(exc: _StructuredValidationRetry) -> dict[str, str]:
    """Build a user message that tells the model what went wrong.

    The repair message includes the specific validation errors so the model
    can fix its output on the next attempt rather than guessing blindly.
    """

    error_lines = []
    for err in exc.validation_error.errors():
        loc = " -> ".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "unknown error")
        error_lines.append(f"  - {loc}: {msg}")
    errors_text = "\n".join(error_lines[:5])  # Cap at 5 errors to avoid prompt bloat.
    return {
        "role": "user",
        "content": (
            "Your previous response was valid JSON but failed schema validation:\n"
            f"{errors_text}\n\n"
            "Please fix these issues and return a corrected response."
        ),
    }


def _base_model_name(model: str) -> str:
    """Return the provider-agnostic lowercase model name."""
    return model.lower().rsplit("/", 1)[-1]


def _is_gpt5_family_model(model: str) -> bool:
    """Return whether a model belongs to the GPT-5 family."""
    return _base_model_name(model).startswith("gpt-5")


def _is_invalid_json_schema_error(error: Exception) -> bool:
    """Return whether an exception indicates provider-side schema rejection."""
    message = str(error).lower()
    return "invalid_json_schema" in message or (
        "invalid schema" in message and "json_schema" in message
    )


def _raise_if_unsupported_gpt5_structured_schema(
    *,
    model: str,
    error: Exception,
    caller: str,
) -> None:
    """Raise a typed capability error for unsupported GPT-5 structured schema paths.

    GPT-5 family models can be selected for structured workloads, but some
    direct/provider-specific JSON-schema transports reject the supplied schema at
    request-validation time. When that happens, callers need a clear,
    non-retryable capability failure rather than a vague provider error or a
    fallback that obscures the real incompatibility.
    """
    if not _is_gpt5_family_model(model) or not _is_invalid_json_schema_error(error):
        return
    _raise_gpt5_structured_schema_capability_error(model=model, error=error, caller=caller)


def _raise_gpt5_structured_schema_capability_error(
    *,
    model: str,
    error: Exception,
    caller: str,
) -> NoReturn:
    """Raise the canonical GPT-5 structured-schema compatibility error."""
    raise LLMCapabilityError(
        f"{caller}: provider rejected structured JSON-schema output for GPT-5-family model "
        f"{model}. llm_client does not currently support this transport/schema combination "
        "reliably. Use a different task/model, or change routing/provider strategy.",
        original=error,
    ) from error


def _call_llm_structured_impl(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Run the synchronous structured-call runtime behind ``client.call_llm_structured``."""
    time = _client.time
    logger = _client.logger
    litellm = _client.litellm
    _rate_limit = _client._rate_limit
    _check_model_deprecation = _client._check_model_deprecation
    _normalize_prompt_ref = _client._normalize_prompt_ref
    _require_tags = _client._require_tags
    _normalize_timeout = _client._normalize_timeout
    _check_budget = _client._check_budget
    _resolve_call_plan = _client._resolve_call_plan
    _routing_policy_label = _client._routing_policy_label
    _is_agent_model = _client._is_agent_model
    _finalize_result = _client._finalize_result
    _build_routing_trace = _client._build_routing_trace
    _log_call_event = _client._log_call_event
    _effective_retry = _client._effective_retry
    _resolve_api_base_for_model = _client._resolve_api_base_for_model
    _background_mode_for_model = _client._background_mode_for_model
    _is_responses_api_model = _client._is_responses_api_model
    _cache_key = _client._cache_key
    exponential_backoff = _client.exponential_backoff
    _strict_json_schema = _client._strict_json_schema
    _prepare_responses_kwargs = _client._prepare_responses_kwargs
    _extract_responses_usage = _client._extract_responses_usage
    _parse_cost_result = _client._parse_cost_result
    _compute_responses_cost = _client._compute_responses_cost
    _build_structured_call_result = _client._build_structured_call_result
    run_sync_with_retry = _client.run_sync_with_retry
    _check_retryable = _client._check_retryable
    _compute_retry_delay = _client._compute_retry_delay
    _maybe_retry_with_openrouter_key_rotation = _client._maybe_retry_with_openrouter_key_rotation
    _prepare_call_kwargs = _client._prepare_call_kwargs
    _first_choice_or_empty_error = _client._first_choice_or_empty_error
    _extract_usage = _client._extract_usage
    _compute_cost = _client._compute_cost
    _is_schema_error = _client._is_schema_error
    _NativeSchemaFallback = _client._NativeSchemaFallback
    run_sync_with_fallback = _client.run_sync_with_fallback
    wrap_error = _client.wrap_error

    _check_model_deprecation(model)
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="call_llm_structured",
    )
    timeout = _normalize_timeout(
        timeout,
        caller="call_llm_structured",
        warning_sink=_entry_warnings,
        logger=logger,
        log_policy_once_enabled=True,
    )
    _check_budget(trace_id, max_budget)
    public_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)
    from llm_client.observability.replay import build_call_snapshot

    call_snapshot = build_call_snapshot(
        public_api="call_llm_structured",
        call_kind="structured",
        requested_model=model,
        messages=messages,
        prompt_ref=prompt_ref,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        fallback_models=fallback_models,
        public_kwargs=public_kwargs,
        response_model=response_model,
    )
    plan = _resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))

    if _is_agent_model(model):
        from llm_client.sdk.agents import _route_call_structured

        if hooks and hooks.before_call:
            hooks.before_call(model, messages, public_kwargs)
        parsed, llm_result = _route_call_structured(
            model, messages, response_model, timeout=timeout, **public_kwargs,
        )
        llm_result = _finalize_result(
            llm_result,
            requested_model=model,
            resolved_model=llm_result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[plan.primary_model],
                selected_model=llm_result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=api_base,
                routing_policy=routing_policy,
            ),
        )
        if hooks and hooks.after_call:
            hooks.after_call(llm_result)
        _log_call_event(
            model=model,
            messages=messages,
            result=llm_result,
            latency_s=time.monotonic() - _log_t0,
            caller="call_llm_structured",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
            execution_path="agent_sdk",
            retry_count=0,
        )
        return cast(T, parsed), llm_result
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    _warnings: list[str] = list(_entry_warnings)
    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"
    last_model_attempted = model

    def _execute_model(model_idx: int, current_model: str) -> tuple[T, LLMCallResult]:
        nonlocal last_model_attempted
        last_model_attempted = current_model
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        background_mode = _background_mode_for_model(
            model=current_model,
            use_responses=_is_responses_api_model(current_model),
            reasoning_effort=reasoning_effort,
        )
        key: str | None = None
        if cache is not None:
            key = _cache_key(current_model, messages, response_model=_model_fqn, **public_kwargs)
            cached = cache.get(key)
            if cached is not None:
                reparsed = response_model.model_validate_json(cached.content)
                cached_result = _finalize_result(
                    cached,
                    cache_hit=True,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    ),
                )
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="responses_api",
                    retry_count=0,
                )
                return reparsed, cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

        backoff_fn = r.backoff or exponential_backoff

        if _is_responses_api_model(current_model):
            schema = _strict_json_schema(response_model.model_json_schema())
            resp_kwargs = _prepare_responses_kwargs(
                current_model,
                messages,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            resp_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                }
            }

            def _invoke_responses_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                try:
                    with _rate_limit.acquire(current_model):
                        response = litellm.responses(**resp_kwargs)
                except Exception as exc:
                    _raise_if_unsupported_gpt5_structured_schema(
                        model=current_model,
                        error=exc,
                        caller="call_llm_structured",
                    )
                    raise
                raw_content = getattr(response, "output_text", None) or ""
                if not raw_content.strip():
                    raise ValueError("Empty content from LLM (responses API structured)")
                parsed = _robust_validate_json(response_model, raw_content)
                usage = _extract_responses_usage(response)
                cost, cost_source = _parse_cost_result(
                    _compute_responses_cost(response, usage),
                    default_source="computed",
                )

                if attempt > 0:
                    logger.info("call_llm_structured (responses) succeeded after %d retries", attempt)

                llm_result = _build_structured_call_result(
                    parsed=parsed,
                    usage=usage,
                    cost=cost,
                    cost_source=cost_source,
                    current_model=current_model,
                    finish_reason="stop",
                    raw_response=response,
                    warnings=_warnings,
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                )
                if hooks and hooks.after_call:
                    hooks.after_call(llm_result)
                if cache is not None and key is not None:
                    cache.set(key, llm_result)
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=llm_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="responses_api",
                    retry_count=attempt,
                )
                return parsed, llm_result

            return cast(tuple[T, LLMCallResult], run_sync_with_retry(
                caller="call_llm_structured",
                model=current_model,
                max_retries=r.max_retries,
                invoke=_invoke_responses_attempt,
                should_retry=lambda exc: _check_retryable(exc, r),
                compute_delay=lambda attempt, exc: _compute_retry_delay(
                    attempt=attempt,
                    error=exc,
                    policy=r,
                    backoff_fn=backoff_fn,
                ),
                warning_sink=_warnings,
                logger=logger,
                on_error=(hooks.on_error if hooks and hooks.on_error else None),
                on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="call_llm_structured",
                ),
            ))

        supports_schema = litellm.supports_response_schema(model=current_model)
        _native_schema_failed = False
        if supports_schema:
            schema = _strict_json_schema(response_model.model_json_schema())
            base_kwargs = _prepare_call_kwargs(
                current_model,
                messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            _schema_hash = _hashlib.sha256(
                _json.dumps(schema, sort_keys=True).encode()
            ).hexdigest()[:16]
            base_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

            _pending_repair_message: dict[str, str] | None = None

            def _invoke_native_schema_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                nonlocal _pending_repair_message
                if _pending_repair_message is not None:
                    base_kwargs["messages"] = list(base_kwargs["messages"]) + [_pending_repair_message]
                    _pending_repair_message = None
                    logger.info("call_llm_structured: appended validation repair message for attempt %d", attempt)
                try:
                    with _rate_limit.acquire(current_model):
                        response = litellm.completion(**base_kwargs)
                    first_choice = _first_choice_or_empty_error(
                        response,
                        model=current_model,
                        provider="litellm_completion_structured",
                    )
                    raw_content = first_choice.message.content or ""
                    if not raw_content.strip():
                        raise ValueError("Empty content from LLM (native JSON schema structured)")
                    try:
                        parsed = _robust_validate_json(response_model, raw_content)
                    except ValidationError as ve:
                        retry_exc = _StructuredValidationRetry(raw_content, ve)
                        _pending_repair_message = _build_validation_repair_message(retry_exc)
                        raise retry_exc from ve
                    usage = _extract_usage(response)
                    cost, cost_source = _parse_cost_result(_compute_cost(response))
                    finish_reason: str = first_choice.finish_reason or "stop"

                    if attempt > 0:
                        logger.info("call_llm_structured (native schema) succeeded after %d retries", attempt)

                    llm_result = _build_structured_call_result(
                        parsed=parsed,
                        usage=usage,
                        cost=cost,
                        cost_source=cost_source,
                        current_model=current_model,
                        finish_reason=finish_reason,
                        raw_response=response,
                        warnings=_warnings,
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    )
                    if hooks and hooks.after_call:
                        hooks.after_call(llm_result)
                    if cache is not None and key is not None:
                        cache.set(key, llm_result)
                    _log_call_event(
                        model=current_model,
                        messages=messages,
                        result=llm_result,
                        latency_s=time.monotonic() - _log_t0,
                        caller="call_llm_structured",
                        task=task,
                        trace_id=trace_id,
                        prompt_ref=prompt_ref,
                        call_snapshot=call_snapshot,
                        execution_path="native_schema",
                        retry_count=attempt,
                    )
                    return parsed, llm_result
                except Exception as exc:
                    _raise_if_unsupported_gpt5_structured_schema(
                        model=current_model,
                        error=exc,
                        caller="call_llm_structured",
                    )
                    if _is_schema_error(exc):
                        raise _NativeSchemaFallback(str(exc)) from exc
                    raise

            def _on_native_schema_error(exc: Exception, attempt: int) -> None:
                if isinstance(exc, _NativeSchemaFallback):
                    return
                if hooks and hooks.on_error:
                    hooks.on_error(exc, attempt)

            try:
                return cast(tuple[T, LLMCallResult], run_sync_with_retry(
                    caller="call_llm_structured",
                    model=current_model,
                    max_retries=r.max_retries,
                    invoke=_invoke_native_schema_attempt,
                    should_retry=lambda exc: (
                        isinstance(exc, _StructuredValidationRetry)
                        or (not isinstance(exc, _NativeSchemaFallback) and _check_retryable(exc, r))
                    ),
                    compute_delay=lambda attempt, exc: _compute_retry_delay(
                        attempt=attempt,
                        error=exc,
                        policy=r,
                        backoff_fn=backoff_fn,
                    ),
                    warning_sink=_warnings,
                    logger=logger,
                    on_error=_on_native_schema_error,
                    on_retry=r.on_retry,
                    maybe_retry_hook=lambda exc, attempt, max_retries: (
                        False if isinstance(exc, _NativeSchemaFallback) else _maybe_retry_with_openrouter_key_rotation(
                            error=exc,
                            attempt=attempt,
                            max_retries=max_retries,
                            current_model=current_model,
                            current_api_base=current_api_base,
                            user_kwargs=public_kwargs,
                            warning_sink=_warnings,
                            on_retry=r.on_retry,
                            caller="call_llm_structured",
                        )
                    ),
                ))
            except _NativeSchemaFallback as schema_error:
                logger.warning(
                    "Native JSON schema rejected by provider (%s), falling back to instructor: %s",
                    current_model,
                    schema_error,
                )
                _native_schema_failed = True

        if not supports_schema or _native_schema_failed:
            import instructor

            client = instructor.from_litellm(litellm.completion)
            base_kwargs = _prepare_call_kwargs(
                current_model,
                messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 2}

            def _invoke_instructor_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                parsed, completion_response = client.chat.completions.create_with_completion(
                    **call_kwargs,
                )

                usage = _extract_usage(completion_response)
                cost, cost_source = _parse_cost_result(_compute_cost(completion_response))
                completion_choice = _first_choice_or_empty_error(
                    completion_response,
                    model=current_model,
                    provider="instructor_completion_structured",
                )
                finish_reason = completion_choice.finish_reason or ""

                if attempt > 0:
                    logger.info("call_llm_structured succeeded after %d retries", attempt)

                llm_result = _build_structured_call_result(
                    parsed=parsed,
                    usage=usage,
                    cost=cost,
                    cost_source=cost_source,
                    current_model=current_model,
                    finish_reason=finish_reason,
                    raw_response=completion_response,
                    warnings=_warnings,
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                )

                if hooks and hooks.after_call:
                    hooks.after_call(llm_result)
                if cache is not None and key is not None:
                    cache.set(key, llm_result)
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=llm_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="call_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="instructor",
                    retry_count=attempt,
                )
                return parsed, llm_result

            return cast(tuple[T, LLMCallResult], run_sync_with_retry(
                caller="call_llm_structured",
                model=current_model,
                max_retries=r.max_retries,
                invoke=_invoke_instructor_attempt,
                should_retry=lambda exc: _check_retryable(exc, r),
                compute_delay=lambda attempt, exc: _compute_retry_delay(
                    attempt=attempt,
                    error=exc,
                    policy=r,
                    backoff_fn=backoff_fn,
                ),
                warning_sink=_warnings,
                logger=logger,
                on_error=(hooks.on_error if hooks and hooks.on_error else None),
                on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="call_llm_structured",
                ),
            ))

        raise RuntimeError("call_llm_structured reached unexpected branch without return")

    try:
        return cast(tuple[T, LLMCallResult], run_sync_with_fallback(
            models=models,
            execute_model=_execute_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=logger,
        ))
    except Exception as e:
        _log_call_event(
            model=last_model_attempted,
            messages=messages,
            error=e,
            latency_s=time.monotonic() - _log_t0,
            caller="call_llm_structured",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
            execution_path="error",
            retry_count=None,
        )
        raise wrap_error(e) from e


async def _acall_llm_structured_impl(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    *,
    timeout: int = 60,
    num_retries: int = 2,
    reasoning_effort: str | None = None,
    api_base: str | None = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: list[str] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    cache: CachePolicy | AsyncCachePolicy | None = None,
    retry: RetryPolicy | None = None,
    fallback_models: list[str] | None = None,
    on_fallback: Callable[[str, Exception, str], None] | None = None,
    hooks: Hooks | None = None,
    config: ClientConfig | None = None,
    **kwargs: Any,
) -> tuple[T, LLMCallResult]:
    """Run the async structured-call runtime behind ``client.acall_llm_structured``."""
    time = _client.time
    logger = _client.logger
    litellm = _client.litellm
    _rate_limit = _client._rate_limit
    _check_model_deprecation = _client._check_model_deprecation
    _normalize_prompt_ref = _client._normalize_prompt_ref
    _require_tags = _client._require_tags
    _normalize_timeout = _client._normalize_timeout
    _check_budget = _client._check_budget
    _resolve_call_plan = _client._resolve_call_plan
    _routing_policy_label = _client._routing_policy_label
    _is_agent_model = _client._is_agent_model
    _finalize_result = _client._finalize_result
    _build_routing_trace = _client._build_routing_trace
    _log_call_event = _client._log_call_event
    _effective_retry = _client._effective_retry
    _resolve_api_base_for_model = _client._resolve_api_base_for_model
    _background_mode_for_model = _client._background_mode_for_model
    _is_responses_api_model = _client._is_responses_api_model
    _cache_key = _client._cache_key
    _async_cache_get = _client._async_cache_get
    _async_cache_set = _client._async_cache_set
    exponential_backoff = _client.exponential_backoff
    _strict_json_schema = _client._strict_json_schema
    _prepare_responses_kwargs = _client._prepare_responses_kwargs
    _extract_responses_usage = _client._extract_responses_usage
    _parse_cost_result = _client._parse_cost_result
    _compute_responses_cost = _client._compute_responses_cost
    _build_structured_call_result = _client._build_structured_call_result
    run_async_with_retry = _client.run_async_with_retry
    _check_retryable = _client._check_retryable
    _compute_retry_delay = _client._compute_retry_delay
    _maybe_retry_with_openrouter_key_rotation = _client._maybe_retry_with_openrouter_key_rotation
    _prepare_call_kwargs = _client._prepare_call_kwargs
    _first_choice_or_empty_error = _client._first_choice_or_empty_error
    _extract_usage = _client._extract_usage
    _compute_cost = _client._compute_cost
    _is_schema_error = _client._is_schema_error
    _NativeSchemaFallback = _client._NativeSchemaFallback
    run_async_with_fallback = _client.run_async_with_fallback
    wrap_error = _client.wrap_error

    _check_model_deprecation(model)
    cfg = config or ClientConfig.from_env()
    _log_t0 = time.monotonic()
    task = kwargs.pop("task", None)
    trace_id = kwargs.pop("trace_id", None)
    max_budget: float | None = kwargs.pop("max_budget", None)
    prompt_ref = _normalize_prompt_ref(kwargs.pop("prompt_ref", None))
    task, trace_id, max_budget, _entry_warnings = _require_tags(
        task, trace_id, max_budget, caller="acall_llm_structured",
    )
    timeout = _normalize_timeout(
        timeout,
        caller="acall_llm_structured",
        warning_sink=_entry_warnings,
        logger=logger,
        log_policy_once_enabled=True,
    )
    _check_budget(trace_id, max_budget)
    public_kwargs = _client._strip_llm_internal_kwargs(dict(kwargs))
    _inject_langfuse_metadata(kwargs, task=task, trace_id=trace_id)
    from llm_client.observability.replay import build_call_snapshot

    call_snapshot = build_call_snapshot(
        public_api="acall_llm_structured",
        call_kind="structured",
        requested_model=model,
        messages=messages,
        prompt_ref=prompt_ref,
        timeout=timeout,
        num_retries=num_retries,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        base_delay=base_delay,
        max_delay=max_delay,
        retry_on=retry_on,
        fallback_models=fallback_models,
        public_kwargs=public_kwargs,
        response_model=response_model,
    )
    plan = _resolve_call_plan(
        model=model,
        fallback_models=fallback_models,
        api_base=api_base,
        config=cfg,
    )
    models = plan.models
    routing_policy = str(plan.routing_trace.get("routing_policy", _routing_policy_label(cfg)))

    if _is_agent_model(model):
        from llm_client.sdk.agents import _route_acall_structured

        if hooks and hooks.before_call:
            hooks.before_call(model, messages, public_kwargs)
        parsed, llm_result = await _route_acall_structured(
            model, messages, response_model, timeout=timeout, **public_kwargs,
        )
        llm_result = _finalize_result(
            llm_result,
            requested_model=model,
            resolved_model=llm_result.resolved_model,
            routing_trace=_build_routing_trace(
                requested_model=model,
                attempted_models=[plan.primary_model],
                selected_model=llm_result.resolved_model,
                requested_api_base=api_base,
                effective_api_base=api_base,
                routing_policy=routing_policy,
            ),
        )
        if hooks and hooks.after_call:
            hooks.after_call(llm_result)
        _log_call_event(
            model=model,
            messages=messages,
            result=llm_result,
            latency_s=time.monotonic() - _log_t0,
            caller="acall_llm_structured",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
            execution_path="agent_sdk",
            retry_count=0,
        )
        return cast(T, parsed), llm_result
    r = _effective_retry(retry, num_retries, base_delay, max_delay, retry_on, on_retry)
    _warnings: list[str] = list(_entry_warnings)
    _model_fqn = f"{response_model.__module__}.{response_model.__qualname__}"
    last_model_attempted = model

    async def _execute_model(model_idx: int, current_model: str) -> tuple[T, LLMCallResult]:
        nonlocal last_model_attempted
        last_model_attempted = current_model
        current_api_base = _resolve_api_base_for_model(current_model, api_base, cfg)
        background_mode = _background_mode_for_model(
            model=current_model,
            use_responses=_is_responses_api_model(current_model),
            reasoning_effort=reasoning_effort,
        )
        key: str | None = None
        if cache is not None:
            key = _cache_key(current_model, messages, response_model=_model_fqn, **public_kwargs)
            cached = await _async_cache_get(cache, key)
            if cached is not None:
                reparsed = response_model.model_validate_json(cached.content)
                cached_result = _finalize_result(
                    cached,
                    cache_hit=True,
                    requested_model=model,
                    resolved_model=current_model,
                    routing_trace=_build_routing_trace(
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        selected_model=current_model,
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    ),
                )
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=cached_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="responses_api",
                    retry_count=0,
                )
                return reparsed, cached_result

        if hooks and hooks.before_call:
            hooks.before_call(current_model, messages, public_kwargs)

        backoff_fn = r.backoff or exponential_backoff

        if _is_responses_api_model(current_model):
            schema = _strict_json_schema(response_model.model_json_schema())
            resp_kwargs = _prepare_responses_kwargs(
                current_model,
                messages,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            resp_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                }
            }

            async def _invoke_responses_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                try:
                    async with _rate_limit.aacquire(current_model):
                        response = await litellm.aresponses(**resp_kwargs)
                except Exception as exc:
                    _raise_if_unsupported_gpt5_structured_schema(
                        model=current_model,
                        error=exc,
                        caller="acall_llm_structured",
                    )
                    raise
                raw_content = getattr(response, "output_text", None) or ""
                if not raw_content.strip():
                    raise ValueError("Empty content from LLM (responses API structured)")
                parsed = _robust_validate_json(response_model, raw_content)
                usage = _extract_responses_usage(response)
                cost, cost_source = _parse_cost_result(
                    _compute_responses_cost(response, usage),
                    default_source="computed",
                )

                if attempt > 0:
                    logger.info("acall_llm_structured (responses) succeeded after %d retries", attempt)

                llm_result = _build_structured_call_result(
                    parsed=parsed,
                    usage=usage,
                    cost=cost,
                    cost_source=cost_source,
                    current_model=current_model,
                    finish_reason="stop",
                    raw_response=response,
                    warnings=_warnings,
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                )
                if hooks and hooks.after_call:
                    hooks.after_call(llm_result)
                if cache is not None and key is not None:
                    await _async_cache_set(cache, key, llm_result)
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=llm_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="responses_api",
                    retry_count=attempt,
                )
                return parsed, llm_result

            return cast(tuple[T, LLMCallResult], await run_async_with_retry(
                caller="acall_llm_structured",
                model=current_model,
                max_retries=r.max_retries,
                invoke=_invoke_responses_attempt,
                should_retry=lambda exc: _check_retryable(exc, r),
                compute_delay=lambda attempt, exc: _compute_retry_delay(
                    attempt=attempt,
                    error=exc,
                    policy=r,
                    backoff_fn=backoff_fn,
                ),
                warning_sink=_warnings,
                logger=logger,
                on_error=(hooks.on_error if hooks and hooks.on_error else None),
                on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="acall_llm_structured",
                ),
            ))

        supports_schema = litellm.supports_response_schema(model=current_model)
        _native_schema_failed = False
        if supports_schema:
            schema = _strict_json_schema(response_model.model_json_schema())
            base_kwargs = _prepare_call_kwargs(
                current_model,
                messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            _schema_hash_async = _hashlib.sha256(
                _json.dumps(schema, sort_keys=True).encode()
            ).hexdigest()[:16]
            base_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }

            _pending_repair_message_async: dict[str, str] | None = None

            async def _invoke_native_schema_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                nonlocal _pending_repair_message_async
                if _pending_repair_message_async is not None:
                    base_kwargs["messages"] = list(base_kwargs["messages"]) + [_pending_repair_message_async]
                    _pending_repair_message_async = None
                    logger.info("acall_llm_structured: appended validation repair message for attempt %d", attempt)
                try:
                    async with _rate_limit.aacquire(current_model):
                        response = await litellm.acompletion(**base_kwargs)
                    first_choice = _first_choice_or_empty_error(
                        response,
                        model=current_model,
                        provider="litellm_completion_structured",
                    )
                    raw_content = first_choice.message.content or ""
                    if not raw_content.strip():
                        raise ValueError("Empty content from LLM (native JSON schema structured)")
                    try:
                        parsed = _robust_validate_json(response_model, raw_content)
                    except ValidationError as ve:
                        retry_exc = _StructuredValidationRetry(raw_content, ve)
                        _pending_repair_message_async = _build_validation_repair_message(retry_exc)
                        raise retry_exc from ve
                    usage = _extract_usage(response)
                    cost, cost_source = _parse_cost_result(_compute_cost(response))
                    finish_reason: str = first_choice.finish_reason or "stop"

                    if attempt > 0:
                        logger.info("acall_llm_structured (native schema) succeeded after %d retries", attempt)

                    llm_result = _build_structured_call_result(
                        parsed=parsed,
                        usage=usage,
                        cost=cost,
                        cost_source=cost_source,
                        current_model=current_model,
                        finish_reason=finish_reason,
                        raw_response=response,
                        warnings=_warnings,
                        requested_model=model,
                        attempted_models=models[:model_idx + 1],
                        requested_api_base=api_base,
                        effective_api_base=current_api_base,
                        background_mode=background_mode,
                        routing_policy=routing_policy,
                    )
                    if hooks and hooks.after_call:
                        hooks.after_call(llm_result)
                    if cache is not None and key is not None:
                        await _async_cache_set(cache, key, llm_result)
                    _log_call_event(
                        model=current_model,
                        messages=messages,
                        result=llm_result,
                        latency_s=time.monotonic() - _log_t0,
                        caller="acall_llm_structured",
                        task=task,
                        trace_id=trace_id,
                        prompt_ref=prompt_ref,
                        call_snapshot=call_snapshot,
                        execution_path="native_schema",
                        retry_count=attempt,
                    )
                    return parsed, llm_result
                except Exception as exc:
                    _raise_if_unsupported_gpt5_structured_schema(
                        model=current_model,
                        error=exc,
                        caller="acall_llm_structured",
                    )
                    if _is_schema_error(exc):
                        raise _NativeSchemaFallback(str(exc)) from exc
                    raise

            def _on_native_schema_error(exc: Exception, attempt: int) -> None:
                if isinstance(exc, _NativeSchemaFallback):
                    return
                if hooks and hooks.on_error:
                    hooks.on_error(exc, attempt)

            try:
                return cast(tuple[T, LLMCallResult], await run_async_with_retry(
                    caller="acall_llm_structured",
                    model=current_model,
                    max_retries=r.max_retries,
                    invoke=_invoke_native_schema_attempt,
                    should_retry=lambda exc: (
                        isinstance(exc, _StructuredValidationRetry)
                        or (not isinstance(exc, _NativeSchemaFallback) and _check_retryable(exc, r))
                    ),
                    compute_delay=lambda attempt, exc: _compute_retry_delay(
                        attempt=attempt,
                        error=exc,
                        policy=r,
                        backoff_fn=backoff_fn,
                    ),
                    warning_sink=_warnings,
                    logger=logger,
                    on_error=_on_native_schema_error,
                    on_retry=r.on_retry,
                    maybe_retry_hook=lambda exc, attempt, max_retries: (
                        False if isinstance(exc, _NativeSchemaFallback) else _maybe_retry_with_openrouter_key_rotation(
                            error=exc,
                            attempt=attempt,
                            max_retries=max_retries,
                            current_model=current_model,
                            current_api_base=current_api_base,
                            user_kwargs=public_kwargs,
                            warning_sink=_warnings,
                            on_retry=r.on_retry,
                            caller="acall_llm_structured",
                        )
                    ),
                ))
            except _NativeSchemaFallback as schema_error:
                logger.warning(
                    "Native JSON schema rejected by provider (%s), falling back to instructor: %s",
                    current_model,
                    schema_error,
                )
                _native_schema_failed = True

        if not supports_schema or _native_schema_failed:
            import instructor

            client = instructor.from_litellm(litellm.acompletion)
            base_kwargs = _prepare_call_kwargs(
                current_model,
                messages,
                timeout=timeout,
                num_retries=r.max_retries,
                reasoning_effort=reasoning_effort,
                api_base=current_api_base,
                kwargs=public_kwargs,
                warning_sink=_warnings,
            )
            call_kwargs = {**base_kwargs, "response_model": response_model, "max_retries": 2}

            async def _invoke_instructor_attempt(attempt: int) -> tuple[T, LLMCallResult]:
                parsed, completion_response = await client.chat.completions.create_with_completion(
                    **call_kwargs,
                )

                usage = _extract_usage(completion_response)
                cost, cost_source = _parse_cost_result(_compute_cost(completion_response))
                completion_choice = _first_choice_or_empty_error(
                    completion_response,
                    model=current_model,
                    provider="instructor_completion_structured",
                )
                finish_reason = completion_choice.finish_reason or ""

                if attempt > 0:
                    logger.info("acall_llm_structured succeeded after %d retries", attempt)

                llm_result = _build_structured_call_result(
                    parsed=parsed,
                    usage=usage,
                    cost=cost,
                    cost_source=cost_source,
                    current_model=current_model,
                    finish_reason=finish_reason,
                    raw_response=completion_response,
                    warnings=_warnings,
                    requested_model=model,
                    attempted_models=models[:model_idx + 1],
                    requested_api_base=api_base,
                    effective_api_base=current_api_base,
                    background_mode=background_mode,
                    routing_policy=routing_policy,
                )

                if hooks and hooks.after_call:
                    hooks.after_call(llm_result)
                if cache is not None and key is not None:
                    await _async_cache_set(cache, key, llm_result)
                _log_call_event(
                    model=current_model,
                    messages=messages,
                    result=llm_result,
                    latency_s=time.monotonic() - _log_t0,
                    caller="acall_llm_structured",
                    task=task,
                    trace_id=trace_id,
                    prompt_ref=prompt_ref,
                    call_snapshot=call_snapshot,
                    execution_path="instructor",
                    retry_count=attempt,
                )
                return parsed, llm_result

            return cast(tuple[T, LLMCallResult], await run_async_with_retry(
                caller="acall_llm_structured",
                model=current_model,
                max_retries=r.max_retries,
                invoke=_invoke_instructor_attempt,
                should_retry=lambda exc: _check_retryable(exc, r),
                compute_delay=lambda attempt, exc: _compute_retry_delay(
                    attempt=attempt,
                    error=exc,
                    policy=r,
                    backoff_fn=backoff_fn,
                ),
                warning_sink=_warnings,
                logger=logger,
                on_error=(hooks.on_error if hooks and hooks.on_error else None),
                on_retry=r.on_retry,
                maybe_retry_hook=lambda exc, attempt, max_retries: _maybe_retry_with_openrouter_key_rotation(
                    error=exc,
                    attempt=attempt,
                    max_retries=max_retries,
                    current_model=current_model,
                    current_api_base=current_api_base,
                    user_kwargs=public_kwargs,
                    warning_sink=_warnings,
                    on_retry=r.on_retry,
                    caller="acall_llm_structured",
                ),
            ))

        raise RuntimeError("acall_llm_structured reached unexpected branch without return")

    try:
        return cast(tuple[T, LLMCallResult], await run_async_with_fallback(
            models=models,
            execute_model=_execute_model,
            on_fallback=on_fallback,
            warning_sink=_warnings,
            logger=logger,
        ))
    except Exception as e:
        _log_call_event(
            model=last_model_attempted,
            messages=messages,
            error=e,
            latency_s=time.monotonic() - _log_t0,
            caller="acall_llm_structured",
            task=task,
            trace_id=trace_id,
            prompt_ref=prompt_ref,
            call_snapshot=call_snapshot,
            execution_path="error",
            retry_count=None,
        )
        raise wrap_error(e) from e
