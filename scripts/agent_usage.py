"""Token accounting for a single OpenAI response using published USD list prices."""

from decimal import Decimal


def _token_count(value):
    return value if type(value) is int and value >= 0 else None


def agent_usage(response):
    usage = getattr(response, "usage", None)
    input_tokens = _token_count(getattr(usage, "input_tokens", None))
    output_tokens = _token_count(getattr(usage, "output_tokens", None))
    metrics = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": None}
    if (input_tokens is None or output_tokens is None
            or getattr(response, "model", None) != "gpt-6.1-sol"
            or getattr(response, "service_tier", None) not in (None, "default")):
        return metrics

    details = getattr(usage, "input_tokens_details", None)
    reads = _token_count(getattr(details, "cached_tokens", None))
    writes = _token_count(getattr(details, "cache_write_tokens", None))
    if reads is None or writes is None or reads + writes > input_tokens:
        return metrics
    # Our request uses the default 30-minute lifetime. Newer provider responses
    # also report the actual write counts by TTL, including 12-hour writes.
    breakdown = getattr(details, "cache_write_tokens_details", None)
    short_writes, long_writes = writes, 0
    if breakdown is not None:
        if hasattr(breakdown, "model_dump"):
            breakdown = breakdown.model_dump()
        if not isinstance(breakdown, dict):
            return metrics
        short_writes = _token_count(breakdown.get("ttl_30m_tokens"))
        long_writes = _token_count(breakdown.get("ttl_12h_tokens"))
        if short_writes is None or long_writes is None or short_writes + long_writes != writes:
            return metrics

    # Standard global GPT-6.1 Sol rates in USD per million text tokens.
    # https://developers.openai.com/api/docs/pricing
    uncached = input_tokens - reads - writes
    input_charge = (2 * uncached + Decimal("0.10") * reads
                    + Decimal("2.50") * short_writes + 5 * long_writes)
    output_charge = 10 * output_tokens  # Provider output already includes reasoning tokens.
    if input_tokens > 272_000:
        input_charge *= 2
        output_charge *= Decimal("1.5")
    metrics["cost_usd"] = format((input_charge + output_charge) / 1_000_000, ".7f")
    return metrics
