"""Optional token-based price estimate; never an API invoice or budget gate."""
import math


def estimate_usage_cost(usage, pricing):
    """Accept one response usage or the guarded client's cumulative report.

    Unknown usage on failed attempts makes the estimate incomplete. Local cache
    hits report zero current tokens in llm.py. Mock usage is always labelled.
    """
    if not isinstance(usage, dict) or not isinstance(pricing, dict):
        raise ValueError('usage and pricing must be dictionaries')
    rates = []
    for key in ['input_per_million', 'cached_input_per_million', 'output_per_million']:
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid configured unit price: ' + key)
        rates.append(value)
    def count(primary, alternative):
        value = usage.get(primary, usage.get(alternative, 0))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Invalid reported token usage: ' + primary)
        return value
    inputs = count('reported_input_tokens', 'input_tokens')
    cached = count('reported_cached_input_tokens', 'cached_input_tokens')
    outputs = count('reported_output_tokens', 'output_tokens')
    if cached > inputs:
        raise ValueError('Cached input tokens cannot exceed total input tokens')
    calls = count('api_calls', 'api_calls')
    unknown = count('usage_unknown_attempts', 'usage_unknown_attempts')
    if calls and 'usage_reported' in usage and not usage['usage_reported']:
        unknown = max(unknown, 1)
    estimate = ((inputs - cached) * rates[0] + cached * rates[1] + outputs * rates[2]) / 1_000_000
    return {'estimated_usd_for_reported_tokens': round(estimate, 8),
            'input_tokens': inputs, 'cached_input_tokens': cached, 'output_tokens': outputs,
            'complete_usage_available': unknown == 0,
            'unknown_usage_attempts': unknown,
            'live_api_calls': usage.get('live_api_calls'), 'mock_calls': usage.get('mock_calls'),
            'pricing_model': pricing.get('model'), 'pricing_as_of': pricing.get('as_of'),
            'pricing_source': pricing.get('source'),
            'is_invoice': False,
            'scope': 'Configured token-rate estimate. Excludes taxes, regional premiums, tools and unreported failed-attempt usage; mock token estimates are hypothetical.'}
