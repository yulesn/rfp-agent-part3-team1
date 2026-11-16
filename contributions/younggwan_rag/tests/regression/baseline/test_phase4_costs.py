import unittest
from rfp_rag.generation.costs import estimate_usage_cost


class CostTests(unittest.TestCase):
    rates = {'model':'example','input_per_million':1.,'cached_input_per_million':.1,'output_per_million':4.}

    def test_cached_tokens_not_double_charged(self):
        value = estimate_usage_cost({'input_tokens':1000000,'cached_input_tokens':500000,'output_tokens':1000000}, self.rates)
        self.assertAlmostEqual(value['estimated_usd_for_reported_tokens'],4.55)
        self.assertFalse(value['is_invoice'])

    def test_current_local_cache_hit_zero_cost(self):
        value=estimate_usage_cost({'input_tokens':0,'output_tokens':0,'api_calls':0},self.rates)
        self.assertEqual(value['estimated_usd_for_reported_tokens'],0)

    def test_unknown_failed_attempt_usage_is_explicitly_incomplete(self):
        value=estimate_usage_cost({'api_calls':1,'usage_reported':False},self.rates)
        self.assertFalse(value['complete_usage_available'])

    def test_client_stats_and_mock_label_are_supported(self):
        value=estimate_usage_cost({'reported_input_tokens':100,'reported_output_tokens':50,'mock_calls':1,'live_api_calls':0},self.rates)
        self.assertEqual(value['mock_calls'],1)
        self.assertEqual(value['live_api_calls'],0)

    def test_invalid_rates_and_tokens_rejected(self):
        with self.assertRaises(ValueError): estimate_usage_cost({'input_tokens':1,'cached_input_tokens':2},self.rates)
        with self.assertRaises(ValueError): estimate_usage_cost({},dict(self.rates,input_per_million=float('nan')))
        with self.assertRaises(ValueError): estimate_usage_cost({'input_tokens':True},self.rates)
