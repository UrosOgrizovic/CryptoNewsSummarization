import unittest
import metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_rouge1(self):
        result = metrics.calculate_rouge(
            ["Bitcoin Price Update: Will China Lead us Down?"],
            ["Bitcoin Price Watch: Gox, EOS, Litecoin, IOTA, Stellar Lumens, NEM"],
            rouge_keys=["rouge1"],
        )
        self.assertEqual({"rouge1": 22.2222}, result)

    def test_calculate_rouge2(self):
        result = metrics.calculate_rouge(
            ["Bitcoin Price Update: Will China Lead us Down?"],
            ["Bitcoin Price Watch: Gox, EOS, Litecoin, IOTA, Stellar Lumens, NEM"],
            rouge_keys=["rouge2"],
        )
        self.assertEqual({"rouge2": 12.5}, result)

    def test_calculate_rougeL(self):
        # beta = 1, LCS = 2, P = 2/4, R = 2/3, result = (1+beta)*P*R/(R+P*beta^2)
        result = metrics.calculate_rouge(
            ["first some string"],
            ["and then some string"],
            rouge_keys=["rougeL"],
        )
        self.assertEqual({"rougeL": 57.1429}, result)

    def test_calculate_rougeLsum(self):
        result = metrics.calculate_rouge(
            ["first some string"],
            ["and then some string"],
            rouge_keys=["rougeLsum"],
        )
        self.assertEqual({"rougeLsum": 57.1429}, result)

    def test_calculate_bleu(self):
        """sentences must be at least 4 words long because of MAX_NGRAM_ORDER in
        sacrebleu/metrics/bleu.py. Otherwise, the result is 0"""
        pred = "first string a a a"
        ref = ["second string b b b"]
        result = metrics.calculate_bleu(pred, ref)
        assert {"bleu": round(10.6822/100, 4)} == result
