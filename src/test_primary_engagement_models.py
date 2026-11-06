"""Independent checks of distribution math and likelihood/SE conventions."""
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import nbinom
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from run_primary_engagement_models import Fit, moments, cdf, fit_one, fit_hurdle


class EngagementModelChecks(unittest.TestCase):
    def fixture(self, name, params):
        x = np.array([[1., 0.], [1., 1.], [1., 0.]])
        k = len(params)
        return Fit(name, np.array(params), np.eye(k), np.eye(k), np.zeros((3, k)),
                   -10., True, "", "[]", [], [], False, 0., True, 2, x, np.array([0., 1., 5.]))

    def test_hurdle_moments_and_probabilities(self):
        f = self.fixture("Hurdle_Logit_ZTNB2", [-.3, .7, 1.5, -.4, 2.])
        m = TruncatedLFNegativeBinomialP([1, 2, 4], f.x, p=2, truncation=0)
        mean, variance, pzero, positive_mean, mu = moments(f)
        native_mean = m.predict(f.params[2:], exog=f.x, which="mean")
        native_var = m.predict(f.params[2:], exog=f.x, which="var")
        assert_allclose(positive_mean, native_mean)
        assert_allclose(variance, (1 - pzero) * native_var + pzero * (1 - pzero) * native_mean**2)
        assert_allclose(cdf(f, 0), pzero)
        assert_allclose(cdf(f, -1), 0)
        assert_allclose(cdf(f, 100000), 1)
        assert_allclose(mean, (1 - pzero) * native_mean)

    def test_zinb_moments_match_native(self):
        f = self.fixture("ZINB2", [-.3, .7, 1.5, -.4, 2.])
        m = ZeroInflatedNegativeBinomialP(f.y, f.x, exog_infl=f.x, p=2)
        mean, variance, pzero, _, _ = moments(f)
        assert_allclose(mean, m.predict(f.params, which="mean"))
        assert_allclose(pzero, m.predict(f.params, which="prob-zero"))
        assert_allclose(cdf(f, 5), m.predict(f.params, which="prob", y_values=np.arange(6)).sum(axis=1))

    def test_unadjusted_nb_group_means_and_hc0(self):
        y = np.array([0, 1, 2, 3, 25, 0, 1, 1, 3, 12.])
        x = pd.DataFrame({"Intercept": np.ones(10), "fake_vs_real": np.repeat([0., 1.], 5)})
        f = fit_one(y, x, "NB2")
        expected_beta = np.log(y[5:].mean() / y[:5].mean())
        expected_var = y[5:].var() / (5 * y[5:].mean()**2) + y[:5].var() / (5 * y[:5].mean()**2)
        assert_allclose(f.params[1], expected_beta, atol=1e-5)
        assert_allclose(f.cov[1, 1], expected_var, rtol=1e-4)
        assert_allclose(f.aic, -2 * f.ll + 2 * 3)
        assert_allclose(f.bic, -2 * f.ll + 3 * np.log(10))

    def test_joint_hurdle_likelihood(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame({"Intercept": np.ones(600), "fake_vs_real": np.repeat([0., 1.], 300)})
        y = rng.negative_binomial(2, .2, 600).astype(float)
        y[rng.random(600) < .4] = 0
        f = fit_hurdle(y, x)
        _, _, zero, _, mu = moments(f)
        a = f.params[-1]
        lp0 = -np.log1p(a * mu) / a
        positive = y > 0
        direct_ll = np.log(zero[~positive]).sum() + (
            np.log1p(-zero[positive]) + nbinom.logpmf(y[positive], 1/a, 1/(1+a*mu[positive]))
            - np.log(-np.expm1(lp0[positive]))).sum()
        assert_allclose(f.ll, direct_ll, atol=1e-6)
        assert_allclose(zero.mean(), (y == 0).mean(), atol=1e-6)
        self.assertEqual(f.k, 5)
        self.assertTrue(f.stable)


if __name__ == "__main__":
    unittest.main()
