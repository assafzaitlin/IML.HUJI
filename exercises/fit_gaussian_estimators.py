from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
pio.templates.default = "simple_white"

EXPECTED_VALUE = 10
VARIANCE = 1
NUM_OF_SAMPLES = 1000
SAMPLES_DIFF = 10
DIFF_COL = 'Difference between actual and estimated variance'
NUM_OF_SAMPLES_COL = 'Sample size'
SAMPLE_VAL_COL = 'Sample value'
PDF_COL = 'Probability density function value'
TITLE = 'Difference between actual and estimated variance by num of samples'
PDF_TITLE = 'Sample value to its probability density function value'
MV_MEAN = np.array([0, 0, 4, 0])
MV_COV = np.array([
    [1, 0.2, 0, 0.5],
    [0.2, 2, 0, 0],
    [0, 0, 1, 0],
    [0.5, 0, 0, 1]
])
MV_NUM_OF_SAMPLES = 1000
MV_HEATMAP_TITLE = "Log Likelihood Heatmap based on f1, f3 values"
F1_COL = 'f1'
F3_COL = 'f3'
LOG_LIKELIHOOD_COL = 'Log Likelihood'

def test_univariate_gaussian():
    # Question 1
    uni_variate_gaussian = UnivariateGaussian()
    samples = np.random.normal(EXPECTED_VALUE, VARIANCE, NUM_OF_SAMPLES)
    uni_variate_gaussian = uni_variate_gaussian.fit(samples)
    print(f"({uni_variate_gaussian.mu_}, {uni_variate_gaussian.var_})")

    # Question 2
    num_of_samples = SAMPLES_DIFF
    results = []
    for i in range(10, 1010, 10):
        partial_samples = samples[:i]
        uni_variate_gaussian.fit(partial_samples)
        results.append((abs(uni_variate_gaussian.mu_ - EXPECTED_VALUE),
                        partial_samples.size))
        num_of_samples += SAMPLES_DIFF
    df = pd.DataFrame(results, columns=[DIFF_COL, NUM_OF_SAMPLES_COL])
    px.bar(df, y=DIFF_COL, x=NUM_OF_SAMPLES_COL, title=TITLE).show()

    # Question 3
    pdf = uni_variate_gaussian.pdf(samples)
    df = pd.DataFrame(zip(samples, pdf), columns=[SAMPLE_VAL_COL, PDF_COL])
    px.scatter(df, x=SAMPLE_VAL_COL, y=PDF_COL, title=PDF_TITLE).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(MV_MEAN, MV_COV, MV_NUM_OF_SAMPLES)
    mv_gaussian = MultivariateGaussian()
    mv_gaussian.fit(samples)
    print(mv_gaussian.mu_)
    print(mv_gaussian.cov_)
    # Question 5 - Likelihood evaluation
    results = []
    for f1 in np.linspace(-10, 10, 200):
        for f3 in np.linspace(-10, 10, 200):
            mu = np.array([f1, 0, f3, 0])
            log_likelihood = MultivariateGaussian.log_likelihood(mu, MV_COV,
                                                                 samples)
            results.append((f1, f3, log_likelihood))
    df = pd.DataFrame(results, columns=[F1_COL, F3_COL, LOG_LIKELIHOOD_COL])
    px.density_heatmap(df, x=F3_COL, y=F1_COL, z=LOG_LIKELIHOOD_COL,
                       histfunc='avg', range_x=[-10, 10], range_y=[-10, 10],
                       title=MV_HEATMAP_TITLE).show()
    # Question 6 - Maximum likelihood
    argmax = df.iloc[df[LOG_LIKELIHOOD_COL].idxmax()]
    max_f1, max_f3, likelihood = argmax[F1_COL], argmax[F3_COL], argmax[LOG_LIKELIHOOD_COL]
    likelihood = format(likelihood, '.3f')
    print(f"maximal log likelihood values are: (f1: {max_f1}, f3: {max_f3}, log likelihood: {likelihood})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
