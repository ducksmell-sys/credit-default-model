"""Models: an interpretable logistic regression (the candidate), a gradient boosting challenger, and a
single-variable benchmark. All predict the probability of default (PD)."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_data import FEATURE_GROUPS, SEED

BENCHMARK_FEATURES = FEATURE_GROUPS["Recent status (PAY_1 dummies)"]


def fit_logistic(X, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return model.fit(X, y)


def fit_gradient_boosting(X, y):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_depth=4, min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=SEED,
    )
    return model.fit(X, y)


def fit_benchmark(X, y):
    """'Look only at last month's repayment status': the simple rule a complex model has to beat."""
    return fit_logistic(X[BENCHMARK_FEATURES], y)


def predict_pd(model, X):
    return model.predict_proba(X)[:, 1]


def standardized_coefficients(model, columns):
    """Coefficients of the logistic regression on standardized inputs (comparable across features)."""
    return dict(zip(columns, model.named_steps["logisticregression"].coef_[0]))
