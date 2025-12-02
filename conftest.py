import pytest
from scipy.stats import weibull_min
import numpy as np

@pytest.fixture(scope="session")
def test_data():
    """Generate consistent test data across all tests"""
    np.random.seed(42)
    return weibull_min.rvs(c=10, scale=100, size=30)