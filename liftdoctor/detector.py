"""Detection logic. v1: statistical baseline learning. v2 will be ML."""
import pandas as pd


class TrendDetector:
    """Learns a channel's normal behavior from its first N hours,
    then flags deviations and abnormal trends."""

    def __init__(self, train_hours=6, z_limit=3.0, slope_limit=3.0):
        self.train_minutes = train_hours * 60
        self.z_limit = z_limit
        self.slope_limit = slope_limit

    def score(self, series: pd.Series) -> pd.DataFrame:
        base = series.iloc[:self.train_minutes]
        mu, sigma = base.mean(), base.std()

        z = (series - mu) / sigma                       # how far from normal
        slope = series.diff().rolling(30).mean()         # trend per minute
        slope_z = slope / slope.rolling(60).std().iloc[:self.train_minutes].mean()

        return pd.DataFrame({
            "value": series,
            "z_score": z,
            "anomaly": (z.abs() > self.z_limit) | (slope_z.abs() > self.slope_limit),
        })