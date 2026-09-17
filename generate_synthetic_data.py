"""Create a reproducible synthetic dataset for the railway-risk demo.

This data is illustrative only. It is generated from assumed relationships and
must not be used for real railway maintenance or safety decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "railwaydata_synthetic.csv"
ASSET_TYPES = {
    "Track and rail": 0.10,
    "Turnout / points": 0.25,
    "Signalling equipment": 0.15,
    "Level crossing": 0.20,
    "Overhead line / electrification": 0.16,
    "Traction power substation": 0.18,
    "Locomotive": 0.22,
    "Passenger coach": 0.12,
    "Freight wagon": 0.14,
    "Bridge": 0.17,
    "Tunnel": 0.13,
    "Platform or station equipment": 0.08,
    "Telecommunications equipment": 0.11,
}


def create_synthetic_dataset(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate fictional records with risk labels based on assumed trends."""
    rng = np.random.default_rng(seed)
    asset_types = list(ASSET_TYPES)
    asset_type = rng.choice(asset_types, size=rows)
    age = rng.integers(1, 41, size=rows)
    failures = rng.poisson(0.4 + age / 18).clip(0, 9)
    days_since_maintenance = rng.integers(5, 366, size=rows)
    usage = rng.integers(20, 101, size=rows)
    condition = np.clip(
        96 - age * 1.05 - failures * 5 - days_since_maintenance * 0.06
        + rng.normal(0, 9, size=rows), 0, 100,
    ).round().astype(int)
    criticality = rng.choice([1, 2, 3, 4, 5], size=rows, p=[0.12, 0.20, 0.32, 0.23, 0.13])

    # Assumed, fictional relationship used only to create demo labels.
    score = (-5.0 + age * 0.06 + failures * 0.48 + days_since_maintenance * 0.008
             + usage * 0.012 - condition * 0.045 + criticality * 0.28
             + np.array([ASSET_TYPES[item] for item in asset_type])
             + rng.normal(0, 0.55, size=rows))
    risk = rng.binomial(1, 1 / (1 + np.exp(-score)))

    return pd.DataFrame({
        "asset_id": [f"SYN{i:05d}" for i in range(1, rows + 1)],
        "asset_type": asset_type,
        "asset_age": age,
        "previous_failures": failures,
        "days_since_maintenance": days_since_maintenance,
        "usage": usage,
        "condition": condition,
        "criticality": criticality,
        "risk": risk,
    })


if __name__ == "__main__":
    dataset = create_synthetic_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {len(dataset)} synthetic rows at: {OUTPUT_PATH}")
    print("High-risk rate:", f"{dataset['risk'].mean():.1%}")
