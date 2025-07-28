import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

df = pd.read_csv("poisson_gaussian_raw_metrics.csv")

df = df[df["Gaussian Level"] != 0]

friedman_results = []
spearman_results = []

for brightness in sorted(df["Brightness"].unique()):
    for metric in df["Metric"].unique():
        subset = df[
            (df["Brightness"] == brightness) &
            (df["Metric"] == metric)
        ]

        gaussian_levels = sorted(subset["Gaussian Level"].unique())
        expected_n = len(gaussian_levels)

        seed_counts = subset.groupby("Seed")["Gaussian Level"].nunique()
        complete_seeds = seed_counts[seed_counts == expected_n].index

        filtered = subset[subset["Seed"].isin(complete_seeds)]

        wide = (
            filtered
            .pivot(index="Seed", columns="Gaussian Level", values="Value")
            .reindex(columns=gaussian_levels)
        )

        if len(wide) < 2:
            continue

        stat, p = friedmanchisquare(*[wide[col].values for col in wide.columns])

        stat = stat.round(4)
        p = p.round(6)

        friedman_results.append({
            "Brightness": brightness,
            "Metric": metric,
            "Friedman Statistic": stat,
            "p-value": p
        })

for brightness in sorted(df["Brightness"].unique()):
    for metric in df["Metric"].unique():
        subset = df[
            (df["Brightness"] == brightness) &
            (df["Metric"] == metric)
        ]

        if subset.empty:
            continue

        agg = (
            subset
            .groupby("Gaussian Level")["Value"]
            .mean()
            .sort_index()
        )

        gaussian_levels = agg.index.tolist()
        values = agg.values.tolist()
        rho, p = spearmanr(gaussian_levels, values)

        rho = rho.round(4)
        p = p.round(6)

        spearman_results.append({
            "Brightness": brightness,
            "Metric": metric,
            "Spearman Correlation (rho)": rho,
            "p-value": p,
            "Trend": "decreasing" if rho < 0 else "increasing" if rho > 0 else "no trend"
        })

pd.DataFrame(friedman_results).to_csv("friedman_results.csv", index=False)
pd.DataFrame(spearman_results).to_csv("spearman_results.csv", index=False)