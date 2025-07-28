import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr

df = pd.read_csv("raw_metrics.csv")
df = df[df["Noise Intensity"] != 0]

friedman_results = []
spearman_results = []

for noise_type in df["Noise Type"].unique():
    for metric in df["Metric"].unique():
        subset = df[(df["Noise Type"] == noise_type) & (df["Metric"] == metric)]

        intensities = sorted(subset["Noise Intensity"].unique())
        expected_n = len(intensities)
        seed_counts = subset.groupby("Seed")["Noise Intensity"].nunique()
        complete_seeds = seed_counts[seed_counts == expected_n].index
        filtered = subset[subset["Seed"].isin(complete_seeds)]
        wide = (
            filtered
            .pivot(index="Seed", columns="Noise Intensity", values="Value")
            .reindex(columns=intensities)
        )

        res = friedmanchisquare(*[wide[col].values for col in wide.columns])

        statistic = res.statistic.round(4)
        pvalue = res.pvalue.round(6)

        friedman_results.append({
            "Noise Type": noise_type,
            "Metric": metric,
            "Friedman Statistic": statistic,
            "p-value": pvalue
        })

for noise_type in df["Noise Type"].unique():
    for metric in df["Metric"].unique():

        subset = df[
            (df["Noise Type"] == noise_type) &
            (df["Metric"] == metric)
        ]

        agg = (
            subset
            .groupby("Noise Intensity")["Value"]
            .mean()
            .sort_index()
        )

        intensities = agg.index.tolist()
        values = agg.values.tolist()
        rho, p = spearmanr(intensities, values)

        rho = rho.round(4)
        p = p.round(6)

        spearman_results.append({
            "Noise Type": noise_type,
            "Metric": metric,
            "Spearman Correlation (rho)": rho,
            "p-value": p,
            "Trend": "decreasing" if rho < 0 else "increasing" if rho > 0 else "no trend"
        })

pd.DataFrame(friedman_results).to_csv("friedman_results.csv", index=False)
pd.DataFrame(spearman_results).to_csv("spearman_results.csv", index=False)