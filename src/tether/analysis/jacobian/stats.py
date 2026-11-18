from collections import namedtuple

import numpy as np

POSITION_BINS = (
    (0, 15),
    (16, 31),
    (32, 63),
    (64, 95),
    (96, 127),
    (128, 191),
    (192, 255),
    (256, 383),
    (384, 512),
)


def position_bin(t):
    for index, (low, high) in enumerate(POSITION_BINS):
        if low <= t <= high:
            return index
    raise ValueError(f"position {t} outside the bins")


def auroc(positive, negative):
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    difference = positive[:, None] - negative[None, :]
    return float(((difference > 0) + 0.5 * (difference == 0)).mean())


_Stratum = namedtuple("_Stratum", "wins positive_clusters negative_clusters")


class StratifiedAuroc:

    def __init__(self, scores, positive, strata, clusters, equal_strata=False):
        keep = ~np.isnan(scores)
        scores, positive, strata, clusters = scores[keep], positive[keep], strata[keep], clusters[keep]
        self.equal_strata = equal_strata
        self.strata = []
        self.n_positive = int(positive.sum())
        self.n_negative = int((~positive).sum())
        for stratum in np.unique(strata):
            inside = strata == stratum
            pos, neg = inside & positive, inside & ~positive
            if pos.any() and neg.any():
                difference = scores[pos][:, None] - scores[neg][None, :]
                self.strata.append(
                    _Stratum((difference > 0) + 0.5 * (difference == 0), clusters[pos], clusters[neg])
                )

    def value(self, cluster_weights=None):
        numerator, denominator, per_stratum = 0.0, 0.0, []
        for stratum in self.strata:
            if cluster_weights is None:
                wins = float(stratum.wins.sum())
                pairs = float(stratum.wins.size)
            else:
                wp, wn = (
                    cluster_weights[stratum.positive_clusters],
                    cluster_weights[stratum.negative_clusters],
                )
                wins = float(wp @ stratum.wins @ wn)
                pairs = float(wp.sum() * wn.sum())
            if pairs > 0:
                numerator += wins
                denominator += pairs
                per_stratum.append(wins / pairs)
        if self.equal_strata:
            return float(np.mean(per_stratum)) if per_stratum else float("nan")
        return numerator / denominator if denominator else float("nan")


class PairedBootstrap:
    def __init__(self, n_clusters, n_resamples=2000, seed=0):
        self.n_clusters = n_clusters
        self.n_resamples = n_resamples
        rng = np.random.default_rng(seed)
        self._weights = [
            np.bincount(rng.integers(0, self.n_clusters, self.n_clusters), minlength=self.n_clusters).astype(
                float
            )
            for _ in range(self.n_resamples)
        ]

    def ci(self, estimator):
        draws = np.array([estimator.value(w) for w in self._weights])
        return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))

    def difference_ci(self, first, second):
        draws = np.array([first.value(w) - second.value(w) for w in self._weights])
        return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def pooled_belief(evidence, t, weights, include_writing_step=True, mask=None):
    end = t + 1 if include_writing_step else t
    step_weights = weights[:end].astype(float)
    if mask is not None:
        step_weights = step_weights * mask[:end]
    total = step_weights.sum()
    if end <= 0 or total <= 0:
        return float("nan")
    return float(step_weights @ evidence[:end] / total)


def schedule_from_kl(bin_means, length):
    bins = sorted(bin_means)
    centres = [(low + high) / 2 for low, high in bins]
    values = np.array([bin_means[b] for b in bins], dtype=float)
    return np.interp(np.arange(length), centres, values / values[0])


def robust_z(value, median, q25, q75):
    return (value - median) / np.maximum(q75 - q25, 1e-6)
