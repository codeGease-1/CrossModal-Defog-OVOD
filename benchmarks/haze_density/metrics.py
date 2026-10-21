"""Scale-independent metrics; undefined correlations remain None (N/A)."""
import numpy as np


def finite_array(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('Metrics require nonempty finite arrays')
    return x


def average_ranks(values):
    x = finite_array(values).ravel()
    order = np.argsort(x, kind='stable')
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def correlation(x, y, rank=False):
    x, y = finite_array(x).ravel(), finite_array(y).ravel()
    if x.shape != y.shape:
        raise ValueError('Metric shapes differ')
    if x.size < 2:
        return None
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def ranking_metrics(truth, prediction, tie_tolerance=1e-6):
    truth, prediction = finite_array(truth).ravel(), finite_array(prediction).ravel()
    if truth.shape != prediction.shape or not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('Invalid ranking inputs or tolerance')
    correct = ties = reversed_count = pairs = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if abs(truth[i] - truth[j]) <= 1e-12:
                continue
            pairs += 1
            signed = (prediction[i] - prediction[j]) * np.sign(truth[i] - truth[j])
            if abs(signed) <= tie_tolerance:
                ties += 1
            elif signed > 0:
                correct += 1
            else:
                reversed_count += 1
    order = np.argsort(truth, kind='stable')
    adjacent = [(prediction[b] - prediction[a]) for a, b in zip(order[:-1], order[1:])
                if truth[b] - truth[a] > 1e-12]
    rho = correlation(truth, prediction, rank=True)
    return {
        'pairwise_accuracy': correct / pairs if pairs else None,
        'tie_rate': ties / pairs if pairs else None,
        'pairwise_reversal_rate': reversed_count / pairs if pairs else None,
        'comparable_pairs': pairs,
        'spearman': rho,
        'spearman_status': 'PASS' if rho is not None else 'N/A',
        'spearman_reason': None if rho is not None else 'constant vector or fewer than two values',
        'monotonicity_violation_rate': float(np.mean(np.asarray(adjacent) < -tie_tolerance)) if adjacent else None,
        'adjacent_tie_rate': float(np.mean(np.abs(adjacent) <= tie_tolerance)) if adjacent else None,
        'adjacent_pairs': len(adjacent),
    }


def region_means(image, grid=4):
    image = finite_array(image)
    if image.ndim != 2 or grid < 1 or min(image.shape) < grid:
        raise ValueError('Expected 2D image at least as large as grid')
    return np.array([block.mean() for rows in np.array_split(image, grid, axis=0)
                     for block in np.array_split(rows, grid, axis=1)])


def spatial_metrics(transmission, prediction, grid=4):
    t, p = finite_array(transmission), finite_array(prediction)
    if t.shape != p.shape or t.ndim != 2:
        raise ValueError('Spatial maps must have identical H,W shapes')
    strength = 1 - t
    if np.ptp(strength) <= 1e-12:
        return {'status': 'N/A', 'reason': 'spatially constant transmission',
                'region_spearman': None, 'region_pairwise_accuracy': None,
                'dense_minus_light_response': None}
    gt_regions, pred_regions = region_means(strength, grid), region_means(p, grid)
    ranking = ranking_metrics(gt_regions, pred_regions)
    low, high = np.quantile(strength, [.25, .75])
    contrast = float(p[strength >= high].mean() - p[strength <= low].mean()) if high > low else None
    return {'status': 'PASS', 'reason': None, 'region_spearman': ranking['spearman'],
            'region_spearman_status': ranking['spearman_status'],
            'region_pairwise_accuracy': ranking['pairwise_accuracy'],
            'region_tie_rate': ranking['tie_rate'], 'dense_minus_light_response': contrast,
            'region_truth': gt_regions.tolist(), 'region_prediction': pred_regions.tolist()}


def rank_residual(transmission, prediction, grid=4):
    truth = region_means(1 - finite_array(transmission), grid)
    pred = region_means(prediction, grid)
    if np.ptp(truth) <= 1e-12:
        return None
    # A dimensionless spatial diagnostic, never a physical transmission error.
    return ((average_ranks(pred) - average_ranks(truth)) / max(1, len(truth) - 1)).reshape(grid, grid)


def response_summary(prediction, threshold=.7):
    p = finite_array(prediction)
    return {'response_mean': float(p.mean()), 'response_p50': float(np.quantile(p, .5)),
            'response_p95': float(np.quantile(p, .95)),
            'area_above_reference_threshold': float((p >= threshold).mean()),
            'reference_threshold': threshold}


def confound_regions(clear_rgb, prediction):
    rgb, p = finite_array(clear_rgb), finite_array(prediction)
    gray = rgb @ np.array([.299, .587, .114])
    if gray.shape != p.shape:
        raise ValueError('Mismatched diagnostic shapes')
    dy, dx = np.gradient(gray)
    masks = {'dark': gray < .2, 'bright': gray > .8,
             'low_texture': np.hypot(dx, dy) < .01}
    return {name: {'pixels': int(mask.sum()),
                   'mean_response': float(p[mask].mean()) if mask.any() else None,
                   'status': 'PASS' if mask.any() else 'N/A'} for name, mask in masks.items()}
