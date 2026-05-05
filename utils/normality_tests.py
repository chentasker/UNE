"""
Normality tests (Anderson-Darling, D'Agostino-Pearson, Shapiro-Wilk)
"""
import numpy as np
from scipy.stats import anderson, normaltest, shapiro
import os
import glob

def ad_normality_test(data, sig_level=5):
    ad = anderson(data, dist='norm')
    idx = np.where(np.isclose(ad.significance_level, sig_level))[0]
    if len(idx) == 0:
        idx = [int(np.argmin(np.abs(ad.significance_level - sig_level)))]
    idx = idx[0]
    ad_stat = float(ad.statistic)
    ad_crit = float(ad.critical_values[idx])
    ad_normal = ad_stat < ad_crit
    return ad_stat, ad_normal


def dp_normality_test(data, alpha=0.05):
    _, dp_pval = normaltest(data)
    return dp_pval, dp_pval > alpha


def shapiro_normality_test(data, alpha=0.05):
    _, p_val = shapiro(data)
    return p_val, p_val > alpha


def per_dim_normality_tests(ds, n_trials=10, subset_size=500, ad_sig_level=5):
    AD_avg_stat = []
    DP_avg_pval = []
    AD_gaussian_p = []
    DP_gaussian_p = []
    for i in range(n_trials):
        print('Trial number', i)
        random_inds = np.random.choice(ds.shape[0], subset_size, replace=False)
        samples = ds[random_inds]
        n_coords = ds.shape[1]
        ad_stats = []
        dp_pvals = []
        gaussian_count_ad = 0
        gaussian_count_dp = 0
        for j in range(n_coords):
            coord_data = samples[:, j]

            # Anderson-Darling test
            ad_stat, ad_normal = ad_normality_test(coord_data, ad_sig_level)
            gaussian_count_ad += int(ad_normal)
            ad_stats.append(ad_stat)
            
            # D'Agostino-Pearson test
            dp_pval, dp_normal = dp_normality_test(coord_data)
            dp_pvals.append(dp_pval)
            gaussian_count_dp += int(dp_normal)

        AD_avg_stat.append(np.mean(ad_stats))
        DP_avg_pval.append(np.mean(dp_pvals))
        AD_gaussian_p.append(100 * gaussian_count_ad / n_coords)
        DP_gaussian_p.append(100 * gaussian_count_dp / n_coords)
    return (
        np.mean(AD_avg_stat), np.mean(DP_avg_pval), np.mean(AD_gaussian_p), np.mean(DP_gaussian_p)
    ), (
        np.std(AD_avg_stat), np.std(DP_avg_pval), np.std(AD_gaussian_p), np.std(DP_gaussian_p)
    )


def random_projection_normality_tests(
    ds,
    n_projections=50,
    subset_size=500,
    ad_sig_level=5,
    random_state=None,
):
    """
    Test normality by projecting data onto random unit vectors.

    If random_state is set, projection directions and row subsets are reproducible.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    AD_stats = []
    DP_pvals = []
    SW_pvals = []
    AD_passes = []
    DP_passes = []
    SW_passes = []

    projections = rng.standard_normal((n_projections, ds.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    for proj in projections:
        random_inds = rng.choice(ds.shape[0], subset_size, replace=False)
        samples = ds[random_inds]
        projected_data = samples @ proj

        # Anderson-Darling test
        ad_stat, ad_normal = ad_normality_test(projected_data, ad_sig_level)
        
        # D'Agostino-Pearson test
        dp_pval, dp_normal = dp_normality_test(projected_data)

        # Shapiro-Wilk test
        sw_pval, sw_normal = shapiro_normality_test(projected_data)

        AD_stats.append(ad_stat)
        DP_pvals.append(dp_pval)
        SW_pvals.append(sw_pval)
        AD_passes.append(float(ad_normal))
        DP_passes.append(float(dp_normal))
        SW_passes.append(float(sw_normal))

    return AD_stats, DP_pvals, SW_pvals, AD_passes, DP_passes, SW_passes


def summarize_random_projection_results(results):
    """Return dict of mean/std metrics from the tuple returned by random_projection_normality_tests."""
    AD_stats, DP_pvals, SW_pvals, AD_passes, DP_passes, SW_passes = results
    return {
        'ad_stat_mean': float(np.mean(AD_stats)),
        'ad_stat_std': float(np.std(AD_stats)),
        'dp_pval_mean': float(np.mean(DP_pvals)),
        'dp_pval_std': float(np.std(DP_pvals)),
        'sw_pval_mean': float(np.mean(SW_pvals)),
        'sw_pval_std': float(np.std(SW_pvals)),
        'ad_pass_rate_pct': float(np.mean(AD_passes) * 100),
        'ad_pass_rate_std_pct': float(np.std(AD_passes) * 100),
        'dp_pass_rate_pct': float(np.mean(DP_passes) * 100),
        'dp_pass_rate_std_pct': float(np.std(DP_passes) * 100),
        'sw_pass_rate_pct': float(np.mean(SW_passes) * 100),
        'sw_pass_rate_std_pct': float(np.std(SW_passes) * 100),
    }

def main():
    noises_dir = os.path.join('.', 'data', 'celeba_train_noises')
    pattern = os.path.join(noises_dir, '*.npy')
    noises_paths = sorted(glob.glob(pattern))
    model_names = list({
        os.path.basename(path).replace('noises_', '').replace('.npy', '')
        for path in noises_paths
    })
    model_names = [
        'ocB16', 'ocL14',
    ]

    all_noises = [
        np.load(os.path.join(noises_dir, f'noises_{model_name}.npy'))
        for model_name in model_names
    ]

    ad_sig_level = 5

    for noises, model_name in zip(all_noises, model_names):
        results = random_projection_normality_tests(
            noises,
            n_projections=5000,
            subset_size=250,
            ad_sig_level=ad_sig_level,
        )
        summ = summarize_random_projection_results(results)
        print(f"Model: {model_name}")
        print(f"  AD Stat: Mean={summ['ad_stat_mean']:.4f}, Std={summ['ad_stat_std']:.2f}")
        print(f"  AD Gaussian %: Mean={summ['ad_pass_rate_pct']:.2f}, Std={summ['ad_pass_rate_std_pct']:.2f}")
        print(f"  DP P-Value: Mean={summ['dp_pval_mean']:.4f}, Std={summ['dp_pval_std']:.4f}")
        print(f"  DP Gaussian %: Mean={summ['dp_pass_rate_pct']:.2f}, Std={summ['dp_pass_rate_std_pct']:.2f}")
        print(f"  SW P-Value: Mean={summ['sw_pval_mean']:.4f}, Std={summ['sw_pval_std']:.4f}")
        print(f"  SW Gaussian %: Mean={summ['sw_pass_rate_pct']:.2f}, Std={summ['sw_pass_rate_std_pct']:.2f}")


if __name__ == '__main__':
    main()
