from .compare_concordance.compare_concordance  import estimate_concordance, _compute_vardiff
import numpy as np
from .utils import cal_pvalue



def compareC(time, status, y_ref, y_new, n_resamples=None, seed=None):

    concordance_y = estimate_concordance(time, status, y_ref)
    concordance_z = estimate_concordance(time, status, y_new)

    diff = concordance_y - concordance_z

    vardiff, var_y, var_z, cov = _compute_vardiff(time, status, y_ref, y_new)

    se = np.sqrt(vardiff)  # estimate standard error

    # estimate std from bootstrap , so se is std here
    if n_resamples is not None:
        if n_resamples > 1:
            if seed is not None:
                np.random.seed(seed)

            samples = []
            for _ in range(n_resamples):
                idx = np.random.choice(len(time), size=len(time), replace=True)
                concordance_x_bootstrap = estimate_concordance(
                    time[idx], status[idx], y_ref[idx]
                )
                concordance_y_bootstrap = estimate_concordance(
                    time[idx], status[idx], y_new[idx]
                )
                samples.append(concordance_x_bootstrap - concordance_y_bootstrap)
            std = np.sqrt(np.var(samples))

            # LET SE = STD
            se = std

    pval = cal_pvalue(mean=diff, se=se)

    c_diff_UCI = diff + 1.96 * se
    c_diff_LCI = diff - 1.96 * se
    return {
        "c_diff": diff,
        "c_diff_UCI": c_diff_UCI,
        "c_diff_LCI": c_diff_LCI,
        "c_diff P-value": pval,
        "c_diff_SE": se,
    }