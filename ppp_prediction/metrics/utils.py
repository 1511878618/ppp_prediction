import numpy as np
def find_best_cutoff(fpr, tpr, thresholds):
    diff = tpr - fpr
    Youden_index = np.argmax(diff)
    optimal_threshold = thresholds[Youden_index]
    optimal_FPR, optimal_TPR = fpr[Youden_index], tpr[Youden_index]
    return optimal_threshold, optimal_FPR, optimal_TPR

