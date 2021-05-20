import math
from collections import OrderedDict
import numpy as np

from represent_input_with_features import represent_input_with_features
import time

def calc_objective_per_iter(w_i, feature2id, histories, relevant_features_list, all_tags, rel_features_for_all_tags_hist):
    """
        Calculate max entropy likelihood for an iterative optimization method
  #      :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """
    start = time.time()
    w_i = np.array(w_i)
    lamda = 2  # TBD
    empirical_counts = np.zeros(feature2id.n_total_features)
    expected_counts = np.zeros(feature2id.n_total_features)
    normalization_term, linear_term = 0, 0
    for history, reps in histories.items():
        relevant_features = relevant_features_list[history]

        """ Linear Term: """
        linear_term += sum(w_i[relevant_features]) * reps

        """ Empirical Counts: """
        empirical_counts[relevant_features] += reps

        """ Normalization Term: """
        inside_log_calc = 0
        expected_counts_temp = np.zeros(feature2id.n_total_features)
        for tag in all_tags:  # TBD: need to find more tags ?
            h = (history[0], history[1], history[2], tag, history[4], history[5])
            f_xi_yTag = rel_features_for_all_tags_hist[h]

            v_mul_f_xi_yTag = sum(w_i[f_xi_yTag])
            exp = np.exp(v_mul_f_xi_yTag)
            inside_log_calc += exp

            # Expected Count:
            p_yTag_xi_v = math.pow(math.e, v_mul_f_xi_yTag)

            expected_counts_temp[f_xi_yTag] += p_yTag_xi_v

        normalization_term += np.log(inside_log_calc) * reps
        expected_counts_temp /= inside_log_calc
        expected_counts += expected_counts_temp * reps

    regularization = -0.5 * lamda * math.pow(np.linalg.norm(w_i), 2)
    regularization_grad = -lamda * w_i
    likelihood = linear_term - normalization_term - regularization  # (1)
    grad = empirical_counts - expected_counts - regularization_grad  # (2)

    end = time.time()
    print("calc_objective_per_iter iteration time:")
    print(end - start)
    return (-1) * likelihood, (-1) * grad
