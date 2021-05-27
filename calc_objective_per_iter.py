import math
import numpy as np
import time


def calc_objective_per_iter(w_i, feature2id, histories, relevant_features_list, all_tags,
                            rel_features_for_all_tags_hist, iteration_count, lamda):
    """
        Calculate max entropy likelihood for an iterative optimization method
  #      :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """
    iteration_count.n += 1
    print("current iteration:")
    print(iteration_count.n)
    print("Weights for current iteration:")
    print(w_i)
    start = time.time()
    w_i = np.array(w_i)
    # lamda = 2  # ToDo
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
            h = (history[0], history[1], history[2], tag, history[4], history[5], history[6])
            tag_rel_features = rel_features_for_all_tags_hist[h]

            tag_weighted_features = sum(w_i[tag_rel_features])
            exp = np.exp(tag_weighted_features)
            inside_log_calc += exp

            # Expected Count:
            expected_counts_temp[tag_rel_features] += exp  # f(xi,yTag)*p(yTag|xi;v)

        normalization_term += np.log(inside_log_calc) * reps
        expected_counts_temp = expected_counts_temp * reps
        expected_counts_temp /= inside_log_calc
        expected_counts += expected_counts_temp

    regularization = 0.5 * lamda * math.pow(np.linalg.norm(w_i), 2)
    regularization_grad = lamda * w_i
    likelihood = linear_term - normalization_term - regularization  # (1)
    grad = empirical_counts - expected_counts - regularization_grad  # (2)

    end = time.time()
    print("calc_objective_per_iter iteration time:")
    print(end - start)
    return (-1) * likelihood, (-1) * grad
