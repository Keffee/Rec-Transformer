def my_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    if ground_truth in solution_str:
        return 1.0
    else:
        return 0.0