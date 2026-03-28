"""
Shared dummy dataset and reward for wp_conciseness.py and danp_conciseness_lm.py.
Same as wp_conciseness.py (lines 42–55).
"""


WP_DUMMY_DATASET = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]


def compute_reward(generated_text, target_text):
    """
    Dummy reward: negative absolute difference in length (for demonstration).
    Replace with a metric that evaluates correctness or quality if needed.
    """
    return -abs(len(generated_text) - len(target_text))


def expand_wp_dummy_dataset(n_train: int, n_eval: int):
    """Repeat the two (prompt, target) pairs cyclically to fill n_train + n_eval samples."""
    pairs = WP_DUMMY_DATASET
    n_total = n_train + n_eval
    reps = (n_total + len(pairs) - 1) // len(pairs)
    expanded = (pairs * reps)[:n_total]
    train_data = expanded[:n_train]
    eval_data = expanded[n_train : n_train + n_eval]
    return train_data, eval_data
