"""The Synthetic Table dataset: a customer table with numbers and categories whose target
depends on interactions between categories (embeddings and attention help)."""
from __future__ import annotations

# (column, number of categories or None for numbers), in the generated column order
SCHEMA = (("age", None), ("income", None), ("tenure", None), ("city", 6), ("plan", 3),
          ("channel", 4))

SYNTHETIC_TABLE_CODE = r'''
def synthetic_table(n: int, noise: float, seed: int, task: str):
    """(frame with age, income, tenure, city, plan, channel; target array)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    cities = ["paris", "lyon", "nice", "lille", "nantes", "brest"]
    plans, channels = ["basic", "pro", "team"], ["web", "store", "phone", "partner"]

    def pick(values: list) -> np.ndarray:
        """Every category appears (the first rows), the rest are random."""
        return np.array(list(values) + list(rng.choice(values, max(0, n - len(values)))))[:n]

    city, plan, channel = pick(cities), pick(plans), pick(channels)
    age = rng.uniform(18, 75, n)
    income = rng.lognormal(10.3, 0.4, n)
    tenure = rng.exponential(3.0, n)
    city_effect = dict(zip(cities, [1.2, -0.6, 0.4, -1.1, 0.3, -0.2]))
    plan_effect = dict(zip(plans, [-1.0, 0.4, 1.0]))
    channel_effect = dict(zip(channels, [0.5, -0.3, 0.0, -0.6]))
    score = (2.0 * np.array([city_effect[c] for c in city]) * np.array([plan_effect[p] for p in plan])
             + np.array([channel_effect[c] for c in channel]) + 0.03 * (age - 45)
             - 0.25 * tenure + 0.6 * (np.log(income) - 10.3) + rng.normal(0, noise, n))
    frame = pd.DataFrame({"age": age.round(1), "income": income.round(0),
                          "tenure": tenure.round(2), "city": city, "plan": plan,
                          "channel": channel})
    if task == "regression":
        return frame, (50 + 12 * score).astype(np.float32)
    return frame, np.where(score > 0, "churn", "stay")
'''


def generate(n: int, noise: float, seed: int, task: str):  # noqa: ANN201
    """(frame, target) — the same rows the training script generates."""
    import numpy as np

    ns: dict = {"np": np}
    exec(compile(SYNTHETIC_TABLE_CODE, "<synthetic-table>", "exec"), ns)  # noqa: S102
    return ns["synthetic_table"](n, noise, seed, task)


def profile(params: dict, _base=None):  # noqa: ANN001, ANN201
    """Data workspace profile of the generated table."""
    from dataclasses import replace

    from ai_made_easy.core.data.profile import profile_frame

    frame, y = generate(int(params["n_samples"]), float(params["noise"]), int(params["seed"]),
                        str(params["task"]))
    frame = frame.assign(target=y)
    result = profile_frame(frame, target="target", task=str(params["task"]))
    return replace(result, type_id="data.synthetic_table", source="generated customer table")
