import random
import numpy as np
import pandas as pd

def set_random_seed(seed):
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module


def split_train_val_test(
    df,
    train_years=list(range(2004, 2017)),
    val_years=list(range(2016, 2021)),
    test_years=list(range(2020, 2024)),
):

    train_data = df[df["Year"].isin(train_years)]
    val_data = df[df["Year"].isin(val_years)]
    test_data = df[df["Year"].isin(test_years)]

    return train_data, val_data, test_data


def random_sample_train_val_test(
    df,
    split=(0.6, 0.2, 0.2),
    train_years=list(range(2004, 2016)),
    val_years=list(range(2016, 2021)),
    test_years=list(range(2020, 2024)),
    seed=42,
):
    if sum(split) != 1.0:
        raise ValueError("Splits should total to 1.0 e.g. (0.6,0.2,0.2)")
    scenario_vars = ["PlantingDay", "Treatment", "NFirstApp", "IrrgDep", "IrrgThresh"]
    groups = df.groupby(scenario_vars, observed=True, sort=False)
    unique_groups = list(groups.groups.keys())
    set_random_seed(seed=seed)
    np.random.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    print(
        f"Scenarios: train({n_train * len(train_years)}), val({n_val * len(val_years)}), test({(n - n_train - n_val) * len(test_years)})"
    )

    train_set = unique_groups[:n_train]
    val_set = unique_groups[n_train : n_train + n_val]
    test_set = unique_groups[n_train + n_val :]

    # Split data between train, val, test fro years to reduce matching
    train_data, val_data, test_data = split_train_val_test(
        df, train_years, val_years, test_years
    )

    def _filter_scenarios(_data, _dset):
        _dset = pd.DataFrame(_dset, columns=scenario_vars)
        for col in scenario_vars:
            if _data[col].dtype.name == "category":
                _dset[col] = _dset[col].astype(_data[col].dtype)
        return _data.merge(_dset, on=scenario_vars, how="inner")

    return (
        _filter_scenarios(train_data, train_set),
        _filter_scenarios(val_data, val_set),
        _filter_scenarios(test_data, test_set),
    )
