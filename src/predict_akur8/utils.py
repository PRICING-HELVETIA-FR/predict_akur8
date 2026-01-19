"""Utility helpers for LUT processing, interpolation preparation, and reporting."""

import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum


class Kind(Enum):
    NUM = 1
    CAT = 2
    NUM_FORCED_TO_CAT = 3
    

DELIM_MODEL_VARS = '::coef::'
DELIM_VARS_INTER = ' x '


def to_float_fr(x: str) -> float:
    """Parse a float string (dot or comma decimal), empty string -> NaN.

    Args:
        x: Input string to parse.
    """
    return np.nan if x == "" else float(x.replace(",", "."))


def is_float_key(k: str) -> bool:
    """Return True when a key can be parsed as a float.

    Args:
        k: Input key to test.
    """
    try:
        _ = to_float_fr(k)
        return True
    except Exception:
        return False


def report_unknown_values(out: pd.DataFrame) -> pd.DataFrame:
    """Report missing coefficients by model/variable/value with counts and percentages.

    Args:
        out: Scoring output dataframe with coefficient columns.
    """
    res_list = []
    coef_cols = [c for c in out.columns if 'coef::' in c]

    for col in coef_cols:
        mask = out[col].isna()
        if not mask.any():
            continue

        model, vars_join = col.split(DELIM_MODEL_VARS)
        variables = vars_join.split(DELIM_VARS_INTER)

        if len(variables) == 1:
            variable = variables[0]
            res_temp = (
                out.loc[mask, variable]
                .value_counts(dropna=False)
                .rename_axis('value')
                .reset_index(name='nb')
            )
        else:
            variable = str(tuple(variables))
            res_temp = (
                out.loc[mask, variables]
                .value_counts(dropna=False)
                .rename_axis(variables)
                .reset_index(name='nb')
            )
            res_temp['value'] = list(res_temp[variables].itertuples(index=False, name=None))
            res_temp = res_temp.drop(columns=variables)

        res_temp['model'] = model
        res_temp['variable'] = variable
        res_list.append(res_temp)

    if not res_list:
        res = pd.DataFrame(columns=['model', 'variable', 'value', 'nb', 'pct'])
    else:
        res = pd.concat(res_list, ignore_index=True)[['model', 'variable', 'value', 'nb']]
        res['pct'] = res['nb'] / len(out)
        res = res.sort_values(by=['model', 'variable', 'value', 'nb'])
        
    return res


def get_unique_values(train_df: pd.DataFrame, lut: pd.DataFrame, var: str) -> pd.DataFrame:
    """Return sorted unique values from training data and LUT for a variable.

    Args:
        train_df: Training dataframe.
        lut: Lookup table dataframe.
        var: Variable name.
    """
    return (
        # Union of training database values and look up table values
        # (interactions look up tables may feature couples of values not present in the training database)
        pd.concat([
            train_df.loc[train_df[var].notna(), [var]], # On retire les nans qui ne sont pas acceptés par le merge_asof
            lut.loc[lut[var].notna(), [var]]
        ])
        .drop_duplicates()
        .sort_values(by=[var])
    )


def complete_lut(train_unique_values: pd.DataFrame, lut: pd.DataFrame, var: str) -> pd.DataFrame:
    """Fill numeric LUT gaps by nearest match using merge_asof.

    Args:
        train_unique_values: Unique values from training data.
        lut: Lookup table dataframe.
        var: Variable name.
    """
    mask_nan = lut[var].isna()
    # merge_asof performs a classical merge + nearest value lookup if no exact match is found
    # It allows to replicate the Akur8 prediction method by providing the coefficient of the closest
    # value among those present in the JSON (training database values are not always present in the JSON
    # because Akur8 creates buckets in case there is more than 256 distinct values for a numerical feature)
    lut_not_nan_completed = pd.merge_asof(
        left=train_unique_values,
        right=lut[~mask_nan],
        left_on=var,
        right_on=var,
        direction='nearest',
        allow_exact_matches=True
    )
    # Adding nan rows of the initial dataframe after merge_asof merge_asof
    return pd.concat([lut[mask_nan], lut_not_nan_completed])


def partition_array(series: pd.Series) -> list:
    """Return list of (value, indices) partitions for a series.

    Args:
        series: Series to partition.
    """
    partition = defaultdict(list)
    for i, x in enumerate(series):
        partition[x].append(i)

    return [(k, np.asarray(v)) for k, v in partition.items()]


def compress_simple_lut(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the first and the last row of each beta plateau
    df must be sorted by var.

    Args:
        df: Lookup table dataframe sorted by variable.
    """
    beta = df["beta"]

    # True if beta differs from the previous row 
    start_plateau = beta != beta.shift(1)

    # True if beta differs from the next row 
    end_plateau = beta != beta.shift(-1)

    return df.loc[start_plateau | end_plateau]


def compress_value1_interaction(
    df: pd.DataFrame,
    var1: str, 
    var2: str,
    atol=1e-12,
    rtol=1e-12,
) -> pd.DataFrame:
    """Reduce interaction LUT by keeping only value1 levels with beta changes.

    Args:
        df: Interaction lookup table dataframe.
        var1: Name of the first variable.
        var2: Name of the second variable.
        atol: Absolute tolerance for comparing betas.
        rtol: Relative tolerance for comparing betas.
    """
    u1 = df[var1].unique()
    u2 = df[var2].unique()
    n1, n2 = len(u1), len(u2)

    arr = df.loc[:, 'beta'].to_numpy().reshape(n1, n2)
    eq_prev = np.isclose(arr[1:], arr[:-1], atol=atol, rtol=rtol, equal_nan=True).all(axis=1)

    start = np.r_[True, ~eq_prev]
    end   = np.r_[~eq_prev, True]

    return df[df[var1].isin(u1[start | end])]