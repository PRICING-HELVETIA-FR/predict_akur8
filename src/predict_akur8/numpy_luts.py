"""Numpy-based LUT implementations for fast Akur8 scoring."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pandas.api.types import is_numeric_dtype

T = TypeVar("NumpyLut")


def _fill_by_grouped_keys(
    interpolated_betas: np.ndarray,
    row_positions: np.ndarray,
    keys: np.ndarray,
    get_lut,
    values_var2: pd.Series,
    df: pd.DataFrame,
    interpolation: str,
    values_cache: dict[str, pd.Series],
    skip_key=None
) -> None:
    """Fill interpolated betas by grouped keys with stable ordering.

    Args:
        interpolated_betas: Output array to fill.
        row_positions: Row positions aligned with keys.
        keys: Grouping keys used to select sub-LUTs.
        get_lut: Callable returning a LUT for a given key.
        values_var2: Series for the second variable of interactions.
        df: Source dataframe.
        interpolation: Interpolation method name.
        values_cache: Optional cache of precomputed series.
        skip_key: Optional key to skip.
    """
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    if keys_sorted.size == 0:
        return
    changes = np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, keys_sorted.size]
    for start, end in zip(starts, ends):
        key = keys_sorted[start]
        if skip_key is not None and key == skip_key:
            continue
        sub_positions = row_positions[order[start:end]]
        lut = get_lut(key)
        if lut is not None:
            interpolated_betas[sub_positions] = lut._compute_betas(
                values_var2.iloc[sub_positions],
                df.iloc[sub_positions],
                values_cache,
                interpolation
            )


class NumpyLut(ABC):
    """Base class for numpy-accelerated LUT lookup."""
    def __init__(self, lut: pd.DataFrame, var: str):
        """Store LUT metadata and target variable name.

        Args:
            lut: Lookup table dataframe.
            var: Variable name for this LUT.
        """
        self.var_name = var
        self.values_type = lut[var].dtype
        self.is_numeric = is_numeric_dtype(lut[var])
        
    def _get_values_to_search(
        self,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series]
    ) -> pd.Series:
        """Fetch and normalize the input series for this LUT.

        Args:
            df: Input dataframe.
            values_cache: Optional cache of precomputed series.
        """
        cached = values_cache.get(self.var_name)
        if cached is not None:
            if cached.index is df.index:
                return cached
            return cached.reindex(df.index)

        try:
            values_to_search = df[self.var_name]
        except KeyError:
            raise Exception(f'The dataframe to predict does not include a column named {self.var_name}')
        
        try:
            if self.is_numeric:
                res = pd.to_numeric(values_to_search, errors='raise').astype(self.values_type).astype(float)
                values_type = 'float'
            else:
                res = values_to_search.fillna('').astype(self.values_type).astype(str)
                values_type = 'str'
        except:
            raise Exception(f'In the dataframe to predict, the column {self.var_name} of must have a type that can be casted to {values_type}')
            
        values_cache[self.var_name] = res
        
        return res
        
    
    def compute_betas(
        self,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> float:
        """Compute betas for the dataframe using the given interpolation.

        Args:
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        values_to_search = self._get_values_to_search(df, values_cache)
        return self._compute_betas(values_to_search, df, values_cache, interpolation)
    
    
    @abstractmethod
    def _compute_betas(
        self,
        values_to_search: pd.Series,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> float:
        """Core beta computation for a prepared series.

        Args:
            values_to_search: Prepared series to look up.
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        raise NotImplementedError
    
    
class NumpyNumLutAbstract(NumpyLut):
    """Shared helpers for numeric LUTs."""
    def __init__(self, lut: pd.DataFrame, var: str):
        """Initialize numeric values array for interval lookup.

        Args:
            lut: Lookup table dataframe.
            var: Variable name for this LUT.
        """
        super().__init__(lut, var)
        self.values = lut[var].dropna().unique()
        self.values_count = len(self.values)
          
          
    def get_interval(self, values_to_search: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return lower/upper index bounds for each value.

        Args:
            values_to_search: Values to locate in the LUT grid.
        """
        idx = np.searchsorted(self.values, values_to_search, side="left")
        idx_max = self.values_count - 1
        idx_inf = np.clip(idx - 1, 0, idx_max)
        idx_sup = np.clip(idx,     0, idx_max)
        return idx_inf, idx_sup


    def nearest_index(self, values_to_search: pd.Series, idx_inf: np.ndarray=None, idx_sup: np.ndarray=None) -> np.ndarray:
        """Return index of nearest value for each input.

        Args:
            values_to_search: Values to locate in the LUT grid.
            idx_inf: Optional lower bound indices.
            idx_sup: Optional upper bound indices.
        """
        if idx_inf is None or idx_sup is None:
            idx_inf, idx_sup = self.get_interval(values_to_search)
        left = self.values[idx_inf]
        right = self.values[idx_sup]
        return np.where(values_to_search - left < right - values_to_search, idx_inf, idx_sup)
        

class NumpyNumLut(NumpyNumLutAbstract):
    """Numeric LUT with interpolation support."""
    def __init__(self, lut: pd.DataFrame, var: str):
        """Load numeric LUT arrays for fast interpolation.

        Args:
            lut: Lookup table dataframe.
            var: Variable name for this LUT.
        """
        super().__init__(lut, var)
        mask_not_nan = lut[var].notna()
        temp = lut.loc[~mask_not_nan, 'beta'].values
        self.betas = lut.loc[mask_not_nan, 'beta'].to_numpy()
        self.empty_beta = temp[0] if len(temp) == 1 else np.nan
        self.slopes = lut.loc[mask_not_nan, 'slope'].to_numpy()
        self.pchip3 = lut.loc[mask_not_nan, 'pchip3'].to_numpy() 
        self.pchip2 = lut.loc[mask_not_nan, 'pchip2'].to_numpy()
        self.pchip1 = lut.loc[mask_not_nan, 'pchip1'].to_numpy()
        
    
    def _compute_betas(
        self,
        values_to_search: pd.Series,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> pd.Series:
        """Compute interpolated betas for numeric inputs.

        Args:
            values_to_search: Prepared series to look up.
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        mask_not_nan = values_to_search.notna()
        # Initialize betas vector with the default coefficient (for NaN values_to_search)
        interpolated_betas = np.full(values_to_search.shape, self.empty_beta, dtype=np.float64)
                
        if not np.any(mask_not_nan):
            return interpolated_betas
        
        # For non-NaN values_to_search, compute the requested interpolation
        values_to_search_not_nan = values_to_search[mask_not_nan]
        idx_inf, idx_sup = self.get_interval(values_to_search[mask_not_nan])
        
        if interpolation == 'nearest':
            idx = self.nearest_index(values_to_search_not_nan, idx_inf, idx_sup)
            interpolated_betas[mask_not_nan] = self.betas[idx]
        elif interpolation == 'linear':
            linear = self.slopes[idx_inf] * (values_to_search_not_nan - self.values[idx_inf]) + self.betas[idx_inf]
            # where condition: exact match or out-of-bounds
            interpolated_betas[mask_not_nan] = np.where(idx_inf == idx_sup, self.betas[idx_inf], linear)
        elif interpolation == 'pchip':
            dx = values_to_search_not_nan - self.values[idx_inf]
            # Horner's method to reduce multiplications
            pchip = self.betas[idx_inf] + dx * (self.pchip1[idx_inf] + dx * (self.pchip2[idx_inf] + dx * self.pchip3[idx_inf]))
            # where condition: exact match or out-of-bounds
            interpolated_betas[mask_not_nan] = np.where(idx_inf == idx_sup, self.betas[idx_inf], pchip)
        else:
            raise Exception(f"La méthode d'interpolation {interpolation} est inconnue.")
        
        return interpolated_betas
        
        
class NumpyCatLut(NumpyLut):
    """Categorical LUT implemented as a dict lookup."""
    def __init__(self, lut: pd.DataFrame, var: str):
        """Build a mapping from category to beta.

        Args:
            lut: Lookup table dataframe.
            var: Variable name for this LUT.
        """
        super().__init__(lut, var)
        self.betas = lut[[var, 'beta']].set_index(var).to_dict()['beta']
        
    def _compute_betas(
        self,
        values_to_search: pd.Series,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> pd.Series:
        """Map categories to betas.

        Args:
            values_to_search: Prepared series to look up.
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        return values_to_search.map(self.betas).to_numpy()


class NumpyNumNumLut(NumpyNumLutAbstract):
    """Interaction LUT for numeric x numeric features."""
    def __init__(self, lut: pd.DataFrame, var1: str, var2):
        """Split sub-LUTs by the first numeric variable.

        Args:
            lut: Lookup table dataframe.
            var1: Name of the first variable.
            var2: Name of the second variable.
        """
        super().__init__(lut, var1)
        mask_nan = lut[var1].isna()
        self.empty_betas = NumpyNumLut(lut[mask_nan], var2)
        self.numpy_num_luts = [
            NumpyNumLut(sublut, var2) 
            for val1, sublut in lut[~mask_nan].groupby(var1, as_index=False, dropna=False)
        ]
        
    
    def _compute_betas(
        self,
        values_to_search: pd.Series,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> pd.Series:
        """Compute betas for numeric x numeric interactions.

        Args:
            values_to_search: Prepared series to look up.
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        mask_nan = values_to_search.isna()
        # Initialize betas vector to NaN
        interpolated_betas = np.full(values_to_search.shape, np.nan, dtype=np.float64)
        values_var2 = self.empty_betas._get_values_to_search(df, values_cache)
        # When the first interaction variable is missing, compute betas with the associated sub-table
        if np.any(mask_nan):
            row_positions = np.flatnonzero(mask_nan.to_numpy())
            interpolated_betas[row_positions] = self.empty_betas._compute_betas(
                values_var2.iloc[row_positions],
                df.iloc[row_positions],
                values_cache,
                interpolation
            )
        # When the first interaction variable is present, look up the nearest value
        mask_not_nan = ~mask_nan
        if np.any(mask_not_nan):
            row_positions = np.flatnonzero(mask_not_nan.to_numpy())
            values_not_nan = values_to_search.to_numpy()[row_positions]
            idxs = self.nearest_index(values_not_nan)
            _fill_by_grouped_keys(
                interpolated_betas,
                row_positions,
                idxs,
                lambda idx: self.numpy_num_luts[idx],
                values_var2,
                df,
                interpolation,
                values_cache
            )
        return interpolated_betas
    
          
    
class NumpyInterLut(NumpyLut):
    """Base class for interaction LUTs keyed by a first variable."""
    def __init__(self, lut: pd.DataFrame, var1: str, var2: str, cls: Type[T]):
        """Build sub-LUTs by grouping on the first variable.

        Args:
            lut: Lookup table dataframe.
            var1: Name of the first variable.
            var2: Name of the second variable.
            cls: LUT class to build for sub-tables.
        """
        super().__init__(lut, var1)
    
        self.numpy_subluts = {
            val1: cls(sublut, var2) 
            for val1, sublut in lut.groupby(var1, as_index=False, dropna=False)
        }
        self._var2_lut = next(iter(self.numpy_subluts.values())) if self.numpy_subluts else None
        
    
    def _compute_betas(
        self,
        values_to_search: pd.Series,
        df: pd.DataFrame,
        values_cache: dict[str, pd.Series],
        interpolation: str='pchip'
    ) -> pd.Series:
        """Compute betas for interaction LUTs via grouped lookup.

        Args:
            values_to_search: Prepared series to look up.
            df: Input dataframe.
            interpolation: Interpolation method name.
            values_cache: Optional cache of precomputed series.
        """
        interpolated_betas = np.full(values_to_search.shape, np.nan, dtype=np.float64)
        if self._var2_lut is None:
            return interpolated_betas
        
        values_var2 = self._var2_lut._get_values_to_search(df, values_cache)
        values_arr = values_to_search.to_numpy()
        codes, uniques = pd.factorize(values_arr, sort=False)
        row_positions = np.arange(values_arr.shape[0])
        _fill_by_grouped_keys(
            interpolated_betas,
            row_positions,
            codes,
            lambda code: self.numpy_subluts.get(uniques[code]),
            values_var2,
            df,
            interpolation,
            values_cache,
            skip_key=-1
        )
        
        return interpolated_betas


class NumpyCatCatLut(NumpyInterLut):
    def __init__(self, lut: pd.DataFrame, var1: str, var2):
        """Categorical x categorical interaction LUT.

        Args:
            lut: Lookup table dataframe.
            var1: Name of the first variable.
            var2: Name of the second variable.
        """
        super().__init__(lut, var1, var2, NumpyCatLut)
        
        
class NumpyCatNumLut(NumpyInterLut):
    def __init__(self, lut: pd.DataFrame, var1: str, var2):
        """Categorical x numeric interaction LUT.

        Args:
            lut: Lookup table dataframe.
            var1: Name of the first variable.
            var2: Name of the second variable.
        """
        super().__init__(lut, var1, var2, NumpyNumLut)