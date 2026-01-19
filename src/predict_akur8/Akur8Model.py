"""Akur8 JSON model parser and scorer utilities."""

import numpy as np
import pandas as pd
from flatten_dict import flatten
from scipy.special import expit
from pandas.api.types import is_numeric_dtype

from .numpy_luts import NumpyCatCatLut, NumpyCatNumLut, NumpyNumLut, NumpyNumNumLut, NumpyCatLut, NumpyLut
from .utils import complete_lut, is_float_key, DELIM_MODEL_VARS, DELIM_VARS_INTER, compress_value1_interaction, compress_simple_lut, get_unique_values
from scipy.interpolate import PchipInterpolator
from math import log
import pickle
import json
from enum import Enum


class Kind(Enum):
    NUM = 1
    CAT = 2
    NUM_FORCED_TO_CAT = 3


class Akur8Model:
    """Akur8 scorer with:
    - JSON parsing via flatten_dict
    - Optional training DataFrame to build train-complete numeric grids
    - Interpolation at scoring for values truly out-of-training

    Unknown categorical modalities => NaN contributions + collected report.
    """
    link: str
    intercept: float
    model_name: str
    interpolate: bool
    var_kind: dict[str, Kind]
    simple_luts: dict[str, NumpyLut] | None
    inter_luts: dict[tuple[str, str], NumpyLut] | None
    __simple_pandas_luts: dict[str, pd.DataFrame] | None
    __inter_pandas_luts: dict[tuple[str, str], pd.DataFrame] | None

    def __init__(
        self,
        model_json: dict[str, object] | str,
        train_df: pd.DataFrame | None = None,
        force_to_categorical: set[str] | None = None,
        model_name: str | None = None,
        interpolate: bool=True,
        compress_look_up_tables: bool=True
    ):
        """Initialize the model from Akur8 JSON and optional training data.

        Args:
            model_json: Akur8 model JSON as a dict or path to the JSON.
            train_df: Optional training dataframe for grid completion.
            force_to_categorical: set of variable that look like numerical 
            but the user wants to force as categorical (no completion, no 
            interpolation, only exact matches). When train_df is supplied, 
            forcing is possible only when the number of distinct values is
            less or equal 256, otherwise it will remain numerical.
            model_name: Optional name override for output columns.
            interpolate: Whether to compute interpolation coefficients.
            compress_look_up_tables: Whether to compress LUTs for speed.
        """
        if isinstance(model_json, str):
            with open(model_json, "r", encoding="utf-8") as f:
                model_json_dict = json.load(f)
        else:
            model_json_dict = model_json
            
        self.link = model_json_dict['linkType']
        self.intercept = float(model_json_dict['intercept'])
        if self.link == 'LOG':
            self.intercept = log(self.intercept)
        self.model_name = model_name if model_name is not None else model_json_dict['projectName']

        # Parse JSON to flat tables
        self.__simple_pandas_luts = self.__parse_simple_df(model_json_dict)   # var, mod, coef
        self.__inter_pandas_luts = self.__parse_inter_df(model_json_dict)    # var1, var2, m1, m2, coef
        
        train_df_copy = train_df.copy() if train_df is not None else None
        
        self.__infer_var_types(train_df_copy, force_to_categorical)
        self.__complete_luts(train_df_copy)
        self.__calculate_interpolators(interpolate)
        self.__compress_luts(compress_look_up_tables)
        self.__luts_to_numpy()
        
        
    def __compress_luts(self, compress_look_up_tables: bool):
        """Optionally compress LUTs to speed up scoring.

        Args:
            compress_look_up_tables: Whether to compress LUTs.
        """
        if not compress_look_up_tables:
            return
        
        # When a variable is only used in interactions, Akur8 creates a simple LUT with all betas = 0. 
        # We remove these LUTs to avoid useless computations at scoring.
        to_delete = [var for var in self.__simple_pandas_luts.keys() if self.__simple_pandas_luts[var]['beta'].eq(0).all()]
        for var in to_delete:
            del self.__simple_pandas_luts[var]
        
        # Compressing numeric LUTs by removing intermediates points in beta plateaus
        for var in self.simple_numeric:
            self.__simple_pandas_luts[var] = compress_simple_lut(self.__simple_pandas_luts[var])
            
        # Same for interactions
        for var1, var2 in self.inter_numeric:
            # Compressing first var
            lut = compress_value1_interaction(self.__inter_pandas_luts[(var1, var2)], var1, var2)
            # Compressing each sub table along second var
            subluts_compressed = list()
            for _, sublut in lut.groupby(var1, as_index=False, dropna=False):
                subluts_compressed.append(compress_simple_lut(sublut))
                
            self.__inter_pandas_luts[(var1, var2)] = pd.concat(subluts_compressed)
    
    
    def __complete_luts(self, train_df: pd.DataFrame):
        """Complete numeric LUTs using training data for nearest matching.

        Args:
            train_df: Training dataframe used to complete LUTs.
        """
        if train_df is None:
            return
        
        # Simples effects
        for var in self.simple_numeric:
            lut = self.__simple_pandas_luts[var]       
            train_unique_values = get_unique_values(train_df, lut, var)
            self.__simple_pandas_luts[var] = complete_lut(train_unique_values, lut, var)
        
        # Interactions
        for var1, var2 in self.inter_numeric:
            lut = self.__inter_pandas_luts[(var1, var2)]
            train_unique_values = get_unique_values(train_df, lut, var2)
            sublut_completed_list = list()

            for _, sublut in lut.groupby(var1, as_index=False, dropna=False):
                sublut_completed = complete_lut(train_unique_values, sublut, var2)
                sublut_completed_list.append(sublut_completed)
                
            self.__inter_pandas_luts[(var1, var2)] = pd.concat(sublut_completed_list)
            
        
    def __parse_simple_df(self, model_json: dict[str, object]) -> dict[str, pd.Series]:
        """Parse simple-effect coefficients into per-variable LUTs.

        Args:
            model_json: Akur8 model JSON as a dict.
        """
        flat = flatten(model_json.get('coefficients', {}))
        if not flat:
            return dict()
        s = pd.Series(flat).reset_index()
        s.columns = ['var', 'mod', 'beta']
        s['var'] = s['var'].astype(str)
        s['mod'] = s['mod'].astype(str)
        s['beta'] = s['beta'].astype(float)
        if self.link == 'LOG':
            s['beta'] = np.log(s['beta'])
        return {
            var: g[['mod', 'beta']].rename(columns={'mod': var})
            for var, g in s.groupby('var', sort=False, as_index=False, dropna=False)
        }


    def __parse_inter_df(self, model_json: dict[str, object]) -> dict[tuple[str, str], pd.Series]:
        """Parse interaction coefficients into per-variable-pair LUTs.

        Args:
            model_json: Akur8 model JSON as a dict.
        """
        flat = flatten(model_json.get('interactionsCoefficients', {}), reducer='tuple')  # (v1,v2,m1,m2)->coef
        if not flat:
            return dict()
        s = pd.Series(flat, name='beta').reset_index()
        s.columns = ['var1', 'var2', 'm1', 'm2', 'beta']
        for c in ['var1', 'var2', 'm1', 'm2']:
            s[c] = s[c].astype(str)
        s['beta'] = s['beta'].astype(float)
        if self.link == 'LOG':
            s['beta'] = np.log(s['beta'])
        return {
            (v1, v2): g[['m1', 'm2', 'beta']].rename(columns={'m1': v1, 'm2': v2})
            for (v1, v2), g in s.groupby(['var1', 'var2'], sort=False, dropna=False)
        }
    
    
    def cast_luts(self, lut: pd.DataFrame, var: str, train_df: pd.DataFrame):
        """Cast a LUT variable to numeric, matching train_df dtype when provided.

        Args:
            lut: Lookup table dataframe to mutate.
            var: Variable name to cast.
            train_df: Training dataframe for dtype matching.
        """
        if self.var_kind[var] in (Kind.NUM, Kind.NUM_FORCED_TO_CAT):
            cast_fun = lambda x: pd.to_numeric(x.fillna('').astype(str).str.replace(',', '.').replace({'': np.nan})).astype(float)
        else:
            # Replace NaN by '' for categorical variables before casting to avoid 'NaN' or 'None' strings
            cast_fun = lambda x: x.fillna('').astype(str)
            
        lut[var] = cast_fun(lut[var])
        if train_df is not None:
            # When train_df is provided, we cast train_df[var] in the same type as lut[var] to allow lookups 
            train_df[var] = cast_fun(train_df[var])
            
        
        
    def __infer_var_types(self, train_df: pd.DataFrame, force_to_categorical: set[str]):
        """Infer variable numeric types and normalize LUT ordering.

        Args:
            train_df: Optional training dataframe.
            force_to_categorical: variable names to be forced as CAT
        """
        self.var_kind = dict()
        force_to_categorical = force_to_categorical or set()
        
        if train_df is None:
            self.__infer_var_types_from_json(force_to_categorical)
        else:
            self.__infer_var_types_from_train_df(train_df, force_to_categorical)
        
        for var, lut in self.__simple_pandas_luts.items():
            self.cast_luts(lut, var, train_df)
            # Sort to accelerate future beta look ups
            lut.sort_values(by=[var], inplace=True)
        
        a_permuter = list()
        for (var1, var2), lut in self.__inter_pandas_luts.items():
            self.cast_luts(lut, var1, train_df)
            self.cast_luts(lut, var2, train_df)
            is_num_var1 = self.var_kind[var1] == Kind.NUM
            is_num_var2 = self.var_kind[var2] == Kind.NUM
        
            df_for_unique_count = train_df if train_df is not None else lut
            # Permutation if we are in one of the following situations :
            # - CAT x CAT or NUM x NUM interaction and the first variable has more distinct values than the second
            # - NUM x CAT interaction: we put the CAT feature firstdevant to allow interpolations on the NUM feature
            if (
                is_num_var1 == is_num_var2 and 
                df_for_unique_count[var1].nunique() > df_for_unique_count[var2].nunique() or 
                is_num_var1 and not is_num_var2
            ):
                a_permuter.append((var1, var2))
                
        for var1, var2 in a_permuter:
            lut = self.__inter_pandas_luts.pop((var1, var2))
            self.__inter_pandas_luts[(var2, var1)] = (
                lut[[var2, var1, lut.columns[2]]]
                .sort_values(by=[var2, var1])
            )
            
        # Sort to accelerate future beta look ups
        for (var1, var2), lut in self.__inter_pandas_luts.items():
            lut.sort_values(by=[var1, var2], inplace=True)
            
        self.simple_numeric = [var for var in self.__simple_pandas_luts.keys() if self.var_kind[var] == Kind.NUM]
        # Interactions whose second feature is numeric
        self.inter_numeric = [(var1, var2) for var1, var2 in self.__inter_pandas_luts.keys() if self.var_kind[var2] == Kind.NUM]
            

    def __infer_var_types_from_json(self, force_to_categorical: set[str]):
        """Infer numeric types by checking if all values parse as floats."""
        values_by_variable: dict[str, set[str]] = {}
        
        for var, lut in self.__simple_pandas_luts.items():
            values_by_variable[var] = set(lut.reset_index()[var].unique())
            
        for (var1, var2), lut in self.__inter_pandas_luts.items():
            temp = lut.reset_index()
            valeurs_var1 = values_by_variable.get(var1, set())
            valeurs_var1 = valeurs_var1.union(temp[var1].unique())
            values_by_variable[var1] = valeurs_var1
            valeurs_var2 = values_by_variable.get(var2, set())
            valeurs_var2 = valeurs_var2.union(temp[var2].unique())
            values_by_variable[var2] = valeurs_var2
        
        for var, values in values_by_variable.items():
            if all(is_float_key(v) for v in values):
                res = Kind.NUM_FORCED_TO_CAT if var in force_to_categorical else Kind.NUM
            else:
                res = Kind.CAT
                
            self.var_kind[var] = res
    
    
    def __infer_var_types_from_train_df(self, train_df: pd.DataFrame, force_to_categorical: set[str]):
        """Infer numeric types from the training dataframe dtypes.

        Args:
            train_df: Training dataframe to inspect.
        """
        variables = set(self.__simple_pandas_luts.keys())
        variables = variables.union(t[0] for t in self.__inter_pandas_luts.keys())
        variables = variables.union(t[1] for t in self.__inter_pandas_luts.keys())
        
        for variable in variables:
            is_numeric = is_numeric_dtype(train_df[variable])
            looks_numeric = is_numeric
            if not is_numeric:
                try:
                    _ = pd.to_numeric(train_df[variable].dropna().unique(), errors='raise')
                    looks_numeric = True
                except:
                    looks_numeric = False
                    
            number_distinct_values = train_df[variable].nunique()
                    
            if is_numeric:
                res = Kind.NUM
                if number_distinct_values <= 256 and variable in force_to_categorical:
                    res = Kind.NUM_FORCED_TO_CAT
            elif not looks_numeric:
                res = Kind.CAT
            elif not is_numeric and looks_numeric:
                if number_distinct_values > 256: # TODO: A affiner en comparant au nombre de valeurs uniques dans les luts
                    res = Kind.NUM
                else:
                    res = Kind.NUM_FORCED_TO_CAT
                    
            self.var_kind[variable] = res
    
    
    def __calculate_interpolators(self, interpolate: bool):
        """Precompute interpolation coefficients for numeric LUTs.

        Args:
            interpolate: Whether to compute interpolation coefficients.
        """
        self.interpolate = interpolate
        if not interpolate:
            return
        
        # Simples effects
        for var in self.simple_numeric:
            self.__simple_pandas_luts[var] = self.__calculate_interpolators_for_var(self.__simple_pandas_luts[var], var)
            
        # Interactions
        for var1, var2 in self.inter_numeric:
            lut = self.__inter_pandas_luts[(var1, var2)]
            
            sublut_list = list()
            for val1, sublut in lut.groupby(var1, as_index=False, dropna=False):
                temp = self.__calculate_interpolators_for_var(sublut, var2)
                sublut_list.append(temp)
                
            self.__inter_pandas_luts[(var1, var2)] = pd.concat(sublut_list)
    
    
    def __calculate_interpolators_for_var(self, lut: pd.DataFrame, var: str) -> pd.DataFrame:
        """Add linear and PCHIP interpolation coefficients to a LUT.

        Args:
            lut: Lookup table dataframe.
            var: Variable name.
        """
        # For linear interpolation
        lut['slope'] = (lut['beta'].shift(-1) - lut['beta']) / (lut[var].shift(-1) - lut[var])
        
        # For Piecewise Cubic Hermite Interpolating Polynomial
        mask_not_nan = lut[var].notna()
        x = lut.loc[mask_not_nan, var].to_numpy()
        y = lut.loc[mask_not_nan, 'beta'].to_numpy()
        interpolator = PchipInterpolator(x=x, y=y)

        coefficients_pchip = (
            pd.DataFrame(interpolator.c.transpose(), columns=['pchip3', 'pchip2', 'pchip1', 'pchip0'])
            .drop(columns=['pchip0']) # Not needed because pchip0 = var
        )
        lut[['pchip3', 'pchip2', 'pchip1']] = np.nan
        positions = np.flatnonzero(mask_not_nan.to_numpy())
        n_segments = coefficients_pchip.shape[0]
        if n_segments > 0:
            col_idx = [lut.columns.get_loc(c) for c in ('pchip3', 'pchip2', 'pchip1')]
            lut.iloc[positions[:n_segments], col_idx] = coefficients_pchip.to_numpy()
        
        return lut
    
    
    def __get_col_coef_name(self, var: str, var2: str=None) -> str:
        """Build output column name for a variable or interaction.

        Args:
            var: Variable name.
            var2: Optional second variable name for interactions.
        """
        if var2 is None:
            return f'{self.model_name}{DELIM_MODEL_VARS}{var}'
        else:
            return f'{self.model_name}{DELIM_MODEL_VARS}{var}{DELIM_VARS_INTER}{var2}'
        
    
    def __luts_to_numpy(self):
        """Convert pandas LUTs into numpy-optimized LUT classes."""
        self.simple_luts = dict()
        self.inter_luts = dict()
        
        # Simple effects
        for var, lut in self.__simple_pandas_luts.items():
            # Num
            if self.var_kind[var] == Kind.NUM:
                self.simple_luts[var] = NumpyNumLut(lut, var)
            # Cat or Num forced ton cat
            else:
                self.simple_luts[var] = NumpyCatLut(lut, var)
                
        self.__simple_pandas_luts = None
                
        # Interactions
        for (var1, var2), lut in self.__inter_pandas_luts.items():
            is_num_1 = self.var_kind[var1] == Kind.NUM
            is_num_2 = self.var_kind[var2] == Kind.NUM
            key = (var1, var2)
            # Num x Num
            if is_num_1 and is_num_2:
                self.inter_luts[key] = NumpyNumNumLut(lut, var1, var2)
            # Cat x Num
            if not is_num_1 and is_num_2:
                self.inter_luts[key] = NumpyCatNumLut(lut, var1, var2) 
            # Cat x Cat
            if not is_num_1 and not is_num_2:
                self.inter_luts[key] = NumpyCatCatLut(lut, var1, var2)
                
        self.__inter_pandas_luts = None
        
        
    def __check_interpolation_method(self, interpolation: str, var: str):
        """Validate interpolation choice against model configuration.

        Args:
            interpolation: Interpolation method name.
            var: Variable name the interpolation is applied to.
        """
        if interpolation in ('linear', 'pchip') and not self.interpolate:
            raise Exception(f"This instance of Akur8Model has been created without computing interpolators (interpolate=False). Therefore it is imposible ton interpolate the variable {var} with interpolator {interpolation}. ")
        
        
    def predict(
        self, 
        df: pd.DataFrame, 
        default_interpolation: str | None=None,
        interpolation_simple: dict[str, str] | None = None, 
        interpolation_inter: dict[tuple[str, str], str] | None = None
    ) -> pd.DataFrame:
        """Score a dataframe and return input with added coefficient and prediction columns.

        Args:
            df: Input dataframe to score.
            default_interpolation: Fallback interpolation method. Default value is 'pchip' if the instance was created with interpolate=True, else 'nearest'.
            interpolation_simple: Per-variable interpolation overrides.
            interpolation_inter: Per-interaction interpolation overrides.
        """
        if self.interpolate:
            default_interpolation = default_interpolation or 'pchip'
        else:
            default_interpolation = default_interpolation or 'nearest'

        if interpolation_simple is None:
            interpolation_simple = dict()
        if interpolation_inter is None:
            interpolation_inter = dict()
        values_cache = dict()
        out = pd.DataFrame(index=df.index)
        col_intercept = f'{self.model_name}::intercept'
        out[col_intercept] = self.intercept
        cols_to_sum = [col_intercept]
        
        for var, lut in self.simple_luts.items():
            coef_column = self.__get_col_coef_name(var)
            interpolation = interpolation_simple.get(var, default_interpolation)
            self.__check_interpolation_method(interpolation, var)
            out[coef_column] = lut.compute_betas(df, values_cache, interpolation)
            cols_to_sum.append(coef_column)
            
        for (var1, var2), lut in self.inter_luts.items():
            coef_column = self.__get_col_coef_name(var1, var2)
            interpolation = interpolation_inter.get((var1, var2), default_interpolation)
            self.__check_interpolation_method(interpolation, var2)
            out[coef_column] = lut.compute_betas(df, values_cache, interpolation)
            cols_to_sum.append(coef_column)
            
        col_prediction = f'{self.model_name}::prediction'
        out[col_prediction] = out[cols_to_sum].sum(axis=1, skipna=False)
        if self.link == 'LOG':
            out[col_prediction] = np.exp(out[col_prediction])
        if self.link == 'LOGIT':
            out[col_prediction] = expit(out[col_prediction])
            
        return pd.concat([df, out], axis=1)
        
    def to_pickle(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
         
            
    @staticmethod
    def from_pickle(path: str):
        with open(path, 'rb') as f:
            # Allow loading pickles created from "src.predict_akur8" imports.
            class _Akur8Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith("src."):
                        module = module.replace("src.", "", 1)
                    return super().find_class(module, name)
            return _Akur8Unpickler(f).load()
