"""Akur8 JSON model parser and scorer utilities."""

import numpy as np
import pandas as pd
from flatten_dict import flatten
from scipy.special import expit

from .numpy_luts import NumpyCatCatLut, NumpyCatNumLut, NumpyNumLut, NumpyNumNumLut, NumpyCatLut, NumpyLut
from .utils import complete_lut, is_float_key, DELIM_MODEL_VARS, DELIM_VARS_INTER, compress_value1_interaction, compress_simple_lut, get_unique_values
from scipy.interpolate import PchipInterpolator
from math import log
from pickle import load, dump


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
    simple_pandas_luts: dict[str, pd.DataFrame] | None
    inter_pandas_luts: dict[tuple[str, str], pd.DataFrame] | None
    var_is_numeric: dict[str, bool]
    simple_numpy_luts: dict[str, NumpyLut] | None
    inter_numpy_luts: dict[tuple[str, str], NumpyLut] | None

    def __init__(
        self,
        model_json: dict[str, object],
        train_df: pd.DataFrame | None = None,
        model_name: str | None = None,
        interpolate: bool=True,
        compress_look_up_tables: bool=True
    ):
        """Initialize the model from Akur8 JSON and optional training data.

        Args:
            model_json: Akur8 model JSON as a dict.
            train_df: Optional training dataframe for grid completion.
            model_name: Optional name override for output columns.
            interpolate: Whether to compute interpolation coefficients.
            compress_look_up_tables: Whether to compress LUTs for speed.
        """
        self.link = model_json['linkType']
        self.intercept = float(model_json['intercept'])
        if self.link == 'LOG':
            self.intercept = log(self.intercept)
        self.model_name = model_name if model_name is not None else model_json['projectName']

        # Parse JSON to flat tables
        self.simple_pandas_luts = self.__parse_simple_df(model_json)   # var, mod, coef
        self.inter_pandas_luts = self.__parse_inter_df(model_json)    # var1, var2, m1, m2, coef
        
        self.__infer_var_types(train_df)
        self.__complete_luts(train_df)
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
        
        for var in self.simple_numeric:
            self.simple_pandas_luts[var] = compress_simple_lut(self.simple_pandas_luts[var])
            
        for var1, var2 in self.inter_numeric:
            # Compressing first var
            lut = compress_value1_interaction(self.inter_pandas_luts[(var1, var2)], var1, var2)
            # Compressing each sub table along second var
            subluts_compressed = list()
            for _, sublut in lut.groupby(var1, as_index=False, dropna=False):
                subluts_compressed.append(compress_simple_lut(sublut))
                
            self.inter_pandas_luts[(var1, var2)] = pd.concat(subluts_compressed)
    
    
    def __complete_luts(self, train_df: pd.DataFrame):
        """Complete numeric LUTs using training data for nearest matching.

        Args:
            train_df: Training dataframe used to complete LUTs.
        """
        if train_df is None:
            return
        
        # Simples effects
        for var in self.simple_numeric:
            lut = self.simple_pandas_luts[var]       
            train_unique_values = get_unique_values(train_df, lut, var)
            self.simple_pandas_luts[var] = complete_lut(train_unique_values, lut, var)
        
        # Interactions
        for var1, var2 in self.inter_numeric:
            lut = self.inter_pandas_luts[(var1, var2)]
            train_unique_values = get_unique_values(train_df, lut, var2)
            sublut_completed_list = list()

            for _, sublut in lut.groupby(var1, as_index=False, dropna=False):
                sublut_completed = complete_lut(train_unique_values, sublut, var2)
                sublut_completed_list.append(sublut_completed)
                
            self.inter_pandas_luts[(var1, var2)] = pd.concat(sublut_completed_list)
            
        
    def __parse_simple_df(self, model_json: dict[str, object]) -> dict[str, pd.Series]:
        """Parse simple-effect coefficients into per-variable LUTs.

        Args:
            model_json: Akur8 model JSON as a dict.
        """
        flat = flatten(model_json.get('coefficients', {}))
        if not flat:
            return None
        s = pd.Series(flat).reset_index()
        s.columns = ['var', 'mod', 'coef']
        s['var'] = s['var'].astype(str)
        s['mod'] = s['mod'].astype(str)
        s['coef'] = s['coef'].astype(float)
        if self.link == 'LOG':
            s['beta'] = np.log(s['coef'])
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
            return None
        s = pd.Series(flat, name='coef').reset_index()
        s.columns = ['var1', 'var2', 'm1', 'm2', 'coef']
        for c in ['var1', 'var2', 'm1', 'm2']:
            s[c] = s[c].astype(str)
        s['beta'] = s['coef'].astype(float)
        if self.link == 'LOG':
            s['beta'] = np.log(s['coef'])
        return {
            (v1, v2): g[['m1', 'm2', 'beta']].rename(columns={'m1': v1, 'm2': v2})
            for (v1, v2), g in s.groupby(['var1', 'var2'], sort=False, dropna=False)
        }
    
    
    def cast_to_num(self, lut: pd.DataFrame, var: str, train_df: pd.DataFrame):
        """Cast a LUT variable to numeric, matching train_df dtype when provided.

        Args:
            lut: Lookup table dataframe to mutate.
            var: Variable name to cast.
            train_df: Training dataframe for dtype matching.
        """
        if self.var_is_numeric[var]:
            lut[var] = pd.to_numeric(lut[var].str.replace(',', '.'))
            if train_df is not None:
                # When train_df is provided, we cast lut[var] in the same type as train_df[var] so merge_asof can work 
                lut[var] = lut[var].astype(train_df[var].dtype)
        
        
    def __infer_var_types(self, train_df) -> dict[str, bool]:
        """Infer variable numeric types and normalize LUT ordering.

        Args:
            train_df: Optional training dataframe.
        """
        if train_df is None:
            self.var_is_numeric = self.__infer_var_types_from_json()
        else:
            self.var_is_numeric = self.__infer_var_types_from_train_df(train_df)
        
        for var, lut in self.simple_pandas_luts.items():
            self.cast_to_num(lut, var, train_df)
            # Sort to accelerate future beta look ups
            lut.sort_values(by=[var], inplace=True)
        
        df_for_unique_count = train_df if train_df is not None else lut
        
        a_permuter = list()
        for (var1, var2), lut in self.inter_pandas_luts.items():
            self.cast_to_num(lut, var1, train_df)
            self.cast_to_num(lut, var2, train_df)
            is_num_var1 = self.var_is_numeric[var1]
            is_num_var2 = self.var_is_numeric[var2]
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
            lut = self.inter_pandas_luts.pop((var1, var2))
            self.inter_pandas_luts[(var2, var1)] = (
                lut[[var2, var1, lut.columns[2]]]
                .sort_values(by=[var2, var1])
            )
            
        # Sort to accelerate future beta look ups
        for (var1, var2), lut in self.inter_pandas_luts.items():
            lut.sort_values(by=[var1, var2], inplace=True)
            
        self.simple_numeric = [var for var in self.simple_pandas_luts.keys() if self.var_is_numeric[var]]
        # Interactions whose second feature is numeric
        self.inter_numeric = [(var1, var2) for var1, var2 in self.inter_pandas_luts.keys() if self.var_is_numeric[var2]]
            

    def __infer_var_types_from_json(self) -> dict[str, bool]:
        """Infer numeric types by checking if all values parse as floats."""
        values_by_variable: dict[str, set[str]] = {}
        
        for var, lut in self.simple_pandas_luts.items():
            values_by_variable[var] = set(lut.reset_index()[var].unique())
            
        for (var1, var2), lut in self.inter_pandas_luts.items():
            temp = lut.reset_index()
            valeurs_var1 = values_by_variable.get(var1, set())
            valeurs_var1 = valeurs_var1.union(temp[var1].unique())
            values_by_variable[var1] = valeurs_var1
            valeurs_var2 = values_by_variable.get(var2, set())
            valeurs_var2 = valeurs_var2.union(temp[var2].unique())
            values_by_variable[var2] = valeurs_var2
        
        return {var: all(is_float_key(v) for v in values) for var, values in values_by_variable.items()}
    
    
    def __infer_var_types_from_train_df(self, train_df: pd.DataFrame) -> dict[str, bool]:
        """Infer numeric types from the training dataframe dtypes.

        Args:
            train_df: Training dataframe to inspect.
        """
        variables = set(self.simple_pandas_luts.keys())
        variables = variables.union(t[0] for t in self.inter_pandas_luts.keys())
        variables = variables.union(t[1] for t in self.inter_pandas_luts.keys())
        
        numeric = train_df[list(variables)].select_dtypes(include=np.number)
        
        return {variable: variable in numeric for variable in variables}
    
    
    def __calculate_interpolators(self, interpolate: bool):
        """Precompute interpolation coefficients for numeric LUTs.

        Args:
            interpolate: Whether to compute interpolation coefficients.
        """
        self.interpolate = interpolate
        if not interpolate:
            return
        
        # Effets simples
        for var in self.simple_numeric:
            self.simple_pandas_luts[var] = self.__calculate_interpolators_for_var(self.simple_pandas_luts[var], var)
            
        # Interactions
        for var1, var2 in self.inter_numeric:
            lut = self.inter_pandas_luts[(var1, var2)]
            
            sublut_list = list()
            for val1, sublut in lut.groupby(var1, as_index=False, dropna=False):
                temp = self.__calculate_interpolators_for_var(sublut, var2)
                sublut_list.append(temp)
                
            self.inter_pandas_luts[(var1, var2)] = pd.concat(sublut_list)
    
    
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
        interpolator = PchipInterpolator(x=lut.loc[mask_not_nan, var], y=lut.loc[mask_not_nan, 'beta'])

        coefficients_pchip = (
            pd.DataFrame(interpolator.c.transpose(), columns=['pchip3', 'pchip2', 'pchip1', 'pchip0'])
            .drop(columns=['pchip0']) # Not needed because pchip0 = var
        )
        lut = pd.concat([lut, coefficients_pchip], axis=1)
        
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
        self.simple_numpy_luts = dict()
        self.inter_numpy_luts = dict()
        
        # Simple effects
        for var, lut in self.simple_pandas_luts.items():
            # Num
            if self.var_is_numeric[var]:
                self.simple_numpy_luts[var] = NumpyNumLut(lut, var)
            # Cat
            else:
                self.simple_numpy_luts[var] = NumpyCatLut(lut, var)
                
        self.simple_pandas_luts = None
                
        # Interactions
        for (var1, var2), lut in self.inter_pandas_luts.items():
            is_num_1 = self.var_is_numeric[var1]
            is_num_2 = self.var_is_numeric[var2]
            key = (var1, var2)
            # Num x Num
            if is_num_1 and is_num_2:
                self.inter_numpy_luts[key] = NumpyNumNumLut(lut, var1, var2)
            # Cat x Num
            if not is_num_1 and is_num_2:
                self.inter_numpy_luts[key] = NumpyCatNumLut(lut, var1, var2) 
            # Cat x Cat
            if not is_num_1 and not is_num_2:
                self.inter_numpy_luts[key] = NumpyCatCatLut(lut, var1, var2)
                
        self.inter_pandas_luts = None
        
        
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
        default_interpolation: str='pchip', 
        interpolation_simple: dict[str, str]=dict(), 
        interpolation_inter: dict[tuple[str, str], str]=dict()
    ) -> pd.DataFrame:
        """Score a dataframe and return input with added coefficient and prediction columns.

        Args:
            df: Input dataframe to score.
            default_interpolation: Fallback interpolation method.
            interpolation_simple: Per-variable interpolation overrides.
            interpolation_inter: Per-interaction interpolation overrides.
        """
        
        values_cache: dict[str, pd.Series] = {
            '__numeric_cols__': set(df.select_dtypes(include=np.number).columns)
        }
        out = pd.DataFrame(index=df.index)
        col_intercept = f'{self.model_name}::intercept'
        out[col_intercept] = self.intercept
        cols_to_sum = [col_intercept]
        
        for var, lut in self.simple_numpy_luts.items():
            coef_column = self.__get_col_coef_name(var)
            interpolation = interpolation_simple.get(var, default_interpolation)
            self.__check_interpolation_method(interpolation, var)
            out[coef_column] = lut.compute_betas(df, interpolation=interpolation, values_cache=values_cache)
            cols_to_sum.append(coef_column)
            
        for (var1, var2), lut in self.inter_numpy_luts.items():
            coef_column = self.__get_col_coef_name(var1, var2)
            interpolation = interpolation_inter.get((var1, var2), default_interpolation)
            self.__check_interpolation_method(interpolation, var2)
            out[coef_column] = lut.compute_betas(df, interpolation=interpolation, values_cache=values_cache)
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
            dump(self, f)
         
            
    @staticmethod
    def from_pickle(path: str):
        with open(path, 'rb') as f:
            return load(f)