# predict_akur8

Akur8 model scorer with look-up tables interpolation.

## Installation (local)

```bash
pip install -e .
```

## Usage

```python
from predict_akur8 import Akur8Model, report_unknown_values

model = Akur8Model(model_json, train_df=train_df, model_name="my_model")
scored = model.predict(df)
unknowns = report_unknown_values(scored)
```

## Model initialization details

`Akur8Model(model_json, train_df=None, model_name=None, interpolate=True, compress_look_up_tables=True)`

- `model_name`: if set, it overrides `model_json["projectName"]` for all output column prefixes.
  If `None`, the JSON project name is used. Example output columns:
  `my_model::intercept`, `my_model::prediction`, `my_model::coef::age`,
  `my_model::coef::age x region`.
- `train_df`: when provided, variable types are inferred from the dataframe dtypes
  (numeric vs categorical) and numeric look-up tables are "completed" using the unique
  training values (nearest match as Akur8 does). This allows scoring values that exist in the
  training data but are not explicitly present in the JSON buckets. When
  `train_df` is omitted, types are inferred from JSON values only and no LUT
  completion is performed.
- `interpolate`: precomputes interpolation coefficients for numeric look-up tables so
  `linear` and `pchip` are available at scoring time.
- `compress_look_up_tables`: removes flat beta plateaus to speed up lookup
  without changing results.

## Pickle helpers

`to_pickle` / `from_pickle` serialize the fully prepared model (parsed look-up tables,
interpolation coefficients, compression, numpy look-up tables) for fast reload.

```python
model = Akur8Model(model_json, train_df=train_df)
model.to_pickle("model.pkl")

model2 = Akur8Model.from_pickle("model.pkl")
scored = model2.predict(df)
```

## Interpolation options

`predict` accepts a default interpolation and per-variable overrides:

```python
scored = model.predict(
    df,
    default_interpolation="pchip",
    interpolation_simple={"age": "linear"},
    interpolation_inter={("age", "duration"): "nearest"},
)
```

Supported methods for numeric features:
- `nearest`: closest grid value (no interpolation coefficients needed).
- `linear`: linear interpolation between grid points (requires `interpolate=True`).
- `pchip`: monotone cubic Hermite interpolation (requires `interpolate=True`).

Notes:
- For categorical variables, interpolation choice has no effect (direct mapping).
- For interactions, interpolation is applied on the second variable of the pair
  when it is numeric; for categorical x categorical pairs, the method is ignored.
