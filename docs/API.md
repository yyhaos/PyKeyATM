# Core API

`keyATM_read(dtm, keep_docnames=False, check=True, split=0.0)` accepts a pandas DataFrame of nonnegative integer word counts. Columns are unique word strings; an optional `doc_id` column supplies document names. Integer-valued floats are accepted. Fractional, negative, missing, infinite, and out-of-range counts are rejected. Empty documents are omitted during fitting and their original document indices remain in `fit["kept_values"]["doc_index_used"]`.

`keyATM(docs, model, no_keyword_topics, keywords, model_settings=None, priors=None, options=None)` fits a keyword-assisted model. `keywords` maps each topic name to its keyword list. `weightedLDA(docs, model, number_of_topics, model_settings=None, priors=None, options=None)` fits the corresponding LDA model. The `model` argument is `base`, `covariates`, or `dynamic`.

Covariate models take a numeric `model_settings["covariates_data"]` matrix with one row per original document, optionally with a `covariates_formula`. The supported covariate sampler is `DirMulti`; the PG sampler is not implemented. Dynamic models require integer `time_index` labels starting at one and increasing by zero or one, plus a positive `num_states` no greater than the number of time periods.

Useful options are `seed`, `iterations`, `thinning`, `llk_per`, `verbose`, `use_weights`, and `weights_type`. Both sampling intervals must be positive integers. `use_weights=False` uses ordinary counts; supported weighting rules are `information-theory`, `information-theory-normalized`, `inv-freq`, and `inv-freq-normalized`. `verbose=False` suppresses fitting progress. Fits do not create cache files by default or modify input dictionaries. Fit outputs include pandas `theta` (document by topic) and `phi` (topic by word) matrices whose rows sum to one, `model_fit`, and `values_iter`.

See the four scripts in `examples/` for self-contained runs. For detailed arguments use the internal function docstrings in `pykeyatm.python.keyATM`.
