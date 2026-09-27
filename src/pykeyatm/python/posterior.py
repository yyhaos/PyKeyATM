import datetime
from pykeyatm import __version__
import warnings
from .utils import *
import numpy as np
import pandas as pd

def keyATM_output(model: dict, keep: list, used_iter: list) -> dict:
    if model.get("class") and "keyATM_fitted" not in model["class"]:
        raise ValueError("`model` is not a keyATM_fitted object")

    values_iter = {}
    model["model"] = extract_full_model_name(model)

    info = {}
    info["allK"] = model["no_keyword_topics"] + len(model["keywords"])
    info["V"] = len(model["vocab"])
    info["N"] = len(model["Z"])
    info["doc_lens"] = [len(z) for z in model["Z"]]
    info["model"] = model["model"]
    info["covmodel"] = model.get("model_settings", {}).get("covariates_model", "") or ""
    info["keyATMdoc_meta"] = model["stored_values"].get("keyATMdoc_meta", None)
    info["used_iter"] = used_iter

    if model["no_keyword_topics"] > 0 and len(model["keywords"]) != 0:
        info["tnames"] = list(model["keywords_raw"].keys()) + [f"Other_{i+1}" for i in range(model["no_keyword_topics"])]
    elif model["no_keyword_topics"] > 0:
        info["tnames"] = [f"Topic_{i+1}" for i in range(model["no_keyword_topics"])]
    else:
        info["tnames"] = list(model["keywords_raw"].keys())

    theta = keyATM_output_theta(model, info)

    if model["options"].get("store_theta", False):
        values_iter["theta_iter"] = keyATM_output_theta_iter(model, info)
    res = keyATM_output_phi(model, info)
    phi = res["phi"]
    topic_counts = res["topic_counts"]
    word_counts = res["word_counts"]

    if model["model"] in ["hmm", "ldahmm"]:
        values_iter["alpha_iter"] = keyATM_output_alpha_iter_hmm(model, info)
        values_iter["time_index"] = model["model_settings"]["time_index"]
    elif model["model"] in ["base", "lda"]:
        if model["options"].get("estimate_alpha", False):
            values_iter["alpha_iter"] = keyATM_output_alpha_iter_base(model, info)

    modelfit = None
    if len(model.get("model_fit", [])) > 0:
        modelfit = pd.DataFrame(model["model_fit"]).T
        modelfit.columns = ["Iteration", "Log Likelihood", "Perplexity"]

    if model["model"] in ["base", "cov", "hmm"]:
        pi_estimated = keyATM_output_pi(model["Z"], model["S"], model["priors"]["gamma"])
    else:
        pi_estimated = None

    if "pi_vectors" in model["stored_values"]:
        values_iter["pi_iter"] = model["stored_values"]["pi_vectors"]

    if model["model"] in ["cov", "ldacov"]:
        values_iter["Lambda_iter"] = model["stored_values"]["Lambda_iter"]

    information = {
        "date_output_made": str(datetime.datetime.now()),
        "version_keyATM": __version__
    }
    if keep is None:
        keep = []

    if model["model"] in ["cov", "ldacov"]:
        if "stored_values" not in keep:
            keep.append("stored_values")
        if "model_settings" not in keep:
            keep.append("model_settings")

        pg_params = model["model_settings"].get("PG_params", {})
        for k in ["theta_last", "PG_Phi", "theta_tilda", "PG_Lambda"]:
            pg_params.pop(k, None)

    if model["model"] in ["hmm", "ldahmm"]:
        if "model_settings" not in keep:
            keep.append("model_settings")

    kept_values = {
        "doc_index_used": model["stored_values"].get("doc_index")
    }
    
    for key in keep:
        if key in model:
            kept_values[key] = model[key]

    if model["options"].get("store_theta", False) and "stored_values" in keep:
        if "Z_tables" in kept_values["stored_values"]:
            kept_values["stored_values"]["Z_tables"] = None

    if model["model"] in ["hmm", "ldahmm"]:
        values_iter["R_iter_last"] = [
            r + 1 for r in model["stored_values"]["R_iter"][-1]
        ]

    result = {
        "keyword_k": len(model["keywords"]),
        "no_keyword_topics": model["no_keyword_topics"],
        "V": len(model["vocab"]),
        "N": len(model["Z"]),
        "model": abb_model_name(model["model"]),
        "theta": theta,
        "phi": phi,
        "topic_counts": topic_counts,
        "word_counts": word_counts,
        "doc_lens": info["doc_lens"],
        "vocab": model["vocab"],
        "priors": model["priors"],
        "options": model["options"],
        "keywords_raw": model["keywords_raw"],
        "model_fit": modelfit,
        "pi": pi_estimated,
        "values_iter": values_iter,
        "information": information,
        "kept_values": kept_values
    }
    result["class"] = ["keyATM_output", model["model"], "dict"]
    return result

def keyATM_output_theta(model, info):
    model_name = model["model"]
    covmodel = info.get("covmodel", "")
    Z = model["Z"]
    allK = info["allK"]

    if model_name in ["cov", "ldacov"] and covmodel == "DirMulti":
        Lambda = model["stored_values"]["Lambda_iter"][-1]
        X = model["model_settings"]["covariates_data_use"]
        Alpha = np.exp(np.dot(X, np.array(Lambda).T))

        def posterior_z_cov(docid):
            zvec = Z[docid]
            alpha = Alpha[docid, :]
            tt = np.bincount(zvec, minlength=allK)
            posterior = (tt + alpha) / (np.sum(tt) + np.sum(alpha))
            return posterior

        theta = np.vstack([posterior_z_cov(docid) for docid in range(len(Z))])

    elif model_name in ["cov", "ldacov"] and covmodel == "PG":
        theta = np.array(model["model_settings"]["PG_params"]["theta_last"])

    elif model_name in ["base", "lda"]:
        if model["options"].get("estimate_alpha", False):
            alpha = np.array(model["stored_values"]["alpha_iter"][-1])
        else:
            alpha = np.array(model["priors"]["alpha"])

        def posterior_z_base(zvec):
            zvec = np.asarray(zvec, dtype=np.int64)
            if zvec.size:
                zvec = zvec[zvec >= 0]
            tt = np.bincount(zvec, minlength=allK)
            posterior = (tt + alpha) / (np.sum(tt) + np.sum(alpha))
            return posterior
        theta = np.vstack([posterior_z_base(z) for z in Z])
    elif model_name in ["hmm", "ldahmm"]:
        # R_iter: last sampled state per time block (0-based, shape = [num_time])
        R_iter = np.asarray(model["stored_values"]["R_iter"][-1], dtype=int)

        # time_index: doc -> time block; normalize to 0-based
        time_index = np.asarray(model["model_settings"]["time_index"], dtype=int)
        if time_index.size and time_index.min() == 1:
            time_index = time_index - 1  # 1..T -> 0..T-1

        # map each doc to its state's index (0..K-1)
        R = R_iter[time_index]  # shape = [num_doc]

        # alpha_iter last: (num_states, num_topics)
        alpha_iter = np.asarray(model["stored_values"]["alpha_iter"][-1], dtype=float)
        alphas = alpha_iter[R]  # per-doc alpha row

        def z_table(zvec):
            zvec = np.asarray(zvec, dtype=int)
            return np.bincount(zvec, minlength=allK)

        Z_table = np.vstack([z_table(z) for z in Z])
        tt = Z_table + alphas
        theta = tt / tt.sum(axis=1, keepdims=True)

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    theta_df = pd.DataFrame(theta, columns=info["tnames"])

    docnames = info.get("keyATMdoc_meta", {}).get("docnames")
    if docnames is not None:
        if theta_df.shape[0] != len(docnames):
            warnings.warn(
                "Document name count does not match theta rows.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            theta_df.index = docnames

    return theta_df

def keyATM_output_theta_iter(model, info):
    model_name = model["model"]
    covmodel = info.get("covmodel", "")

    if model_name in ["cov", "ldacov"] and covmodel == "PG":
        return model["stored_values"]["theta_PG"]

    Z_tables = model["stored_values"]["Z_tables"]
    num_docs = len(model["Z"])
    allK = info["allK"]

    if model_name in ["cov", "ldacov"] and covmodel == "DirMulti":
        def posterior_theta(x):
            Z_table = np.array(Z_tables[x])
            Lambda = np.array(model["stored_values"]["Lambda_iter"][x])
            X = np.array(model["model_settings"]["covariates_data_use"])
            Alpha = np.exp(np.dot(X, Lambda.T))
            tt = Z_table + Alpha
            theta = tt / tt.sum(axis=1, keepdims=True)
            return theta

    elif model_name in ["hmm", "ldahmm"]:
        alpha_iter_list = model["stored_values"]["alpha_iter"]

        def posterior_theta(x):
            Z_table = np.array(Z_tables[x])
            R_iter = np.asarray(
                model["stored_values"]["R_iter"][x], dtype=int
            )
            time_index = np.asarray(
                model["model_settings"]["time_index"], dtype=int
            )
            if time_index.size and time_index.min() == 1:
                time_index = time_index - 1
            R = R_iter[time_index]

            alpha_iter = np.asarray(alpha_iter_list[x], dtype=float)
            alpha_mat = alpha_iter[R]
            tt = Z_table + alpha_mat
            theta = tt / tt.sum(axis=1, keepdims=True)
            return theta

    else:  # base or lda
        alpha_iter_list = model["stored_values"]["alpha_iter"]

        def posterior_theta(x):
            Z_table = np.array(Z_tables[x])
            alpha = np.array(alpha_iter_list[x])
            alpha_broadcasted = np.broadcast_to(alpha, Z_table.shape)
            tt = Z_table + alpha_broadcasted
            theta = tt / (Z_table.sum(axis=1, keepdims=True) + np.sum(alpha))
            return theta

    theta_iter = [posterior_theta(i) for i in range(len(Z_tables))]
    return theta_iter

def keyATM_output_phi(model, info):
    W_flat = [w for doc in model["W"] for w in doc]
    Z_flat = [z for doc in model["Z"] for z in doc]

    vocab = model["vocab"]
    all_words = [vocab[i] for i in W_flat]
    all_topics = Z_flat

    model_name = model["model"]

    if model_name in ["base", "cov", "hmm"]:
        pi_estimated = keyATM_output_pi(model["Z"], model["S"], model["priors"]["gamma"])
        S_flat = [s for doc in model["S"] for s in doc]

        obj = keyATM_output_phi_calc_key(
            all_words, all_topics, S_flat, pi_estimated,
            keywords_raw=model["keywords_raw"],
            vocab=model["vocab"],
            priors=model["priors"],
            tnames=info["tnames"],
            model=model
        )
    elif model_name in ["lda", "ldacov", "ldahmm"]:
        obj = keyATM_output_phi_calc_lda(
            all_words=all_words, all_topics=all_topics,
            vocab=model["vocab"],
            priors=model["priors"]["beta"],
            tnames=info["tnames"]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return obj


def keyATM_output_phi_calc_lda(all_words, all_topics, vocab, priors, tnames):
    res_tibble = (
        pd.DataFrame({"Word": all_words, "Topic": all_topics})
        .groupby(["Topic", "Word"])
        .size()
        .reset_index(name="Count")
    )

    phi_df = (
        res_tibble.pivot(index="Topic", columns="Word", values="Count")
        .fillna(0)
    )
    phi_df = phi_df.reindex(index=range(len(tnames)), columns=vocab, fill_value=0)

    phi = phi_df.to_numpy(dtype=float)

    topic_counts = phi.sum(axis=1)
    word_counts = phi.sum(axis=0)

    if np.isscalar(priors):
        phi = phi + float(priors)
    else:
        phi = phi + np.asarray(priors, dtype=float)

    row_sums = phi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    phi = phi / row_sums

    phi_df = pd.DataFrame(phi, index=tnames, columns=vocab)
    return {"phi": phi_df, "topic_counts": topic_counts, "word_counts": word_counts}

def keyATM_output_pi(model_Z, model_S, prior):
    prior = np.asarray(prior, dtype=float)
    topics = np.asarray([z for doc in model_Z for z in doc], dtype=int)
    routes = np.asarray([s for doc in model_S for s in doc], dtype=float)
    counts = np.bincount(topics, minlength=len(prior))
    keyword_counts = np.bincount(topics, weights=routes, minlength=len(prior))
    denominator = prior.sum(axis=1) + counts
    # Regular topics have zero route priors and always use the regular route,
    # including when no tokens are currently assigned to them.
    probability = np.divide(prior[:, 0] + keyword_counts, denominator,
                            out=np.zeros(len(prior)), where=denominator > 0)
    return pd.DataFrame({"Topic": np.arange(1, len(prior) + 1),
                         "count": counts, "Proportion": 100 * probability})


def keyATM_output_phi_calc_key(all_words, all_topics, all_s, pi_estimated,
                               keywords_raw, vocab, priors, tnames, model):
    n_topics, n_words = len(tnames), len(vocab)
    word_index = {word: index for index, word in enumerate(vocab)}
    words = np.asarray([word_index[word] for word in all_words], dtype=int)
    topics = np.asarray(all_topics, dtype=int)
    routes = np.asarray(all_s, dtype=int)
    counts = np.zeros((2, n_topics, n_words), dtype=float)
    np.add.at(counts, (routes, topics, words), 1.)
    regular = counts[0] + np.asarray(priors.get("beta_s0", priors["beta"]), dtype=float)
    regular /= regular.sum(axis=1, keepdims=True)
    keyword = np.zeros((n_topics, n_words), dtype=float)
    for topic, terms in enumerate(keywords_raw.values()):
        support = np.asarray([word_index[word] for word in terms], dtype=int)
        if "beta_s1" in priors:
            prior = np.asarray(priors["beta_s1"], dtype=float)[topic, support]
        else:
            prior = priors["beta_s"]
        row = counts[1, topic, support] + prior
        keyword[topic, support] = row / row.sum()
    probability = np.asarray(pi_estimated["Proportion"], dtype=float) / 100.
    probability[len(keywords_raw):] = 0.
    phi = regular * (1. - probability[:, None]) + keyword * probability[:, None]
    return {"phi": pd.DataFrame(phi, index=tnames, columns=vocab),
            "topic_counts": counts.sum(axis=(0, 2)),
            "word_counts": counts.sum(axis=(0, 1))}


def keyATM_output_alpha_iter_hmm(model, info):
    """
    Convert stored alpha_iter (list of 2D arrays) into a long DataFrame:
    columns: ['State', 'Iteration', 'Topic', 'alpha'].
    - model['stored_values']['alpha_iter']: list of (num_states x allK) arrays
    - info['allK']: total number of topics (int)
    - info['used_iter']: list of iteration indices aligned with alpha_iter
    """
    alpha_iters = model["stored_values"]["alpha_iter"]
    allK = int(info["allK"])
    used_iter = info["used_iter"]

    out_frames = []
    for i, x in enumerate(alpha_iters):
        # ensure numpy 2D array
        arr = np.asarray(x, dtype=float)
        # columns as 1..allK (to mirror R's paste0(1:allK))
        cols = [str(k) for k in range(1, allK + 1)]
        df = pd.DataFrame(arr, columns=cols)

        # add State (1..num_states) and Iteration
        df["State"] = np.arange(1, df.shape[0] + 1, dtype=int)
        df["Iteration"] = used_iter[i]

        # wide -> long (gather)
        long_df = df.melt(
            id_vars=["State", "Iteration"],
            var_name="Topic",
            value_name="alpha"
        )
        # cast Topic to integer like R's as.integer
        long_df["Topic"] = long_df["Topic"].astype(int)

        out_frames.append(long_df)

    alpha_iter = pd.concat(out_frames, ignore_index=True)
    return alpha_iter

def keyATM_output_alpha_iter_base(model, info):
    topics = [str(i) for i in range(1, info['allK'] + 1)]

    alpha_iter_df = pd.DataFrame(
        model['stored_values']['alpha_iter'], columns=topics
    )

    alpha_iter_df['Iteration'] = info['used_iter']

    alpha_iter_long = alpha_iter_df.melt(
        id_vars='Iteration',
        var_name='Topic',
        value_name='alpha'
    )

    alpha_iter_long['Topic'] = alpha_iter_long['Topic'].astype(int)

    return alpha_iter_long

def top_words_keyATM_output(x, n=10, measure="probability", show_keyword=True):
    check_arg_type(x, expected_type="keyATM_output")
    modelname = extract_full_model_name(x)

    if measure not in ["probability", "lift"]:
        raise ValueError("`measure` must be one of ['probability', 'lift']")

    if modelname in ["lda", "ldacov", "ldahmm"]:
        show_keyword = False

    res = top_words_calc(
        n=n,
        measure=measure,
        show_keyword=show_keyword,
        theta=x["theta"],
        phi=x["phi"],
        word_counts=x["word_counts"],
        keywords_raw=x["keywords_raw"],
        vocab_map=x["vocab"],
    )
    return res


def top_words_calc(n, measure, show_keyword,
                   theta, phi, word_counts, keywords_raw,
                   vocab_map=None):
    # phi -> DataFrame
    if not isinstance(phi, pd.DataFrame):
        phi = pd.DataFrame(phi)

    # vocab_map: use x["vocab"] if provided; otherwise fall back to phi.columns
    if vocab_map is not None:
        vocab = list(vocab_map)
        if len(vocab) != phi.shape[1]:
            raise ValueError(f"len(vocab_map)={len(vocab)} != phi.shape[1]={phi.shape[1]}")
    else:
        vocab = phi.columns.to_list()

    if isinstance(keywords_raw, dict):
        keywords_raw = [keywords_raw[k] for k in sorted(keywords_raw)]

    if n is None:
        n = theta.shape[0]

    if measure == "probability":
        def measuref(xrow):
            return [vocab[i] for i in np.argsort(xrow)[::-1][:n]]
    elif measure == "lift":
        wfreq = word_counts / np.sum(word_counts)
        def measuref(xrow):
            lift = xrow / wfreq
            return [vocab[i] for i in np.argsort(lift)[::-1][:n]]
    else:
        raise ValueError("`measure` must be 'probability' or 'lift'.")

    res = np.array([measuref(row) for row in phi.to_numpy()]).T
    
    if show_keyword:
        for i in range(res.shape[1]):
            kw_i = keywords_raw[i] if i < len(keywords_raw) else []
            for row_idx in range(res.shape[0]):
                word = res[row_idx, i]
                labels = []
                for j in range(len(keywords_raw)):
                    if word in keywords_raw[j]:
                        labels.append(f"[{j+1}]")
                if word in kw_i:
                    labels = ["[✓]"]
                if labels:
                    res[row_idx, i] = f"{word}{''.join(labels)}"
    return pd.DataFrame(res, columns=[f"Topic {i+1}" for i in range(res.shape[1])])

def top_topics(x, n=2):
    check_arg_type(x, expected_type="keyATM_output")
    theta = x["theta"]

    if hasattr(theta, "values"):
        theta = theta.values

    n_docs, n_topics = theta.shape
    if n > n_topics:
        n = n_topics

    topic_names = [f"Topic_{i+1}" for i in range(n_topics)]
    res = []
    for row in theta:
        idx = np.argsort(row)[::-1][:n]
        res.append([topic_names[i] for i in idx])

    df = pd.DataFrame(res, columns=[f"Rank{i+1}" for i in range(n)])
    return df

def top_docs(x, n=10):
    """
    Show the top documents for each topic.

    Parameters:
    - x: keyATM_output object (dict-like), must contain "theta"
    - n: number of documents to show (default=10)

    Returns:
    - Pandas DataFrame with top n documents (indices) per topic
    """
    check_arg_type(x, expected_type="keyATM_output")

    theta = x["theta"]

    # if it's a DataFrame, get the numpy array
    if hasattr(theta, "values"):
        theta = theta.values

    n_docs, n_topics = theta.shape
    if n is None:
        n = n_docs

    res = {}
    for j in range(n_topics):  # each topic
        col = theta[:, j]
        idx = np.argsort(col)[::-1][:n]
        # +1 to match R (docs start from 1 instead of 0)
        res[f"Topic_{j+1}"] = (idx + 1).tolist()

    df = pd.DataFrame(res)
    return df
