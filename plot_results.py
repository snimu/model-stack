
import ast

from tabulate import tabulate
import polars as pl
from rich import print


def load_df(file: str):
    df = pl.read_csv(file)
    cols = [
        "val_loss_stack",
        "val_losses",
        "use_first_layer",
        "use_last_layer",
        "norm_wte",
        "norm_lm_head",
        "norm_inter_model",
        "model_ids",
        "coconut_everies",
    ]
    results = {c: df[c].to_list() for c in df.columns if c in cols}
    for c in results:
        if "norm" in c:
            results[c] = ["none" if v is None else v for v in results[c]]
        else:
            results[c] = [(ast.literal_eval(v) if isinstance(v, str) else v) for v in results[c]]

    results["unique_model"] = [m[0] != m[1] for m in results["model_ids"]]
    results["mean_val_loss"] = [sum(l) / len(l) for l in results["val_losses"]]
    results["coconut_every"] = [sum(c) / len(c) for c in results["coconut_everies"]]
    results = {k: v for k, v in results.items() if k not in ("model_ids", "val_losses", "coconut_everies")}
    df = pl.DataFrame(results).sort([col for col in results.keys() if "loss" not in col])
    cols.extend(["mean_val_loss", "unique_model", "coconut_every"])
    results = {c: df[c].to_list() for c in df.columns if c in cols}
    return results
    


def tabulate_all(file: str):
    print(tabulate(load_df(file), headers="keys", tablefmt="pipe", floatfmt=".2f"))


def tabulate_layer_use(file: str):
    df = pl.DataFrame(load_df(file))
    results = dict(val_loss_stack=[], mean_val_loss=[], use_first_layer=[], use_last_layer=[], unique_model=[])
    for use_first_layer, use_last_layer, unique_model in [
        (False, False, True), (True, False, True), (False, True, True), (True, True, True),
        (False, False, False), (True, False, False), (False, True, False), (True, True, False),
    ]:
        df_loc = df.filter(pl.col("use_first_layer") == use_first_layer)
        df_loc = df.filter(pl.col("use_last_layer") == use_last_layer)
        results["val_loss_stack"].append(df_loc["val_loss_stack"].mean())
        results["mean_val_loss"].append(df_loc["mean_val_loss"].mean())
        results["use_first_layer"].append(use_first_layer)
        results["use_last_layer"].append(use_last_layer)
        results["unique_model"].append(unique_model)
    print(tabulate(results, headers="keys", tablefmt="pipe", floatfmt=".2f"))


def tabulate_norms(file: str):
    df = pl.DataFrame(load_df(file))
    results = dict(val_loss_stack=[], mean_val_loss=[], norm_wte=[], norm_lm_head=[], norm_inter_model=[], unique_model=[])
    combinations = list(set((wte, lm, inter) for wte, lm, inter in zip(df["norm_wte"], df["norm_lm_head"], df["norm_inter_model"])))
    for unique_model in [True, False]:
        for norm_wte, norm_lm_head, norm_inter_model in combinations:
            df_loc = df.filter(
                (pl.col("norm_wte") == norm_wte)
                & (pl.col("norm_lm_head") == norm_lm_head)
                & (pl.col("norm_inter_model") == norm_inter_model)
                & (pl.col("unique_model") == unique_model)
            )
            results["val_loss_stack"].append(df_loc["val_loss_stack"].mean())
            results["mean_val_loss"].append(df_loc["mean_val_loss"].mean())
            results["norm_wte"].append(norm_wte)
            results["norm_lm_head"].append(norm_lm_head)
            results["norm_inter_model"].append(norm_inter_model)
            results["unique_model"].append(unique_model)
        print(tabulate(results, headers="keys", tablefmt="pipe", floatfmt=".2f"))


def tabulate_coconut(file: str):
    df = pl.DataFrame(load_df(file))
    combinations = list(set(
        (use_first_layer, use_last_layer, coconut_every)
        for use_first_layer, use_last_layer, coconut_every
        in zip(df["use_first_layer"], df["use_last_layer"], df["coconut_every"])
    ))
    results = dict(val_loss_stack=[], mean_val_loss=[], use_first_layer=[], use_last_layer=[], coconut_every=[])
    for use_first_layer, use_last_layer, coconut_every in combinations:
        df_loc = df.filter(
            (pl.col("use_first_layer") == use_first_layer)
            & (pl.col("use_last_layer") == use_last_layer)
            & (pl.col("coconut_every") == coconut_every)
        )
        results["val_loss_stack"].append(df_loc["val_loss_stack"].mean())
        results["mean_val_loss"].append(df_loc["mean_val_loss"].mean())
        results["use_first_layer"].append(use_first_layer)
        results["use_last_layer"].append(use_last_layer)
        results["coconut_every"].append(coconut_every)
    print(tabulate(results, headers="keys", tablefmt="pipe", floatfmt=".2f"))

    



if __name__ == "__main__":
    file = "results_coconut.csv"
    tabulate_coconut(file)
