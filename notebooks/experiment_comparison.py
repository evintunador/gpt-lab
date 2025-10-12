import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from typing import Dict, Any, List, Optional
    import json

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    from gpt_lab.results_comparator import (
        find_run_directories,
        compare_runs,
        LogSchema,
        RunResult,
        STRATEGY_REGISTRY,
    )
    return (
        LogSchema,
        STRATEGY_REGISTRY,
        compare_runs,
        find_run_directories,
        json,
        mo,
        os,
        pd,
        px,
    )


@app.cell
def _(mo):
    # UI for entering glob patterns, relative to the 'experiments' directory.
    run_patterns_input = mo.ui.text(
        value="nano_gpt/runs/*, modded_nano_gpt/runs/*",
        label="Run Globs (comma-separated)",
        full_width=True,
    )

    # Display the input form with instructions
    mo.md(
        f"""
        ## 1. Select Experiment Runs
        Enter comma-separated glob patterns to find experiment directories.
        Paths are relative to the `experiments/` directory.

        {run_patterns_input}
        """
    )
    return (run_patterns_input,)


@app.cell
def _(LogSchema, find_run_directories, mo, os, run_patterns_input):
    glob_patterns = [
        os.path.join("experiments", p.strip())
        for p in run_patterns_input.value.split(",")
        if p.strip()
    ]

    schema = LogSchema()
    found_run_dirs = find_run_directories(glob_patterns, schema=schema)
    run_rows = [{"run": os.path.basename(d), "path": d} for d in found_run_dirs]

    runs_output = (
        mo.vstack([
            mo.md(f"**Found {len(found_run_dirs)} runs**"),
            mo.ui.table(run_rows, page_size=10),
        ])
        if found_run_dirs
        else mo.md("No runs found for the given patterns.")
    )

    runs_output
    return (found_run_dirs,)


@app.cell
def _(found_run_dirs, mo, os):
    run_options = {os.path.basename(p): p for p in found_run_dirs}

    selected_runs_input = mo.ui.multiselect(
        options=run_options,
        value=list(run_options.keys()),  # preselect all by passing KEYS, not values
        label="Select runs to include",
        full_width=True,
    )
    selected_runs_input
    mo.vstack([
        mo.md("## 2. Choose Runs to Include"),
        selected_runs_input
    ])
    return run_options, selected_runs_input


@app.cell
def _(mo, run_options, selected_runs_input):
    selected_run_dirs = selected_runs_input.value
    label_for_path = {v: k for k, v in run_options.items()}
    selected_rows = [{"run": label_for_path[p], "path": p} for p in selected_run_dirs]

    mo.ui.table(selected_rows, page_size=10)
    return (selected_run_dirs,)


@app.cell
def _(json, mo):
    log_filename_input = mo.ui.text(
        value="log_rank_0.jsonl",
        label="Log filename",
        full_width=True,
    )
    step_keys_input = mo.ui.text(
        value="global_step, training_step, step, iteration, epoch",
        label="Preferred step keys (comma-separated, leftmost wins)",
        full_width=True,
    )

    metrics_editor = mo.ui.code_editor(
        language="json",
        value=json.dumps([
            {"display_name": "Train Loss", "paths": ["train_loss", "loss"], "strategy": "last_value"},
            {"display_name": "Val Loss", "paths": ["val_loss"], "strategy": "best_value", "goal": "maximize" if False else "minimize"},
            {"display_name": "HellaSwag Accuracy", "paths": ["results.accuracy"], "strategy": "best_value", "goal": "maximize"},
        ], indent=2),
        label="Metric definitions (JSON list)",
        min_height=220,
    )

    hparams_editor = mo.ui.code_editor(
        language="json",
        value=json.dumps([
            {"display_name": "Learning Rate", "paths": ["training.learning_rate", "lr"]},
            {"display_name": "Weight Decay", "paths": ["optimizer.weight_decay"]},
            {"display_name": "Batch Size", "paths": ["data.batch_size", "batch_size"]},
        ], indent=2),
        label="Hyperparameter definitions (JSON list)",
        min_height=220,
    )

    mo.vstack([
        mo.md("## 3. Configure Parsing"),
        mo.md("Adjust the log schema and definitions."),
        log_filename_input, 
        step_keys_input,
        mo.accordion({
            "Metric definitions (JSON)": metrics_editor,
            "Hyperparameter definitions (JSON)": hparams_editor,
        }),
    ])
    return hparams_editor, log_filename_input, metrics_editor, step_keys_input


@app.cell
def _(
    LogSchema,
    hparams_editor,
    json,
    log_filename_input,
    metrics_editor,
    mo,
    step_keys_input,
):
    def _parse_json_list(s: str):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    metric_definitions = _parse_json_list(metrics_editor.value)
    hparam_definitions = _parse_json_list(hparams_editor.value)

    log_schema = LogSchema(
        log_filename=log_filename_input.value.strip() or "log_rank_0.jsonl",
        preferred_step_keys=[k.strip() for k in step_keys_input.value.split(",") if k.strip()],
    )

    mo.md(f"- Parsed {len(metric_definitions)} metric defs, {len(hparam_definitions)} hparam defs.")
    return hparam_definitions, log_schema, metric_definitions


@app.cell
def _(
    STRATEGY_REGISTRY,
    compare_runs,
    hparam_definitions,
    log_schema,
    metric_definitions,
    mo,
    os,
    pd,
    selected_run_dirs,
):
    run_results = compare_runs(
        run_dirs=selected_run_dirs,
        metric_definitions=metric_definitions,
        hparam_definitions=hparam_definitions,
        schema=log_schema,
    )

    # Build metrics (long)
    metric_rows = []
    for run in run_results:
        run_name = os.path.basename(run.run_path)
        commit_hash = (run.git_info or {}).get("commit_hash", None)
        for metric_name, m in run.metrics.items():
            strat = m.strategy
            func = STRATEGY_REGISTRY.get(strat)
            agg_value = None
            if func is not None:
                if strat == "best_value":
                    agg_value = func(m.values, m.goal or "minimize")
                elif strat == "last_value":
                    agg_value = func(m.values)
                elif strat == "time_series":
                    # no aggregation here; will be plotted later
                    agg_value = None
            metric_rows.append({
                "run": run_name,
                "metric": metric_name,
                "strategy": strat,
                "goal": m.goal,
                "agg_value": agg_value,
                "values_len": len(m.values or []),
                "step_key": m.selected_step_key,
                "commit": commit_hash,
                "path": run.run_path,
            })

    metrics_long_df = pd.DataFrame(metric_rows)

    # Metrics (wide)
    if not metrics_long_df.empty:
        metrics_wide_df = (
            metrics_long_df
            .pivot_table(index=["run", "commit", "path"], columns="metric", values="agg_value", aggfunc="first")
            .reset_index()
        )
    else:
        metrics_wide_df = pd.DataFrame()

    # Hyperparameters (long)
    hparam_rows = []
    for run in run_results:
        run_name = os.path.basename(run.run_path)
        commit_hash = (run.git_info or {}).get("commit_hash", None)
        for hp_name, hp_val in (run.hyperparameters or {}).items():
            hparam_rows.append({
                "run": run_name,
                "hparam": hp_name,
                "value": hp_val,
                "commit": commit_hash,
                "path": run.run_path,
            })
    hparams_long_df = pd.DataFrame(hparam_rows)

    # Hyperparameters (wide)
    if not hparams_long_df.empty:
        hparams_wide_df = (
            hparams_long_df
            .pivot_table(index=["run", "commit", "path"], columns="hparam", values="value", aggfunc="first")
            .reset_index()
        )
    else:
        hparams_wide_df = pd.DataFrame()

    # Display
    def _rows(df: pd.DataFrame):
        return df.to_dict("records") if not df.empty else []

    summary = mo.vstack([
        mo.md("## 4. Summary of Selected Runs"),
        mo.md(f"- Parsed {len(run_results)} runs"),
        mo.ui.table(_rows(metrics_wide_df), page_size=10),
        mo.accordion({
            "Hyperparameters": mo.ui.table(_rows(hparams_wide_df), page_size=10),
        }),
    ])

    summary
    return (run_results,)


@app.cell
def _(mo, run_results):
    # Gather available metric names across selected runs
    available_metrics = sorted({
        m_name
        for run in run_results
        for m_name in (run.metrics or {}).keys()
    })

    metric_select = mo.ui.dropdown(
        options=available_metrics or ["<no metrics>"],
        value=(available_metrics[0] if available_metrics else "<no metrics>"),
        label="Metric",
        full_width=True,
    )

    mo.vstack([
        mo.md("## 5. Visualize Metrics"),
        metric_select,
    ])
    return (metric_select,)


@app.cell
def _(log_schema, metric_select, mo, run_results):
    # Determine available step keys (union across runs for selected metric)
    def _step_keys_for_metric(metric_name: str):
        keys = []
        for run in run_results:
            m = (run.metrics or {}).get(metric_name)
            if m and m.step_series:
                keys.extend(list(m.step_series.keys()))
        # preserve preference order from schema, then others
        pref = [k for k in log_schema.preferred_step_keys if k in keys]
        extra = [k for k in keys if k not in pref]
        ordered = []
        [ordered.append(k) for k in pref + extra if k not in ordered]
        return ordered or ["index"]

    step_keys = _step_keys_for_metric(metric_select.value)
    step_key_select = mo.ui.dropdown(
        options=step_keys,
        value=step_keys[0],
        label="Step key",
    )

    smooth_slider = mo.ui.slider(
        start=1, stop=51, step=2, value=1,
        label="Rolling window (odd, >=1)",
    )

    error_mode = mo.ui.dropdown(
        options={"None": "none", "Std dev": "std", "SEM": "sem"},
        value="None",
        label="Bar error bars",
    )

    mo.vstack([
        step_key_select,
        mo.hstack([smooth_slider, error_mode], justify="start"),
    ])
    return (error_mode,)


@app.cell
def _(
    STRATEGY_REGISTRY,
    error_mode,
    metric_select,
    mo,
    np,
    os,
    pd,
    px,
    run_results,
):
    # Build comparison bar data (one value per run based on strategy)
    def _bar_df(metric_name: str) -> pd.DataFrame:
        rows = []
        for run in run_results:
            m = (run.metrics or {}).get(metric_name)
            if not m:
                continue
            strat = m.strategy
            func = STRATEGY_REGISTRY.get(strat)
            agg = None
            if func:
                if strat == "best_value":
                    agg = func(m.values, m.goal or "minimize")
                elif strat == "last_value":
                    agg = func(m.values)
                elif strat == "time_series":
                    # fall back to last for comparison
                    agg = m.values[-1] if m.values else None
            if agg is None:
                continue
            # compute error statistics if requested
            err = None
            if error_mode.value == "std":
                err = float(np.std(m.values)) if m.values else None
            elif error_mode.value == "sem":
                err = float(np.std(m.values) / max(len(m.values), 1)**0.5) if m.values else None
            rows.append({
                "run": os.path.basename(run.run_path),
                "value": float(agg),
                "error": err,
                "strategy": strat,
                "goal": m.goal,
            })
        return pd.DataFrame(rows)

    bar_df = _bar_df(metric_select.value)

    # Figures (guard for empties)
    if not bar_df.empty:
        fig_bar = px.bar(bar_df, x="run", y="value", color="run", title=f"Comparison — {metric_select.value}")
        if error_mode.value != "none" and "error" in bar_df:
            fig_bar.update_traces(error_y=dict(array=bar_df["error"]))
        fig_bar.update_layout(template="plotly_white", showlegend=False, xaxis_title="", yaxis_title=metric_select.value)
    else:
        fig_bar = None

    mo.vstack([
        mo.md("### Direct comparison"),
        (fig_bar if fig_bar is not None else mo.md("_No comparison data available for this selection._")),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
