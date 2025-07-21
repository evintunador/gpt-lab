import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import glob
    import os
    return glob, mo, os, pd, px


@app.cell
def _(mo):
    mo.md("""# Benchmark Results Viewer""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Individual Module Hyperparameter Benchmark""")
    return


@app.cell
def _():
    plot_titles = [
        "Forward Time (ms)",
        "Backward Time (ms)",
        "Forward Peak Memory (GB)",
        "Backward Peak Memory (GB)",
    ]
    return (plot_titles,)


@app.cell
def _(glob, mo, os):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(current_dir, 'bench_results/*.csv'))

    if not csv_files:
        mo.md("No benchmark CSV files found in this directory.")

    # Create a mapping from a user-friendly name to the file path
    csv_options = {os.path.basename(f): f for f in csv_files}

    csv_selector = mo.ui.dropdown(
        options=csv_options,
        label="Select Benchmark CSV:",
        #value=csv_options.values(),
    )
    csv_selector
    return csv_options, csv_selector


@app.cell
def _(csv_selector, pd):
    if csv_selector.value:
        df = pd.read_csv(csv_selector.value)
    else:
        df = pd.DataFrame()
    return (df,)


@app.cell
def _(df, mo):
    assert not df.empty

    # Identify columns to create filters for (exclude known non-filter columns)
    cols_to_filter = [
        col for col in df.columns 
        if col not in ['value', 'measurement'] and df[col].dtype == 'object'
    ]
    print(cols_to_filter)

    # The x-axis is likely the numeric column that isn't 'value'
    x_axis_col = next((col for col in df.columns if df[col].dtype != 'object' and col != 'value'), None)
    print(x_axis_col)

    assert cols_to_filter

    filters = {
        col: mo.ui.multiselect(
            df[col].unique().tolist(), 
            label=f"Filter {col}", 
        )
        for col in cols_to_filter
    }
    print(filters)

    filters_form = mo.md(" ".join(["{" + col + "}" for col in cols_to_filter])).batch(**filters).form(show_clear_button=True)
    filters_form
    return filters_form, x_axis_col


@app.cell
def _():
    return


@app.cell
def _(df, filters_form):
    filtered_df = df.copy()
    for col, control in filters_form.value.items():
        if control is not None:
            filtered_df = filtered_df[filtered_df[col].isin(control)]
    #filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df, mo, plot_titles, px, x_axis_col):
    if filtered_df.empty or not x_axis_col:
        mo.md("No data to plot. Adjust filters or select a different CSV.")

    series_cols = [
        col
        for col in filtered_df.columns
        if col not in ["value", "measurement"] and filtered_df[col].dtype == "object"
    ]

    plot_df = filtered_df.copy()
    if series_cols:
        plot_df["series"] = plot_df[series_cols].apply(
            lambda row: "-".join(row.values.astype(str)), axis=1
        )
        color_arg = "series"
    else:
        color_arg = None

    color_discrete_map = None
    if color_arg:
        series_names = plot_df[color_arg].unique()
        colors = px.colors.qualitative.Plotly
        color_discrete_map = {
            series: colors[i % len(colors)] for i, series in enumerate(series_names)
        }

    plots = {}
    metrics = plot_df["measurement"].unique()

    for metric in plot_titles:
        if metric in metrics:
            metric_df = plot_df[plot_df["measurement"] == metric]
            if not metric_df.empty:
                fig = px.line(
                    metric_df,
                    x=x_axis_col,
                    y="value",
                    color=color_arg,
                    title=metric,
                    markers=True,
                    color_discrete_map=color_discrete_map,
                )
                fig.update_layout(
                    margin=dict(l=30, r=30, t=40, b=30), showlegend=False
                )
                plots[metric] = fig

    if not plots:
        mo.md("No metrics measured for this selection.")

    # Create a custom legend
    legend_items = []
    if color_discrete_map:
        for series, color in color_discrete_map.items():
            legend_items.append(
                mo.md(
                    f"""
                    <div style="display: flex; align-items: center; margin-right: 15px;">
                        <div style="width: 12px; height: 12px; background-color: {color}; margin-right: 5px;"></div>
                        <span>{series}</span>
                    </div>
                    """
                )
            )
    custom_legend = mo.hstack(legend_items, justify="center")

    # Display plots in a 2x2 grid, handling missing plots gracefully
    row1 = mo.hstack(
        [plots.get(plot_titles[0]), plots.get(plot_titles[1])], justify="center"
    )
    row2 = mo.hstack(
        [plots.get(plot_titles[2]), plots.get(plot_titles[3])], justify="center"
    )
    mo.vstack([custom_legend, row1, row2])
    return


@app.cell
def _(mo):
    mo.md(r"""## Cross-File Module Comparison""")
    return


@app.cell
def _(csv_options, mo):
    csv_multiselector = mo.ui.multiselect(
        options=csv_options,
        label="Select Benchmark CSV:",
        #value=csv_options.values(),
    )
    csv_multiselector
    return (csv_multiselector,)


@app.cell
def _(csv_multiselector, pd):
    if csv_multiselector.value:
        dfs = [pd.read_csv(val) for val in csv_multiselector.value]
    else:
        dfs = []
    return (dfs,)


@app.cell
def _(csv_multiselector, dfs, mo, os, pd):
    if not dfs:
        df_ = pd.DataFrame()
        filters_form_ = mo.md("No CSVs selected. Please select one or more benchmark CSVs from the dropdown above.")
        x_axis_col_ = None
    else:
        processed_dfs = []
        for path, single_df in zip(csv_multiselector.value, dfs):
            # Extract module name from filename, e.g., 'MLP_mps.csv' -> 'MLP'
            module_name = os.path.basename(path).split('_')[0]
            single_df['module'] = module_name
            processed_dfs.append(single_df)

        # Concatenate all dataframes. Pandas handles mismatched columns by filling with NaN.
        df_ = pd.concat(processed_dfs, ignore_index=True)

        # Identify columns to create filters for. This will now include 'module'.
        cols_to_filter_ = [
            col_ for col_ in df_.columns 
            if col_ not in ['value', 'measurement', 'module'] and df_[col_].dtype == 'object'
        ]

        # The x-axis is likely the numeric column that isn't 'value'
        x_axis_col_ = next((col_ for col_ in df_.columns if df_[col_].dtype != 'object' and col_ != 'value'), None)

        # Create filters. We explicitly handle NaN values by converting them to a
        # selectable 'N/A' string in the filter options.
        filters_ = {}
        for col_ in cols_to_filter_:
            # Get unique values, filling NaNs with a string 'N/A'
            options_ = sorted(df_[col_].fillna('N/A').unique().tolist(), key=str)
            filters_[col_] = mo.ui.multiselect(
                options=options_,
                label=f"Filter {col_}",
                # Default to selecting all available options
                value=options_,
            )

        filters_form_ = mo.md(" ".join([f"{{{col_}}}" for col_ in cols_to_filter_])).batch(**filters_).form(show_clear_button=True)
    filters_form_
    return df_, filters_form_, x_axis_col_


@app.cell
def _(df_, filters_form_):
    """
    filtered_df__ = df_.copy()
    for col__, control__ in filters_form_.value.items():
        if control__ is not None:
            filtered_df__ = filtered_df__[filtered_df__[col__].isin(control__)]
    """

    # Start with a copy of the merged dataframe to apply filters to.
    filtered_df__ = df_.copy()

    # The `filters_form_.value` holds the current selections from the UI.
    # It's a dict like {'column_name': ['value1', 'N/A']}.
    for col__, selected_options__ in filters_form_.value.items():

        # Only apply a filter if the user has selected any options for it.
        if selected_options__:

            # Check if the user wants to include rows where this parameter is not applicable.
            include_na__ = 'N/A' in selected_options__

            # Get the list of actual parameter values the user selected.
            standard_options__ = [opt__ for opt__ in selected_options__ if opt__ != 'N/A']

            # Case 1: User selected both 'N/A' and other values.
            if include_na__ and standard_options__:
                # Keep rows where the column's value is in the list OR the value is NaN.
                filtered_df__ = filtered_df__[
                    filtered_df__[col__].isin(standard_options__) | filtered_df__[col__].isna()
                ]

            # Case 2: User selected only standard values.
            elif standard_options__:
                filtered_df__ = filtered_df__[filtered_df__[col__].isin(standard_options__)]

            # Case 3: User selected only 'N/A'.
            elif include_na__:
                filtered_df__ = filtered_df__[filtered_df__[col__].isna()]

    # The output of this cell is `filtered_df_`.
    # It is now correctly filtered and ready for plotting in the next cell.
    return (filtered_df__,)


@app.cell
def _(filtered_df__, mo, plot_titles, px, x_axis_col_):
    # This code assumes `filtered_df_` (the filtered DataFrame) and 
    # `x_axis_col_` (the name of the x-axis column) are available from previous cells.
    plot_ = None
    if filtered_df__.empty or not x_axis_col_:
        # Use mo.md to display a message if there's nothing to plot.
        mo.md("### No data to plot. Please adjust filters or select different CSVs.")
    else:
        # Identify columns to use for creating the plot series. This automatically
        # includes the new 'module' column as well as other parameters.
        series_cols_ = [
            col___
            for col___ in filtered_df__.columns
            if col___ not in ["value", "measurement", x_axis_col_] and filtered_df__[col___].dtype == "object"
        ]

        plot_df_ = filtered_df__.copy()

        # Create the 'series' column for coloring the plots.
        # It will look like 'MLP-relu-float16', 'GatedMLP-relu-float16', etc.
        # The .astype(str) gracefully handles any lingering 'nan' values for the legend.
        if series_cols_:
            plot_df_["series"] = plot_df_[series_cols_].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
            color_arg_ = "series"
        else:
            color_arg_ = None

        # --- Plotting Logic (largely unchanged) ---
        color_discrete_map_ = None
        if color_arg_:
            # Sort series names for a consistent legend order
            series_names_ = sorted(plot_df_[color_arg_].unique())
            colors_ = px.colors.qualitative.Plotly
            color_discrete_map_ = {
                series: colors_[i % len(colors_)] for i, series in enumerate(series_names_)
            }

        plots_ = {}
        metrics_ = plot_df_["measurement"].unique()

        for metric_ in plot_titles:
            if metric_ in metrics_:
                metric_df_ = plot_df_[plot_df_["measurement"] == metric_]
                if not metric_df_.empty:
                    fig_ = px.line(
                        metric_df_,
                        x=x_axis_col_, # Use the new x-axis variable
                        y="value",
                        color=color_arg_,
                        title=metric_,
                        markers=True,
                        color_discrete_map=color_discrete_map_,
                    )
                    fig_.update_layout(
                        margin=dict(l=30, r=30, t=40, b=30), showlegend=False
                    )
                    plots_[metric_] = fig_

        if not plots_:
            mo.md("No metrics measured for this selection.")
        else:
            # Create a custom legend
            legend_items_ = []
            if color_discrete_map_:
                for series_, color_ in color_discrete_map_.items():
                    legend_items_.append(
                        mo.md(
                            f"""
                            <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
                                <div style="width: 12px; height: 12px; background-color: {color_}; margin-right: 5px; border-radius: 2px;"></div>
                                <span style="font-size: 0.9em;">{series_}</span>
                            </div>
                            """
                        )
                    )
            custom_legend_ = mo.hstack(legend_items_, justify="center", wrap=True)

            # Display plots in a 2x2 grid, handling missing plots gracefully
            row1_ = mo.hstack(
                [plots_.get(plot_titles[0]), plots_.get(plot_titles[1])], justify="center"
            )
            row2_ = mo.hstack(
                [plots_.get(plot_titles[2]), plots_.get(plot_titles[3])], justify="center"
            )
            plot_ = mo.vstack([custom_legend_, row1_, row2_], align="center")
    plot_
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
