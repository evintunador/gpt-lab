import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    TODO: 

    - [ ] change unused x-axes to sliders
    - [ ] make legend names separator \n instead of -
    - [ ] change x-axis to multiselect & implement 3d plot
    """
    )
    return


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
def _(glob, mo, os):
    plot_titles = [
        "Forward Time (ms)",
        "Backward Time (ms)",
        "Forward Peak Memory (GB)",
        "Backward Peak Memory (GB)",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(current_dir, 'bench_results/*.csv'))

    if not csv_files:
        mo.md("No benchmark CSV files found in this directory.")

    # Create a mapping from a user-friendly name to the file path
    csv_options = {os.path.basename(f): f for f in sorted(csv_files)}
    return csv_options, plot_titles


@app.cell
def _(csv_options, mo):
    csv_multiselector = mo.ui.multiselect(
        options=csv_options,
        label="Select Benchmark CSV:",
        value=list(csv_options.keys())[:1],
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
def _(csv_multiselector, dfs, os, pd):
    if not dfs:
        df = pd.DataFrame()
    else:
        processed_dfs = []
        for path, single_df in zip(csv_multiselector.value, dfs):
            # Extract module name from filename, e.g., 'MLP_mps.csv' -> 'MLP'
            module_name = os.path.basename(path).split('_')[0]
            single_df['module'] = module_name
            processed_dfs.append(single_df)
        del path, single_df

        # Concatenate all dataframes. Pandas handles mismatched columns by filling with NaN.
        df = pd.concat(processed_dfs, ignore_index=True)
    return (df,)


@app.cell
def _(df, dfs, mo):
    if not dfs:
        x_axis_options = []
        x_axis_dropdown = None
    else:
        x_axis_options = [
            option for option in df.columns 
            if (
                df[option].dtype not in ['object', 'bool'] 
                and option != 'value'
            )
        ]
        x_axis_dropdown = mo.ui.dropdown(
            options=x_axis_options,
            label="Select x-axis for measurement:",
            value=x_axis_options[0],
        )
    x_axis_dropdown
    return x_axis_dropdown, x_axis_options


@app.cell
def _(df, dfs, mo):
    if not dfs:
        filters_form = mo.md("No CSVs selected. Please select one or more benchmark CSVs from the dropdown above.")
    else:
        # Identify columns to create filters for. This will now include 'module'.
        cols_to_filter = [
            col for col in df.columns 
            if (
                col not in ['value', 'measurement', 'module'] 
                and df[col].dtype == 'object'
            )
        ]

        # We explicitly handle NaN values by converting them to a selectable 'N/A' string in the filter options.
        filters = {}
        for col in cols_to_filter:
            # Get unique values, filling NaNs with a string 'N/A'
            options_ = sorted(df[col].fillna('N/A').unique().tolist(), key=str)
            filters[col] = mo.ui.multiselect(
                options=options_,
                label=f"Filter '{col}': ",
                value=options_[:1],
            )

        filters_form = mo.md("\n".join([f"{{{col}}}\n" for col in cols_to_filter])).batch(**filters).form(show_clear_button=True)
    filters_form
    return (filters_form,)


@app.cell
def _(df, dfs, mo, x_axis_dropdown, x_axis_options):
    if not dfs:
        slice_sliders_form = None
    else:
        slice_dims = [x for x in x_axis_options if x != x_axis_dropdown.value]

        sliders = {}
        for slider in slice_dims:
            lo = df[slider].min()
            hi = df[slider].max()
            sliders[slider] = mo.ui.slider(
                steps=sorted(df[slider].unique().tolist()),
                show_value=True,
                label=f"Slice '{slice}': ",
            )

        slice_sliders_form = mo.md("\n".join([f"{{{slider}}}\n" for slider in slice_dims])).batch(**sliders).form(show_clear_button=True)
    slice_sliders_form
    return


@app.cell
def _(df, filters_form):
    # Start with a copy of the merged dataframe to apply filters to.
    filtered_df = df.copy()

    # The `filters_form_.value` holds the current selections from the UI.
    # It's a dict like {'column_name': ['value1', 'N/A']}.
    for column, selected_options in filters_form.value.items():

        # Only apply a filter if the user has selected any options for it.
        if selected_options:

            # Check if the user wants to include rows where this parameter is not applicable.
            include_na = 'N/A' in selected_options

            # Get the list of actual parameter values the user selected.
            standard_options = [opt for opt in selected_options if opt != 'N/A']

            # Case 1: User selected both 'N/A' and other values.
            if include_na and standard_options:
                # Keep rows where the column's value is in the list OR the value is NaN.
                filtered_df = filtered_df[
                    filtered_df[column].isin(standard_options) | filtered_df[column].isna()
                ]

            # Case 2: User selected only standard values.
            elif standard_options:
                filtered_df = filtered_df[filtered_df[column].isin(standard_options)]

            # Case 3: User selected only 'N/A'.
            elif include_na:
                filtered_df = filtered_df[filtered_df[column].isna()]
    return (filtered_df,)


@app.cell
def _(filtered_df, mo, plot_titles, px, x_axis_dropdown):
    plot = None
    if filtered_df.empty or not x_axis_dropdown.value:
        # Use mo.md to display a message if there's nothing to plot.
        mo.md("### No data to plot. Please adjust filters or select different CSVs.")
    else:
        # Identify columns to use for creating the plot series. This automatically
        # includes the new 'module' column as well as other parameters.
        series_cols = [
            series_col for series_col in filtered_df.columns
            if (
                series_col not in ["value", "measurement", x_axis_dropdown.value] 
                and filtered_df[series_col].dtype == "object"
            )
        ]

        plot_df = filtered_df.copy()

        # Create the 'series' column for coloring the plots.
        # It will look like 'MLP-relu-float16', 'GatedMLP-relu-float16', etc.
        # The .astype(str) gracefully handles any lingering 'nan' values for the legend.
        if series_cols:
            plot_df["series"] = plot_df[series_cols].apply(
                lambda row: "-".join(row.values.astype(str)), axis=1
            )
            color_arg = "series"
        else:
            color_arg = None

        # --- Plotting Logic (largely unchanged) ---
        color_discrete_map = None
        if color_arg:
            # Sort series names for a consistent legend order
            series_names = sorted(plot_df[color_arg].unique())
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
                        x=x_axis_dropdown.value,
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
        del metric

        if not plots:
            mo.md("No metrics measured for this selection.")
        else:
            # Create a custom legend
            legend_items = []
            if color_discrete_map:
                for series, color in color_discrete_map.items():
                    legend_items.append(
                        mo.md(
                            f"""
                            <div style="display: flex; align-items: center; margin-right: 15px; margin-bottom: 5px;">
                                <div style="width: 12px; height: 12px; background-color: {color}; margin-right: 5px; border-radius: 2px;"></div>
                                <span style="font-size: 0.9em;">{series}</span>
                            </div>
                            """
                        )
                    )
            custom_legend = mo.hstack(legend_items, justify="center", wrap=True)

            # Display plots in a 2x2 grid, handling missing plots gracefully
            row1 = mo.hstack(
                [plots.get(plot_titles[0]), plots.get(plot_titles[1])], justify="center"
            )
            row2 = mo.hstack(
                [plots.get(plot_titles[2]), plots.get(plot_titles[3])], justify="center"
            )
            plot = mo.vstack([custom_legend, row1, row2], align="center")
    plot
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
