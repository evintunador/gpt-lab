import marimo

__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import glob
    import os
    return glob, go, mo, os, pd, px


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
        x_axis_dropdown_2 = None
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
        # optional second x-axis. if equals the first or "None", then 2d plot later isntead of 3d
        x_axis_dropdown_2 = mo.ui.dropdown(
            options=["None"] + x_axis_options,
            label="Optoinal second x-axis: ",
            value="None",
        )
    mo.vstack([x_axis_dropdown, x_axis_dropdown_2]) if x_axis_dropdown else None
    return x_axis_dropdown, x_axis_dropdown_2, x_axis_options


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
def _(df, dfs, mo, x_axis_dropdown, x_axis_dropdown_2, x_axis_options):
    if not dfs:
        slice_sliders_form = None
    else:
        active_axes = {x_axis_dropdown.value}
        if x_axis_dropdown_2 is not None and x_axis_dropdown_2.value != "None":
            active_axes.add(x_axis_dropdown_2.value)

        slice_dims = [x for x in x_axis_options if x not in active_axes]

        sliders = {}
        for slider in slice_dims:
            sliders[slider] = mo.ui.slider(
                steps=sorted(df[slider].unique().tolist()),
                show_value=True,
                label=f"Slice '{slider}': ",
            )

        slice_sliders_form = mo.md("\n".join([f"{{{slider}}}\n" for slider in slice_dims])).batch(**sliders).form(show_clear_button=True)
    slice_sliders_form
    return active_axes, slice_sliders_form


@app.cell
def _(active_axes, df, filters_form, slice_sliders_form):
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

    # Apply numeric slice selections, excluding active axes
    if slice_sliders_form is not None:
        for col_name, val in slice_sliders_form.value.items():
            if col_name not in active_axes and val is not None:
                filtered_df = filtered_df[filtered_df[col_name] == val]
    return (filtered_df,)


@app.cell
def _(mo):
    connect_rows = mo.ui.switch(value=True, label="Connect along x (row lines)")
    connect_cols = mo.ui.switch(value=True, label="Connect along y (column lines)")
    floor_proj = mo.ui.switch(value=True, label="Show floor projection")
    show_markers = mo.ui.switch(value=True, label="Show markers")
    marker_size = mo.ui.slider(steps=list(range(3, 17)), value=8, label="Marker size")
    line_width = mo.ui.slider(steps=list(range(1, 8)), value=2, label="Line width")
    line_opacity = mo.ui.slider(steps=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], value=0.7, label="Line opacity")
    mo.hstack([connect_rows, connect_cols, floor_proj, show_markers, marker_size, line_width, line_opacity], wrap=True)
    return (
        connect_cols,
        connect_rows,
        floor_proj,
        line_opacity,
        line_width,
        marker_size,
        show_markers,
    )


@app.cell
def _(
    connect_cols,
    connect_rows,
    filtered_df,
    floor_proj,
    go,
    line_opacity,
    line_width,
    marker_size,
    mo,
    plot_titles,
    px,
    show_markers,
    x_axis_dropdown,
    x_axis_dropdown_2,
):
    plot = None
    if filtered_df.empty or not x_axis_dropdown.value:
        mo.md("### No data to plot. Please adjust filters or select different CSVs.")
    else:
        # Determine series columns (categoricals)
        series_cols = [
            series_col for series_col in filtered_df.columns
            if series_col not in ["value", "measurement", x_axis_dropdown.value]
            and (x_axis_dropdown_2 is None or series_col != getattr(x_axis_dropdown_2, "value", None))
            and filtered_df[series_col].dtype == "object"
        ]
        plot_df = filtered_df.copy()
        color_arg = None
        if series_cols:
            plot_df["series"] = plot_df[series_cols].apply(lambda row: "\n".join(row.values.astype(str)), axis=1)
            color_arg = "series"

        # consistent color mapping
        color_discrete_map = None
        if color_arg:
            series_names = sorted(plot_df[color_arg].unique())
            colors = px.colors.qualitative.Plotly
            color_discrete_map = {series: colors[i % len(colors)] for i, series in enumerate(series_names)}

        use_3d = (
            x_axis_dropdown_2 is not None
            and x_axis_dropdown_2.value not in (None, "None")
            and x_axis_dropdown_2.value != x_axis_dropdown.value
        )

        plots = {}
        metrics = plot_df["measurement"].unique()

        for metric in plot_titles:
            if metric not in metrics:
                continue
            metric_df = plot_df[plot_df["measurement"] == metric]
            if metric_df.empty:
                continue

            if use_3d:
                fig = go.Figure()
                X = x_axis_dropdown.value
                Y = x_axis_dropdown_2.value

                # if no color field, synthesize a single series for grouping
                series_field = color_arg or "_series"
                if series_field == "_series":
                    metric_df = metric_df.copy()
                    metric_df["_series"] = "series"

                zmin = float(metric_df["value"].min())

                for series_value, g in metric_df.groupby(series_field):
                    g = g.copy()
                    series_color = (
                        color_discrete_map.get(series_value)
                        if color_discrete_map else px.colors.qualitative.Plotly[0]
                    )

                    if show_markers.value:
                        fig.add_trace(go.Scatter3d(
                            x=g[X], y=g[Y], z=g["value"],
                            mode="markers",
                            marker=dict(size=marker_size.value, color=series_color),
                            showlegend=False,
                            hovertemplate=f"{X}: %{{x}}<br>{Y}: %{{y}}<br>value: %{{z:.3f}}<extra></extra>",
                            name=str(series_value),
                        ))

                    # connect along X for each fixed Y (row lines)
                    if connect_rows.value:
                        for yv, gy in g.groupby(Y):
                            gy = gy.sort_values(X)
                            fig.add_trace(go.Scatter3d(
                                x=gy[X], y=[yv]*len(gy), z=gy["value"],
                                mode="lines",
                                line=dict(color=series_color, width=line_width.value),
                                opacity=line_opacity.value,
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            if floor_proj.value:
                                fig.add_trace(go.Scatter3d(
                                    x=gy[X], y=[yv]*len(gy), z=[zmin]*len(gy),
                                    mode="lines",
                                    line=dict(color=series_color, width=1, dash="dot"),
                                    opacity=0.3,
                                    showlegend=False,
                                    hoverinfo="skip",
                                ))

                    # connect along Y for each fixed X (column lines)
                    if connect_cols.value:
                        for xv, gx in g.groupby(X):
                            gx = gx.sort_values(Y)
                            fig.add_trace(go.Scatter3d(
                                x=[xv]*len(gx), y=gx[Y], z=gx["value"],
                                mode="lines",
                                line=dict(color=series_color, width=line_width.value),
                                opacity=line_opacity.value,
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            if floor_proj.value:
                                fig.add_trace(go.Scatter3d(
                                    x=[xv]*len(gx), y=gx[Y], z=[zmin]*len(gx),
                                    mode="lines",
                                    line=dict(color=series_color, width=1, dash="dot"),
                                    opacity=0.3,
                                    showlegend=False,
                                    hoverinfo="skip",
                                ))

                fig.update_layout(
                    title=metric,
                    margin=dict(l=30, r=30, t=40, b=30),
                    showlegend=False,
                    scene=dict(
                        xaxis=dict(showbackground=False, gridcolor="#e6e6e6", zerolinecolor="#cccccc"),
                        yaxis=dict(showbackground=False, gridcolor="#e6e6e6", zerolinecolor="#cccccc"),
                        zaxis=dict(showbackground=False, gridcolor="#e6e6e6", zerolinecolor="#cccccc"),
                        aspectmode="cube",
                    ),
                )
            else:
                fig = px.line(
                    metric_df,
                    x=x_axis_dropdown.value,
                    y="value",
                    color=color_arg,
                    title=metric,
                    markers=True,
                    color_discrete_map=color_discrete_map,
                )
                fig.update_layout(margin=dict(l=30, r=30, t=40, b=30), showlegend=False)

            plots[metric] = fig

        if not plots:
            mo.md("No metrics measured for this selection.")
        else:
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
            row1 = mo.hstack([plots.get(plot_titles[0]), plots.get(plot_titles[1])], justify="center")
            row2 = mo.hstack([plots.get(plot_titles[2]), plots.get(plot_titles[3])], justify="center")
            plot = mo.vstack([custom_legend, row1, row2], align="center")
    plot
    return


if __name__ == "__main__":
    app.run()
