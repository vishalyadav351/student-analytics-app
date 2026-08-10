import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. DASH APP CONFIGURATION
# ==========================================
app = dash.Dash(__name__)
app.title = "Universal Quantum Student Analytics Engine"
server = app.server

card_style = {
    "backgroundColor": "rgba(15, 23, 42, 0.6)",
    "border": "1px solid rgba(255, 255, 255, 0.05)",
    "borderRadius": "16px",
    "padding": "25px",
    "boxShadow": "0 12px 40px 0 rgba(0, 0, 0, 0.6)",
}

input_style = {
    "backgroundColor": "#0f172a",
    "color": "#fff",
    "border": "1px solid #334155",
    "borderRadius": "6px",
    "padding": "8px",
    "width": "100%",
    "boxSizing": "border-box"
}

# ==========================================
# 2. DYNAMIC ANALYTICS PIPELINE ENGINE
# ==========================================
def run_dynamic_analytics(df_input, current_subjects, max_marks):
    if df_input.empty or not current_subjects:
        return pd.DataFrame()

    df_processed = df_input.copy()
    
    for sub in current_subjects:
        if sub in df_processed.columns:
            df_processed[sub] = pd.to_numeric(df_processed[sub], errors='coerce').fillna(0)

    # Calculation according to Custom Max Marks
    df_processed["Total Marks"] = df_processed[current_subjects].sum(axis=1)
    max_total = len(current_subjects) * max_marks
    df_processed["Percentage"] = (df_processed["Total Marks"] / max_total) * 100
    df_processed["Grade"] = pd.cut(df_processed["Percentage"], bins=[-1, 50, 60, 75, 90, 100], labels=["F", "D", "C", "B", "A"])
    df_processed["Rank"] = df_processed["Percentage"].rank(ascending=False, method="min").astype(int)
    
    mean_val = df_processed["Percentage"].mean()
    std_val = df_processed["Percentage"].std() if df_processed["Percentage"].std() > 0 else 1
    df_processed["Z-Score"] = (df_processed["Percentage"] - mean_val) / std_val
    
    def compute_ml_clusters(pct):
        if pct < 55:
            return "High-Risk Friction Node"
        elif pct >= 82:
            return "Elite High-Potential Node"
        else:
            return "Consistent Core Performer"
            
    df_processed["AI ML Cluster Profile"] = df_processed["Percentage"].apply(compute_ml_clusters)
    
    q1 = df_processed["Percentage"].quantile(0.25)
    q3 = df_processed["Percentage"].quantile(0.75)
    iqr = q3 - q1
    df_processed["Anomaly System Flag"] = np.where(
        (df_processed["Percentage"] < (q1 - 1.5 * iqr)) | (df_processed["Percentage"] > (q3 + 1.5 * iqr)), 
        "ANOMALY DETECTED", "SYSTEM NORMAL"
    )
    
    return df_processed.sort_values("Rank")

# ==========================================
# 3. ADVANCED DASHBOARD LAYOUT
# ==========================================
app.layout = html.Div(
    style={"fontFamily": "'Inter', sans-serif", "backgroundColor": "#030712", "padding": "40px", "color": "#f3f4f6", "minHeight": "100vh"},
    children=[
        # Banner Header & Top Ranker Showcase
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "background": "linear-gradient(135deg, #1e1b4b 0%, #030712 100%)", "padding": "25px", "borderRadius": "16px", "border": "1px solid #312e81", "marginBottom": "25px"},
            children=[
                html.Div([
                    html.H1("UNIVERSAL QUANTUM STUDENT ANALYTICS PLATFORM", style={"margin": "0", "fontSize": "26px", "color": "#fff"}),
                    html.P("Dynamic Custom Subject Engine, Custom Max Marks, ML Projections & Audit Tools", style={"margin": "5px 0 0 0", "color": "#9ca3af", "fontSize": "14px"})
                ]),
                html.Div(id="topper-medal-showcase")
            ]
        ),

        # Step 1: Custom Subjects & Max Marks Setup
        html.Div(
            style={**card_style, "marginBottom": "25px", "borderLeft": "5px solid #a855f7"},
            children=[
                html.H3("⚙️ STEP 1: DEFINE CUSTOM SUBJECTS & MAXIMUM MARKS", style={"margin": "0 0 10px 0", "fontSize": "15px", "color": "#a855f7"}),
                html.P("Subjects commas se separate karke daalein aur Max Marks (e.g. 20, 50, 100) set karein:", style={"fontSize": "12px", "color": "#94a3b8"}),
                html.Div(
                    style={"display": "flex", "gap": "15px", "alignItems": "center"},
                    children=[
                        html.Div(style={"width": "60%"}, children=[
                            html.Label("Subjects:", style={"fontSize": "11px", "color": "#94a3b8"}),
                            dcc.Input(id="input-custom-subjects", type="text", value="DAA, DA, DMW, DBMS, CC, COI", style=input_style)
                        ]),
                        html.Div(style={"width": "20%"}, children=[
                            html.Label("Max Marks Per Subject:", style={"fontSize": "11px", "color": "#94a3b8"}),
                            dcc.Input(id="input-max-marks", type="number", min=1, max=1000, value=100, style=input_style)
                        ]),
                        html.Div(style={"width": "20%", "marginTop": "18px"}, children=[
                            html.Button("🔧 LOCK SUBJECTS & MARKS", id="btn-set-subjects", n_clicks=0, style={"backgroundColor": "#a855f7", "color": "#fff", "border": "none", "padding": "10px", "borderRadius": "6px", "fontWeight": "bold", "cursor": "pointer", "width": "100%"})
                        ])
                    ]
                )
            ]
        ),

        # Step 2: Dynamic Input Form
        html.Div(
            style={**card_style, "marginBottom": "30px"},
            children=[
                html.H3("➕ STEP 2: INSERT STUDENT RECORD", style={"margin": "0 0 15px 0", "fontSize": "15px", "color": "#10b981"}),
                html.Div(id="dynamic-input-fields-container"),
                html.Br(),
                html.Button("⚡ INJECT RECORD & RE-RUN PIPELINE", id="btn-add-node", n_clicks=0, style={"backgroundColor": "#10b981", "color": "#fff", "border": "none", "padding": "12px", "borderRadius": "8px", "fontWeight": "bold", "cursor": "pointer", "width": "100%"})
            ]
        ),

        # Subject Specific Topper Control Card
        html.Div(
            style={**card_style, "marginBottom": "25px", "borderLeft": "5px solid #fbbf24"},
            children=[
                html.H3("👑 SUBJECT TOPPER TELEMETRY", style={"margin": "0 0 10px 0", "fontSize": "15px", "color": "#fbbf24"}),
                html.Div(
                    style={"display": "flex", "gap": "20px", "alignItems": "center"},
                    children=[
                        html.Div(style={"width": "40%"}, children=[
                            html.Label("Select Subject to check highest score:", style={"fontSize": "12px", "color": "#94a3b8"}),
                            dcc.Dropdown(id="subject-telemetry-dropdown", clearable=False, style={"backgroundColor": "#0f172a", "color": "#000", "borderRadius": "6px"})
                        ]),
                        html.Div(id="subject-telemetry-output", style={"fontSize": "14px", "fontWeight": "bold", "color": "#fff"})
                    ]
                )
            ]
        ),

        # Analytics Tabs
        dcc.Tabs(
            id="workspace-tabs", value="macro-tab",
            parent_style={"borderBottom": "2px solid #1e293b"},
            style={"height": "48px", "marginBottom": "30px"},
            children=[
                dcc.Tab(label="📊 GLOBAL BATCH AUDIT & STATS", value="macro-tab", style={"background": "#0f172a", "color": "#6b7280", "border": "none", "fontWeight": "bold"}, selected_style={"background": "#1e293b", "color": "#06b6d4", "borderBottom": "3px solid #06b6d4", "fontWeight": "bold"}),
                dcc.Tab(label="🎯 STUDENT RADAR & AI PREDICTIVE FORECAST", value="micro-tab", style={"background": "#0f172a", "color": "#6b7280", "border": "none", "fontWeight": "bold"}, selected_style={"background": "#1e293b", "color": "#06b6d4", "borderBottom": "3px solid #06b6d4", "fontWeight": "bold"})
            ]
        ),

        html.Div(id="workspace-tab-render"),
        html.Br(),

        # Master Data Table & Custom File Export
        html.Div(
            style=card_style,
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"},
                    children=[
                        html.H3("📋 SECURE MASTER ARCHIVE DATAFRAME (✏️ EDITABLE TABLE)", style={"margin": "0", "fontSize": "15px", "color": "#fff"}),
                        html.Div(
                            style={"display": "flex", "gap": "10px", "alignItems": "center"},
                            children=[
                                dcc.Input(
                                    id="input-filename",
                                    type="text",
                                    placeholder="File Name (e.g. btech_cse)",
                                    value="btech_cse",
                                    style={**input_style, "width": "230px"}
                                ),
                                html.Button("📥 EXPORT REPORT (CSV)", id="btn-download-csv", style={"backgroundColor": "#06b6d4", "color": "#fff", "border": "none", "padding": "10px 20px", "borderRadius": "8px", "fontWeight": "bold", "cursor": "pointer"}),
                                dcc.Download(id="download-dataframe-csv"),
                            ]
                        )
                    ]
                ),
                dash_table.DataTable(
                    id="table",
                    editable=True,
                    row_deletable=True,
                    page_size=10,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "padding": "12px", "backgroundColor": "#0f172a", "color": "#9ca3af", "border": "1px solid #1e293b"},
                    style_header={"backgroundColor": "#1e293b", "color": "#06b6d4", "fontWeight": "bold"}
                )
            ]
        ),
        
        # State Stores
        dcc.Store(id="active-subjects-store", data=["DAA", "DA", "DMW", "DBMS", "CC", "COI"]),
        dcc.Store(id="max-marks-store", data=100),
        dcc.Store(id="in-memory-pipeline-store", data=[])
    ]
)

# ==========================================
# 4. INTERACTIVE CALLBACK LOGIC
# ==========================================

# 1. Subject Configuration & Max Marks Handler
@app.callback(
    [Output("active-subjects-store", "data"),
     Output("max-marks-store", "data"),
     Output("in-memory-pipeline-store", "data", allow_duplicate=True)],
    Input("btn-set-subjects", "n_clicks"),
    [State("input-custom-subjects", "value"),
     State("input-max-marks", "value")],
    prevent_initial_call=True
)
def update_subjects_and_max_marks(n_clicks, custom_input, max_m):
    max_val = int(max_m) if max_m and int(max_m) > 0 else 100
    if not custom_input:
        return ["DAA", "DA", "DMW"], max_val, []
    parsed_subjects = [s.strip() for s in custom_input.split(",") if s.strip()]
    return parsed_subjects, max_val, []

# 2. Dynamic Input Fields Generator
@app.callback(
    Output("dynamic-input-fields-container", "children"),
    [Input("active-subjects-store", "data"),
     Input("max-marks-store", "data")]
)
def generate_dynamic_inputs(active_subjects, max_m):
    fields = [
        html.Div([html.Label("Roll No:", style={"fontSize":"11px"}), dcc.Input(id="input-roll", type="text", placeholder="e.g., 34567", style=input_style)]),
        html.Div([html.Label("Full Name:", style={"fontSize":"11px"}), dcc.Input(id="input-name", type="text", placeholder="Student Name", style=input_style)])
    ]
    for sub in active_subjects:
        fields.append(
            html.Div([
                html.Label(f"{sub} (Max {max_m}):", style={"fontSize":"11px"}),
                dcc.Input(id={"type": "subject-score-input", "subject": sub}, type="number", min=0, max=max_m, style=input_style)
            ])
        )
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": f"repeat({min(len(fields), 7)}, 1fr)", "gap": "10px"},
        children=fields
    )

# 3. Insert Student Record
@app.callback(
    [Output("in-memory-pipeline-store", "data"),
     Output("input-roll", "value"),
     Output("input-name", "value"),
     Output({"type": "subject-score-input", "subject": dash.ALL}, "value")],
    Input("btn-add-node", "n_clicks"),
    [State("input-roll", "value"),
     State("input-name", "value"),
     State({"type": "subject-score-input", "subject": dash.ALL}, "value"),
     State({"type": "subject-score-input", "subject": dash.ALL}, "id"),
     State("active-subjects-store", "data"),
     State("in-memory-pipeline-store", "data")],
    prevent_initial_call=True
)
def inject_student_record(n_clicks, roll, name, score_values, score_ids, active_subs, current_data):
    empty_scores = [None] * len(score_values)
    
    if not n_clicks or not roll or not name:
        return current_data, roll, name, score_values
        
    new_entry = {"Roll No": str(roll), "Name": str(name)}
    for val, item_id in zip(score_values, score_ids):
        sub_name = item_id["subject"]
        new_entry[sub_name] = int(val if val is not None else 0)
        
    current_data.append(new_entry)
    
    return current_data, "", "", empty_scores

# 4. Handle Direct Table Editing & Sync Store
@app.callback(
    Output("in-memory-pipeline-store", "data", allow_duplicate=True),
    Input("table", "data"),
    State("active-subjects-store", "data"),
    prevent_initial_call=True
)
def update_store_from_edited_table(edited_table_data, active_subs):
    if not edited_table_data:
        return []
    
    updated_records = []
    for row in edited_table_data:
        record = {"Roll No": str(row.get("Roll No", "")), "Name": str(row.get("Name", ""))}
        for sub in active_subs:
            val = row.get(sub, 0)
            record[sub] = int(val) if str(val).isdigit() or isinstance(val, (int, float)) else 0
        updated_records.append(record)
        
    return updated_records

# 5. Master Table Columns & Data Render
@app.callback(
    [Output("table", "columns"),
     Output("table", "data")],
    [Input("in-memory-pipeline-store", "data"),
     Input("active-subjects-store", "data"),
     Input("max-marks-store", "data")]
)
def update_table_view(stored_records, active_subs, max_m):
    raw_df = pd.DataFrame(stored_records)
    processed_df = run_dynamic_analytics(raw_df, active_subs, max_m)
    
    if processed_df.empty:
        return [], []
        
    columns_config = []
    editable_cols = ["Roll No", "Name"] + active_subs
    
    for col in processed_df.columns:
        is_editable = col in editable_cols
        columns_config.append({
            "name": col.upper(), 
            "id": col, 
            "editable": is_editable
        })

    return columns_config, processed_df.to_dict("records")

# 6. Update Topper Banner & Dropdown Options
@app.callback(
    [Output("topper-medal-showcase", "children"),
     Output("subject-telemetry-dropdown", "options"),
     Output("subject-telemetry-dropdown", "value")],
    [Input("in-memory-pipeline-store", "data"),
     Input("active-subjects-store", "data"),
     Input("max-marks-store", "data")]
)
def update_banner_and_dropdown(stored_records, active_subs, max_m):
    options = [{"label": s.upper(), "value": s} for s in active_subs]
    default_val = active_subs[0] if active_subs else None
    
    df_active = run_dynamic_analytics(pd.DataFrame(stored_records), active_subs, max_m)
    
    if df_active.empty:
        topper_html = html.Div(style={"color": "#64748b", "fontSize": "12px"}, children="No data added yet")
    else:
        topper = df_active.iloc[0]
        topper_html = html.Div(
            style={"background": "rgba(6, 182, 212, 0.1)", "border": "1px solid #06b6d4", "padding": "10px 20px", "borderRadius": "12px", "textAlign": "right"},
            children=[
                html.Span("👑 OVERALL BATCH RANK #1 TOPPER", style={"fontSize": "11px", "fontWeight": "bold", "color": "#06b6d4", "display": "block"}),
                html.H3(f"{topper['Name']} (Roll: {topper['Roll No']})", style={"margin": "2px 0 0 0", "fontSize": "16px", "color": "#fff"}),
                html.Span(f"Total: {topper['Total Marks']} / {len(active_subs)*max_m} | Score: {topper['Percentage']:.2f}%", style={"fontSize": "12px", "color": "#9ca3af"})
            ]
        )
    return topper_html, options, default_val

# 7. Calculate Subject Specific Topper
@app.callback(
    Output("subject-telemetry-output", "children"),
    [Input("subject-telemetry-dropdown", "value"),
     Input("in-memory-pipeline-store", "data"),
     Input("max-marks-store", "data")]
)
def calculate_subject_telemetry(selected_sub, stored_records, max_m):
    if not stored_records or not selected_sub:
        return "Add student records to view topper."
        
    df_curr = pd.DataFrame(stored_records)
    if selected_sub not in df_curr.columns:
        return "Invalid subject."

    df_curr[selected_sub] = pd.to_numeric(df_curr[selected_sub], errors='coerce').fillna(0)
    idx_max = df_curr[selected_sub].idxmax()
    top_student = df_curr.loc[idx_max, "Name"]
    top_marks = df_curr.loc[idx_max, selected_sub]
    sub_avg = df_curr[selected_sub].mean()
    
    return html.Div([
        html.Span("👑 Highest Scorer: ", style={"color": "#fbbf24"}),
        f"{top_student} ({top_marks} / {max_m} Marks) | ",
        html.Span("⚡ Subject Avg: ", style={"color": "#06b6d4"}),
        f"{sub_avg:.1f} Marks"
    ])

# 8. Workspace Tab Render
@app.callback(
    Output("workspace-tab-render", "children"),
    [Input("workspace-tabs", "value"),
     Input("in-memory-pipeline-store", "data"),
     Input("active-subjects-store", "data"),
     Input("max-marks-store", "data")]
)
def render_tab_analytics(active_tab, stored_records, active_subs, max_m):
    raw_df = pd.DataFrame(stored_records)
    processed_df = run_dynamic_analytics(raw_df, active_subs, max_m)

    if processed_df.empty:
        return html.Div("Data add karein tab yahan Batch Analytics aur Graphs show honge.", style={"textAlign": "center", "color": "#64748b", "padding": "30px"})

    if active_tab == "macro-tab":
        sub_stats = []
        for s in active_subs:
            sub_stats.append({
                "Subject": s.upper(), 
                "Mean Score": round(processed_df[s].mean(), 2),
                "Max Score": int(processed_df[s].max()), 
                "Min Score": int(processed_df[s].min()),
                "Std Deviation": round(processed_df[s].std(), 2)
            })
        df_stats = pd.DataFrame(sub_stats)

        fig_audit_bar = go.Figure()
        fig_audit_bar.add_trace(go.Bar(x=df_stats["Subject"], y=df_stats["Mean Score"], name="Mean Score", marker_color="#06b6d4"))
        fig_audit_bar.add_trace(go.Bar(x=df_stats["Subject"], y=df_stats["Max Score"], name="Max Score", marker_color="#3b82f6"))
        fig_audit_bar.add_trace(go.Bar(x=df_stats["Subject"], y=df_stats["Min Score"], name="Min Score", marker_color="#f43f5e"))
        fig_audit_bar.update_layout(
            template="plotly_dark", barmode="group", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af", margin=dict(t=30, b=10, l=10, r=10),
            title="Cross-Subject Performance Bounds"
        )

        cohort_counts = processed_df["AI ML Cluster Profile"].value_counts().reset_index()
        cohort_counts.columns = ["Profile", "Count"]
        fig_cohort_pie = px.pie(
            cohort_counts, values="Count", names="Profile", hole=0.4,
            template="plotly_dark", color_discrete_sequence=["#3b82f6", "#10b981", "#f43f5e"],
            title="Batch Cluster Profile Vector Distribution"
        ).update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#9ca3af", margin=dict(t=30, b=10, l=10, r=10))

        return html.Div([
            html.Div(style={"display": "flex", "gap": "25px", "marginBottom": "25px"}, children=[
                html.Div(style={"width": "50%", **card_style}, children=[dcc.Graph(figure=fig_audit_bar)]),
                html.Div(style={"width": "50%", **card_style}, children=[dcc.Graph(figure=fig_cohort_pie)])
            ]),
            html.Div(style=card_style, children=[
                html.H4("📊 DESCRIPTIVE STATISTICAL AUDIT MATRIX", style={"margin": "0 0 15px 0", "fontSize": "14px", "color": "#38bdf8"}),
                dash_table.DataTable(
                    columns=[{"name": sx.upper(), "id": sx} for sx in df_stats.columns],
                    data=df_stats.to_dict("records"),
                    style_cell={"textAlign": "center", "padding": "10px", "backgroundColor": "#0f172a", "color": "#9ca3af", "border": "1px solid #1e293b"},
                    style_header={"backgroundColor": "#1e293b", "color": "#a855f7", "fontWeight": "bold"}
                )
            ])
        ])

    else:
        return html.Div(
            style={"display": "flex", "gap": "25px"},
            children=[
                html.Div(style={"width": "35%", **card_style, "borderLeft": "5px solid #38bdf8"}, children=[
                    html.H3("🎯 VECTOR INTEGRATION TARGET SEARCH", style={"marginTop": "0", "fontSize": "14px", "color": "#fff"}),
                    dcc.Dropdown(
                        id="student-dropdown",
                        options=[{"label": f"Rank #{rank:02d} | {r} — {n}", "value": r} for r, n, rank in zip(processed_df["Roll No"], processed_df["Name"], processed_df["Rank"])],
                        value=processed_df["Roll No"].iloc[0],
                        clearable=False,
                        style={"backgroundColor": "#0f172a", "color": "#000", "borderRadius": "8px"}
                    ),
                    html.Div(id="micro-stats-card-output", style={"marginTop": "20px"})
                ]),
                html.Div(style={"width": "65%", "display": "flex", "gap": "20px"}, children=[
                    html.Div(style={"width": "50%", **card_style}, children=[dcc.Graph(id="student-radar-chart")]),
                    html.Div(style={"width": "50%", **card_style, "borderLeft": "4px solid #a855f7"}, children=[
                        html.H5("PREDICTIVE VECTOR GRADIENT FORECAST", style={"color": "#a855f7", "fontSize": "11px", "fontWeight": "bold", "margin": "0 0 5px 0"}),
                        dcc.Graph(id="student-trajectory-sparkline")
                    ])
                ])
            ]
        )

# 9. Micro Individual Radar & AI Prediction Sync
@app.callback(
    [Output("student-radar-chart", "figure"),
     Output("student-trajectory-sparkline", "figure"),
     Output("micro-stats-card-output", "children")],
    [Input("student-dropdown", "value"),
     Input("in-memory-pipeline-store", "data"),
     Input("active-subjects-store", "data"),
     Input("max-marks-store", "data")],
    prevent_initial_call=False
)
def sync_micro_profile(selected_roll, stored_records, active_subs, max_m):
    df_active = run_dynamic_analytics(pd.DataFrame(stored_records), active_subs, max_m)
    
    if df_active.empty or not selected_roll or selected_roll not in df_active["Roll No"].values:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return empty_fig, empty_fig, html.Div("No Data Selected")

    student_info = df_active[df_active["Roll No"] == selected_roll].iloc[0]
    marks = [student_info[sub] for sub in active_subs]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=marks, theta=active_subs, fill='toself', name=student_info["Name"],
        fillcolor='rgba(6, 182, 212, 0.15)', line=dict(color='#06b6d4', width=3)
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_m], gridcolor="#1e293b", color="#4b5563"),
            angularaxis=dict(gridcolor="#1e293b", color="#e5e7eb")
        ),
        showlegend=False, template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"MATRIX CORE: {student_info['Name'].upper()}", font=dict(size=11, color="#06b6d4")),
        margin=dict(t=40, b=20, l=20, r=20)
    )

    simulated_next_scores = []
    np.random.seed(int(student_info['Roll No']) if str(student_info['Roll No']).isdigit() else 42)
    for mark in marks:
        weight = np.random.choice([1.05, 0.98, 1.03, 0.95])
        simulated_next_scores.append(min(max_m, max(0, int(mark * weight))))

    fig_spark = go.Figure()
    fig_spark.add_trace(go.Scatter(x=active_subs, y=marks, mode="lines+markers", name="Current Matrix", line=dict(color="#06b6d4", width=2)))
    fig_spark.add_trace(go.Scatter(x=active_subs, y=simulated_next_scores, mode="lines", name="AI Forecast Vector", line=dict(color="#a855f7", width=2, dash="dash")))
    fig_spark.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=15, b=15, l=15, r=15), font_color="#94a3b8", showlegend=True,
        legend=dict(orientation="h", y=1.1, x=1), yaxis=dict(range=[0, max_m * 1.05], gridcolor="#1e293b")
    )

    stats_html = html.Div([
        html.P([html.B("Overall Rank: "), html.Span(f"#{student_info['Rank']}", style={"color": "#fbbf24"})]),
        html.P([html.B("Percentage: "), html.Span(f"{student_info['Percentage']:.2f}%", style={"color": "#10b981"})]),
        html.P([html.B("Statistical Z-Score: "), html.Span(f"{student_info['Z-Score']:.2f}", style={"color": "#38bdf8"})]),
        html.P([html.B("AI Pipeline Cluster: "), html.Span(f"{student_info['AI ML Cluster Profile']}", style={"color": "#a855f7"})]),
        html.P([html.B("IQR Anomaly Tracker: "), html.Span(f"{student_info['Anomaly System Flag']}", style={"color": "#f43f5e" if "ANOMALY" in student_info['Anomaly System Flag'] else "#64748b"})]),
    ])

    return fig_radar, fig_spark, stats_html

# 10. Dynamic Custom File Name CSV Export Handler
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    [State("in-memory-pipeline-store", "data"),
     State("active-subjects-store", "data"),
     State("max-marks-store", "data"),
     State("input-filename", "value")],
    prevent_initial_call=True,
)
def export_dataset_csv(n_clicks, stored_records, active_subs, max_m, user_filename):
    if not stored_records:
        return None
        
    final_df = run_dynamic_analytics(pd.DataFrame(stored_records), active_subs, max_m)
    
    filename = user_filename.strip() if user_filename and user_filename.strip() else "btech_cse"
    if not filename.endswith(".csv"):
        filename += ".csv"
        
    return dcc.send_data_frame(final_df.to_csv, filename, index=False)

if __name__ == "__main__":
    app.run(debug=True)