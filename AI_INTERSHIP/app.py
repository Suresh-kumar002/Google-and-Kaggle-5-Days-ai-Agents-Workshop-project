import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
import time
from dotenv import load_dotenv

# Import our ML and Agent modules
from ml_pipeline import run_ml_pipeline
from agent import OrbitAgent

# Page configuration
st.set_page_config(
    page_title="Orbit: Solid-State Battery Analytics & AI Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (Dark Obsidian theme)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #090B10;
        color: #F1F5F9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Header */
    .header-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(to right, #60A5FA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
    }
    
    /* Glassmorphic Cards */
    .metric-card {
        background: rgba(22, 28, 45, 0.8);
        border: 1px solid rgba(47, 58, 86, 0.8);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3B82F6;
        box-shadow: 0 8px 30px 0 rgba(59, 130, 246, 0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.075em;
    }
    .metric-value {
        font-size: 2.2rem;
        color: #FFFFFF;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .metric-desc {
        font-size: 0.78rem;
        color: #64748B;
    }
    
    /* SOH Status Tags */
    .status-tag {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        width: 100%;
        margin-top: 10px;
    }
    .status-healthy { 
        background-color: rgba(6, 78, 59, 0.4); 
        color: #34D399; 
        border: 1px solid #059669;
    }
    .status-degraded { 
        background-color: rgba(120, 53, 15, 0.4); 
        color: #FBBF24; 
        border: 1px solid #D97706;
    }
    .status-failed { 
        background-color: rgba(127, 29, 29, 0.4); 
        color: #FCA5A5; 
        border: 1px solid #DC2626;
    }
    
    /* Terminal Logs */
    .terminal-container {
        background-color: #05070D;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .terminal-header {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-weight: bold;
        font-size: 0.75rem;
        text-transform: uppercase;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
        margin-bottom: 12px;
        letter-spacing: 0.08em;
    }
    .terminal-log {
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        color: #38BDF8;
        font-size: 0.88rem;
        white-space: pre-wrap;
        max-height: 250px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_battery_model():
    model_path = 'battery_model.joblib'
    if not os.path.exists(model_path):
        with st.spinner("Training baseline Machine Learning models, please wait..."):
            run_ml_pipeline()
    return joblib.load(model_path)

# Initialize Session State
if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False
if 'agent_report' not in st.session_state:
    st.session_state.agent_report = None
if 'agent_logs' not in st.session_state:
    st.session_state.agent_logs = []

# Load models and label encoder
model_data = load_battery_model()
model_cap = model_data['model_capacity']
model_res = model_data['model_resistance']
le = model_data['label_encoder']
features = model_data['features']
model_metrics = model_data['metrics']

# --- APP LAYOUT ---
# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-title">ORBIT ANALYTICS: SOLID-STATE BATTERY INTELLIGENCE PLATFORM</div>
    <div class="header-subtitle">End-to-End AI-Agent Reasoning Loop & Degradation Predictor Pipeline</div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Interactive Degradation Predictor", 
    "🧠 Autonomous AI Agent Hub", 
    "⚙️ Model Diagnostics & Pipeline",
    "📁 Kaggle Submissions"
])

# ================= TAB 1: INTERACTIVE PREDICTOR =================
with tab1:
    st.subheader("Physical Design Parameter Simulation")
    st.markdown("Adjust the cell geometry, chemistry configurations, and operational parameters to project the degradation profile over 2000 cycles in real-time.")
    
    # 2-column simulation dashboard
    col_inputs, col_results = st.columns([1, 2])
    
    with col_inputs:
        st.write("🔧 **Cell Assembly Parameters**")
        
        etype = st.selectbox(
            "Electrolyte Composition Type",
            options=['Sulfide', 'Oxide', 'Polymer'],
            index=0,
            help="Sulfide is high-conductivity, Oxide is thermally stable, Polymer is lightweight and flexible."
        )
        
        thickness = st.slider(
            "Electrolyte Thickness (µm)",
            min_value=20.0,
            max_value=150.0,
            value=80.0,
            step=5.0,
            help="Thinner electrolytes increase energy density but speed up degradation and lithium short-circuit risks."
        )
        
        st.write("🔋 **Operating Environment**")
        
        c_rate = st.slider(
            "Charge/Discharge Rate (C-rate)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="1C charges/discharges the cell in 1 hour. Higher rates accelerate chemical side reactions and cell aging."
        )
        
        temp = st.slider(
            "Operating Temperature (°C)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=2.5,
            help="Optimal is ~30°C. Cold environments decrease ionic transport, while heat speeds up interface breakdown."
        )
        
        pressure = st.slider(
            "Applied Stack Pressure (MPa)",
            min_value=0.1,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Stack pressure maintains solid-to-solid contact. Low pressure causes contact loss; high pressure causes lithium dendrites."
        )
        
        target_cycle = st.slider(
            "Target Cycle Number",
            min_value=1,
            max_value=2000,
            value=1000,
            step=10,
            help="The cycle index to evaluate the state of health."
        )

    # Predictions
    # Encode electrolyte
    etype_encoded = le.transform([etype])[0]
    
    # Perform prediction for target cycle
    feat_df_target = pd.DataFrame(
        [[etype_encoded, thickness, c_rate, temp, pressure, target_cycle]],
        columns=['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number']
    )
    pred_cap_target = model_cap.predict(feat_df_target)[0]
    pred_res_target = model_res.predict(feat_df_target)[0]
    
    # Evaluate SOH
    if pred_cap_target >= 80.0:
        soh_label = "Healthy"
        soh_class = "status-healthy"
    elif pred_cap_target >= 70.0:
        soh_label = "Degraded"
        soh_class = "status-degraded"
    else:
        soh_label = "Failed (EOL)"
        soh_class = "status-failed"

    with col_results:
        # KPI Metric Cards Row
        card_col1, card_col2, card_col3 = st.columns(3)
        
        with card_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Capacity Retention</div>
                <div class="metric-value">{pred_cap_target:.2f}%</div>
                <div class="metric-desc">Percentage of initial cell capacity</div>
            </div>
            """, unsafe_allow_html=True)
            
        with card_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Interfacial Resistance</div>
                <div class="metric-value">{pred_res_target:.1f} Ω</div>
                <div class="metric-desc">Resistance across solid boundaries</div>
            </div>
            """, unsafe_allow_html=True)
            
        with card_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">State of Health (SOH)</div>
                <div class="status-tag {soh_class}">{soh_label}</div>
                <div class="metric-desc" style="margin-top: 15px;">Standard automotive failure limit is &lt;70%</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly degradation curve projection
        st.write("📊 **Degradation Profiles over Cycles (1 - 2000)**")
        
        # Precompute curves (cycles: 1, 50, 100, ..., 2000)
        cycles_range = np.arange(1, 2050, 50)
        curves_data = []
        for cyc in cycles_range:
            f_df = pd.DataFrame(
                [[etype_encoded, thickness, c_rate, temp, pressure, cyc]],
                columns=features
            )
            p_cap = model_cap.predict(f_df)[0]
            p_res = model_res.predict(f_df)[0]
            curves_data.append((cyc, p_cap, p_res))
            
        curves_df = pd.DataFrame(curves_data, columns=['cycle', 'capacity', 'resistance'])
        
        # Create dual-y axis Plotly chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Capacity line
        fig.add_trace(
            go.Scatter(
                x=curves_df['cycle'], 
                y=curves_df['capacity'], 
                name="Capacity Retention (%)",
                line=dict(color="#3B82F6", width=3),
                hovertemplate="%{y:.2f}% at cycle %{x}"
            ),
            secondary_y=False
        )
        
        # Resistance line
        fig.add_trace(
            go.Scatter(
                x=curves_df['cycle'], 
                y=curves_df['resistance'], 
                name="Interfacial Resistance (Ohm)",
                line=dict(color="#EF4444", width=3, dash='dash'),
                hovertemplate="%{y:.1f} Ohm at cycle %{x}"
            ),
            secondary_y=True
        )
        
        # Vertical marker for target cycle
        fig.add_vline(x=target_cycle, line_width=2, line_dash="dot", line_color="#F59E0B")
        fig.add_annotation(x=target_cycle, y=95, text=f"Target Cycle: {target_cycle}", showarrow=True, arrowhead=1, bgcolor="#F59E0B", bordercolor="#D97706", font=dict(color="#000000"))
        
        # Chart layout
        fig.update_layout(
            title_text=f"Predicted Cell Life Profile ({etype} Electrolyte, {thickness}µm, {c_rate}C, {temp}°C, {pressure}MPa)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.4)',
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#1E293B", title="Cycle Number"),
            yaxis=dict(gridcolor="#1E293B", title="Capacity Retention (%)", range=[35, 105]),
            yaxis2=dict(title="Interfacial Resistance (Ohm)", showgrid=False, range=[0, 480]),
            font=dict(color="#E2E8F0")
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: AUTONOMOUS AGENT HUB =================
with tab2:
    st.subheader("Orbit Agent Simulation Sandbox")
    st.markdown("Ask the agent to conduct an analysis. The agent works recursively through planning, searching literature, generating code, executing it locally, and synthesizing a research brief.")
    
    # Pre-set questions
    goal_options = [
        "Predict the degradation profile of Oxide electrolyte battery cell at 45 C under 3 MPa pressure after 1200 cycles.",
        "Predict the capacity retention of a Sulfide-based battery cell of 50 um thickness after 1500 cycles at 55 C under 2.0 MPa stack pressure.",
        "Search Wikipedia for solid-state batteries and write a report detailing how ionic conductivity affects cell performance.",
        "Custom query..."
    ]
    
    selected_goal_opt = st.selectbox("Select or Write a Goal Query", options=goal_options, index=0)
    
    if selected_goal_opt == "Custom query...":
        user_goal = st.text_area("Write custom goal for Orbit Agent:", value="Evaluate a Polymer-based battery cell of 120 um thickness at 20 C under 8 MPa pressure at cycle 800.")
    else:
        user_goal = selected_goal_opt
        
    iterations = st.slider("Maximum Iterations", min_value=3, max_value=8, value=5)
    
    # Run Agent Loop
    if st.button("Launch Orbit Agent Loop 🚀", key="run_agent_btn"):
        st.session_state.agent_running = True
        st.session_state.agent_logs = []
        st.session_state.agent_report = None
        
        # Run agent and collect logs
        agent = OrbitAgent()
        
        # We hook or intercept stdout or use step_logs to display steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.write("Initialising Orbit Agent session...")
        agent.initialize_session(user_goal, max_iterations=iterations)
        
        step_idx = 0
        while not agent.state.is_complete and agent.state.iteration_count < agent.state.max_iterations:
            step_idx = agent.state.iteration_count + 1
            progress_val = int((step_idx - 1) / agent.state.max_iterations * 100)
            progress_bar.progress(progress_val)
            
            # Step planning
            status_text.write(f"Iteration {step_idx}: Planning next actions...")
            plan_status = agent.plan_node()
            if plan_status["complete"]:
                break
                
            current_task = agent.state.current_task
            current_plan = list(agent.state.plan)
            
            # Action Mapping
            task_lower = current_task.lower()
            if "search wikipedia" in task_lower:
                action_name = "WebSearch"
            elif "query local knowledge base" in task_lower:
                action_name = "WebSearch"
            elif "write and execute python sandbox" in task_lower:
                action_name = "PythonSandbox"
            elif "analyze the output" in task_lower:
                action_name = "AnalyzeSandbox"
            elif "synthesize" in task_lower:
                action_name = "Synthesize"
            else:
                action_name = "SystemAction"
                
            status_text.write(f"Iteration {step_idx}: Executing {action_name} for '{current_task}'...")
            
            # Run action
            action_result = agent.act_node()
            
            # Log step
            step_log = {
                "iteration": step_idx,
                "plan": current_plan,
                "task": current_task,
                "action": action_name,
                "result": action_result,
                "memory_before": agent.state.memory_summary
            }
            
            # Update memory summary
            agent.sense_node(action_result)
            step_log["memory_after"] = agent.state.memory_summary
            
            # Append log
            st.session_state.agent_logs.append(step_log)
            agent.state.step_logs.append(step_log)
            
            # Render the log step immediately to the UI
            with st.expander(f"Step {step_idx}: {current_task} ({action_name})", expanded=True):
                st.write("**Plan**: ", current_plan)
                st.write("**Thought / Sub-task**: ", current_task)
                
                # Show source/input
                if action_name == "PythonSandbox":
                    st.write("**Generated Python Sandbox Script**:")
                    st.code(action_result, language="python") # Wait, action_result has output stdout.
                    # Actually, for PythonSandbox, the code was run. Let's show stdout.
                
                st.markdown("**Action Output / Observation**:")
                if action_name == "PythonSandbox":
                    st.code(action_result)
                else:
                    st.write(action_result)
                    
                st.write("**Updated State Memory**:")
                st.info(agent.state.memory_summary)
                
            time.sleep(0.5)
            
        progress_bar.progress(100)
        status_text.write("Agent task complete. Compiling final evaluation brief...")
        
        # Run final loop to synthesize report
        agent.plan_node()
        
        st.session_state.agent_report = agent.state.final_output
        st.session_state.agent_running = False
        st.success("Orbit Agent successfully completed the goal!")

    # Display final report if exists
    if st.session_state.agent_report:
        st.write("---")
        st.markdown("### 📄 Compiled Research Brief")
        st.markdown(st.session_state.agent_report)
        
        # Download report
        st.download_button(
            label="Download Research Report",
            data=st.session_state.agent_report,
            file_name="orbit_agent_report.md",
            mime="text/markdown"
        )

# ================= TAB 3: DIAGNOSTICS & RETRAINING =================
with tab3:
    st.subheader("Model Diagnostic & Training Dashboard")
    st.markdown("Explore feature importances, view performance scores, and retrain the underlying machine learning models on a fresh batch of simulated data.")
    
    col_met1, col_met2 = st.columns(2)
    
    with col_met1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Capacity Model Accuracy</div>
            <div class="metric-value">R² = {model_metrics['capacity']['r2']:.4f}</div>
            <div class="metric-desc">MSE: {model_metrics['capacity']['mse']:.3f} | MAE: {model_metrics['capacity']['mae']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_met2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Resistance Model Accuracy</div>
            <div class="metric-value">R² = {model_metrics['resistance']['r2']:.4f}</div>
            <div class="metric-desc">MSE: {model_metrics['resistance']['mse']:.3f} | MAE: {model_metrics['resistance']['mae']:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Feature Importances Plots
    st.write("📊 **Feature Contribution to Degradation Outcomes**")
    imp_df = pd.DataFrame(model_data['feature_importances'])
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        fig_cap_imp = go.Figure(go.Bar(
            x=imp_df['Capacity_Importance'],
            y=imp_df['Feature'],
            orientation='h',
            marker_color='#3B82F6'
        ))
        fig_cap_imp.update_layout(
            title="Capacity Retention - Feature Importance",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.4)',
            xaxis=dict(gridcolor="#1E293B", title="Relative Importance"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#E2E8F0"),
            margin=dict(l=100, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_cap_imp, use_container_width=True)
        
    with col_plot2:
        fig_res_imp = go.Figure(go.Bar(
            x=imp_df['Resistance_Importance'],
            y=imp_df['Feature'],
            orientation='h',
            marker_color='#EF4444'
        ))
        fig_res_imp.update_layout(
            title="Interfacial Resistance - Feature Importance",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.4)',
            xaxis=dict(gridcolor="#1E293B", title="Relative Importance"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#E2E8F0"),
            margin=dict(l=100, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_res_imp, use_container_width=True)
        
    # Pipeline control
    st.write("⚙️ **Pipeline Management Operations**")
    if st.button("Trigger Pipeline Training Run 🔄", key="retrain_model_btn"):
        with st.spinner("Executing pipeline: Generating dataset, applying Label Encoders, splitting data, and training Random Forest Regressors..."):
            run_ml_pipeline()
            st.cache_resource.clear()
            st.success("Pipeline retraining completed! Reloading new models...")
            st.rerun()

# ================= TAB 4: KAGGLE SUBMISSIONS =================
with tab4:
    st.subheader("Kaggle Submission Export Center")
    st.markdown("Generate predictions on the test set and format the outputs according to Kaggle competition guidelines.")
    
    if os.path.exists('test.csv'):
        test_df = pd.read_csv('test.csv')
        st.write(f"📄 **Mock Kaggle Test Set Preview** ({len(test_df)} rows):")
        st.dataframe(test_df.head(10), use_container_width=True)
    else:
        st.warning("No test set (test.csv) found. Run the ML pipeline first to generate it.")
        
    if st.button("Regenerate Kaggle Submission File (submission.csv) 📁", key="gen_sub_btn"):
        with st.spinner("Predicting capacity and resistance on test set..."):
            agent = OrbitAgent()
            agent.generate_submission_file()
            st.success("Submission file created successfully!")
            
    if os.path.exists('submission.csv'):
        sub_df = pd.read_csv('submission.csv')
        st.write("📄 **Generated Submission Preview**:")
        st.dataframe(sub_df.head(10), use_container_width=True)
        
        # Download submission.csv
        with open('submission.csv', 'rb') as f:
            st.download_button(
                label="Download submission.csv",
                data=f,
                file_name="submission.csv",
                mime="text/csv"
            )
