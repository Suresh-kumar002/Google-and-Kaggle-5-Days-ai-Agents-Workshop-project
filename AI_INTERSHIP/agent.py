import os
import sys
import time
import subprocess
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field
import wikipedia
from dotenv import load_dotenv

# Ensure wikipedia doesn't throw too many warnings
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

@dataclass
class AgentState:
    goal: str
    plan: List[str] = field(default_factory=list)
    current_task: str = ""
    memory_summary: str = ""
    iteration_count: int = 0
    max_iterations: int = 6
    is_complete: bool = False
    final_output: str = ""
    step_logs: List[Dict[str, Any]] = field(default_factory=list)

class OrbitAgent:
    def __init__(self):
        self.state = None
        self.knowledge_base = {
            "solid-state battery": "Solid-state batteries use solid electrolytes instead of liquid. Key metrics: energy density (400-500 Wh/kg), cycle life (1000+ cycles), safety (non-flammable).",
            "ionic conductivity": "Ionic conductivity describes how fast lithium ions move. Sulfide electrolytes show promising conductivity (10^-3 to 10^-2 S/cm) at room temperature. Oxide offers better stability but lower conductivity (~10^-4 S/cm).",
            "interfacial resistance": "Interfacial resistance is the resistance at the boundary between solid electrode and solid electrolyte. It remains the primary bottleneck (~15-100 Ω·cm²). Oxide coating layers can reduce degradation.",
            "stack pressure": "Solid-state batteries require stack pressure (typically 1 to 5 MPa) to maintain solid-solid contact. Too low pressure leads to contact loss and fast degradation. Too high pressure leads to lithium dendrite growth."
        }
        
        # Initialize Gemini API
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.use_llm = False
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai_model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_llm = True
                print("[INFO] Gemini API initialized successfully. Agent running in LLM mode.")
            except Exception as e:
                print(f"[WARN] Failed to initialize Gemini API ({str(e)}). Falling back to No-API mode.")
                self.use_llm = False
        else:
            print("[INFO] GEMINI_API_KEY not found in environment. Agent running in No-API fallback mode.")

    def initialize_session(self, user_goal: str, max_iterations: int = 6):
        self.state = AgentState(goal=user_goal, max_iterations=max_iterations)

    def _call_llm(self, prompt: str) -> str:
        if not self.use_llm:
            return ""
        try:
            response = self.genai_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[WARN] Gemini API call failed: {str(e)}. Falling back to local reasoning engine.")
            self.use_llm = False  # Temporary disable for this session
            return ""

    def _parse_parameters_from_goal(self, goal: str) -> Dict[str, Any]:
        """
        Parses parameters like electrolyte type, cycles, temp, pressure from the natural language goal.
        """
        params = {
            'electrolyte_type': 'Sulfide',
            'electrolyte_thickness_um': 80.0,
            'c_rate': 1.0,
            'temperature_c': 25.0,
            'pressure_mpa': 5.0,
            'cycle_number': 1000
        }
        
        goal_lower = goal.lower()
        
        # Electrolyte type
        if 'oxide' in goal_lower:
            params['electrolyte_type'] = 'Oxide'
        elif 'polymer' in goal_lower:
            params['electrolyte_type'] = 'Polymer'
        elif 'sulfide' in goal_lower:
            params['electrolyte_type'] = 'Sulfide'
            
        # Electrolyte thickness
        thick_match = re.search(r'(\d+)\s*(?:um|µm|micrometer|thickness)', goal_lower)
        if thick_match:
            params['electrolyte_thickness_um'] = float(thick_match.group(1))
            
        # Cycles
        cycles_match = re.search(r'(\d+)\s*(?:cycle|iter)', goal_lower)
        if cycles_match:
            params['cycle_number'] = int(cycles_match.group(1))
            
        # Temperature
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:°c\b|c\b|deg\b|degree\b)', goal_lower)
        if temp_match:
            params['temperature_c'] = float(temp_match.group(1))
            
        # Pressure
        pres_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mpa\b|bar\b|pressure\b)', goal_lower)
        if pres_match:
            params['pressure_mpa'] = float(pres_match.group(1))
            
        # C-rate
        crate_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:c-rate\b|crate\b|c-rate\b|c\b)', goal_lower)
        if crate_match:
            # Avoid matching '°C' as C-rate
            if not goal_lower[max(0, crate_match.start()-2):crate_match.start()].strip().endswith('°'):
                params['c_rate'] = float(crate_match.group(1))
                
        return params

    def plan_node(self) -> Dict[str, Any]:
        if not self.state:
            return {"status": "error", "complete": False}
        
        if self.use_llm:
            prompt = f"""
You are OrbitAgent, an autonomous battery research assistant.
Your goal: {self.state.goal}
Current iteration: {self.state.iteration_count + 1} of {self.state.max_iterations}
Current memory summary of previous actions: {self.state.memory_summary}

Your available tools:
1. WebSearch(query): Searches wikipedia & local knowledge base for concepts.
2. PythonSandbox(code): Runs python code. Note that a trained Random Forest model predicting capacity retention and interfacial resistance is available in 'battery_model.joblib'.
3. Synthesize(report): Writes the final summary report.

Decide what tool to execute next. 
If the goal is fully achieved and you have the final results, output:
GOAL_ACHIEVED: <your final comprehensive report in markdown>

Otherwise, output in this exact format:
THOUGHT: <your reasoning about what information is needed next>
PLAN: [<step 1>, <step 2>, ...]
NEXT_TASK: <the next task name, matching one of: 'Search Wikipedia for <topic>', 'Write and execute Python sandbox script to query trained battery ML model', 'Analyze the output of Python sandbox predictions', 'Synthesize final comprehensive report'>
"""
            llm_out = self._call_llm(prompt)
            if llm_out:
                if "GOAL_ACHIEVED" in llm_out:
                    self.state.is_complete = True
                    self.state.final_output = llm_out.split("GOAL_ACHIEVED:")[-1].strip()
                    return {"status": "complete", "complete": True}
                
                # Parse thought, plan, task
                thought_match = re.search(r'THOUGHT:\s*(.*?)\n', llm_out)
                plan_match = re.search(r'PLAN:\s*\[(.*?)\]', llm_out)
                task_match = re.search(r'NEXT_TASK:\s*(.*?)(?:\n|$)', llm_out)
                
                if task_match:
                    self.state.current_task = task_match.group(1).strip()
                if plan_match:
                    self.state.plan = [p.strip().strip("'\"") for p in plan_match.group(1).split(",")]
                
                if self.state.current_task:
                    return {"status": "planned", "complete": False}

        # Fallback to local rule-based planning
        goal = self.state.goal
        it = self.state.iteration_count
        
        if "Final report synthesized" in self.state.memory_summary or it >= self.state.max_iterations - 1:
            self.state.is_complete = True
            self.state.final_output = self.state.memory_summary
            return {"status": "complete", "complete": True}

        is_prediction_query = any(w in goal.lower() for w in ['predict', 'degradation', 'retention', 'capacity', 'resistance', 'cycle', 'model', 'forecast'])
        
        if is_prediction_query:
            if it == 0:
                self.state.plan = ["Search for general solid-state battery degradation mechanisms", "Write Python code to load ML model and predict performance", "Execute ML prediction in sandbox", "Synthesize report"]
                self.state.current_task = "Search Wikipedia for solid-state battery degradation"
            elif it == 1:
                self.state.plan = ["Write Python code to load ML model and predict performance", "Execute ML prediction in sandbox", "Synthesize report"]
                self.state.current_task = "Write and execute Python sandbox script to query trained battery ML model"
            elif it == 2:
                self.state.plan = ["Analyze ML prediction results", "Synthesize report"]
                self.state.current_task = "Analyze the output of Python sandbox predictions"
            else:
                self.state.plan = ["Synthesize report"]
                self.state.current_task = "Synthesize final comprehensive report"
        else:
            if it == 0:
                self.state.plan = ["Search Wikipedia for solid-state battery cells", "Query local knowledge base for battery specs", "Synthesize report"]
                self.state.current_task = "Search Wikipedia for solid-state battery"
            elif it == 1:
                self.state.plan = ["Query local knowledge base for battery specs", "Synthesize report"]
                self.state.current_task = "Query local knowledge base for ionic conductivity and pressure limits"
            else:
                self.state.plan = ["Synthesize report"]
                self.state.current_task = "Synthesize final comprehensive report"

        return {"status": "planned", "complete": False}

    def act_node(self) -> str:
        task = self.state.current_task
        task_lower = task.lower()
        
        if "search wikipedia" in task_lower:
            search_query = self.state.goal
            for term in ["predict", "degradation", "retention", "capacity", "resistance", "what is", "search for"]:
                search_query = search_query.replace(term, "")
            search_query = search_query.strip()
            
            if not search_query:
                search_query = "Solid-state battery"
                
            return self._web_search(search_query)
            
        elif "query local knowledge base" in task_lower:
            return self._query_kb(self.state.goal)
            
        elif "write and execute python sandbox" in task_lower:
            params = self._parse_parameters_from_goal(self.state.goal)
            
            if self.use_llm:
                prompt = f"""
Write a Python script that loads the trained machine learning model from 'battery_model.joblib' and uses it to make predictions.
Model dictionary key details:
- 'model_capacity': scikit-learn regressor for capacity retention prediction
- 'model_resistance': scikit-learn regressor for interfacial resistance prediction
- 'label_encoder': label encoder for electrolyte_type
- 'features': feature names list: ['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number']

Goal parameters:
- Electrolyte Type: '{params['electrolyte_type']}'
- Thickness: {params['electrolyte_thickness_um']} um
- C-rate: {params['c_rate']} C
- Temperature: {params['temperature_c']} °C
- Stack Pressure: {params['pressure_mpa']} MPa
- Target Cycle Number: {params['cycle_number']}

Your Python script should:
1. Load 'battery_model.joblib' using joblib.
2. Prepare a pandas DataFrame with features in correct order: ['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number'].
3. Note that 'electrolyte_type' must be encoded using the loaded label_encoder: `le.transform(['{params['electrolyte_type']}'])[0]`.
4. Run predictions for the target cycle number and print them clearly.
5. Generate a projection over cycles [1, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] and print a table containing columns Cycle | Capacity Retention (%) | Resistance (Ohm).

Output ONLY the raw Python code block enclosed in ```python and ```.
"""
                code_out = self._call_llm(prompt)
                code_match = re.search(r'```python\n(.*?)```', code_out, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    return self._python_sandbox(code)
            
            # Default fallback code if not using LLM or LLM code generation failed
            code = f"""
import joblib
import pandas as pd
import numpy as np

# Load trained model
model_data = joblib.load('battery_model.joblib')
model_cap = model_data['model_capacity']
model_res = model_data['model_resistance']
le = model_data['label_encoder']

# Features
etype = '{params['electrolyte_type']}'
thick = {params['electrolyte_thickness_um']}
crate = {params['c_rate']}
temp = {params['temperature_c']}
pres = {params['pressure_mpa']}
cycle = {params['cycle_number']}

# Encode electrolyte type
etype_encoded = le.transform([etype])[0]

# Prepare feature dataframe
features_df = pd.DataFrame([[etype_encoded, thick, crate, temp, pres, cycle]], 
                           columns=['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number'])

# Run predictions
predicted_cap = model_cap.predict(features_df)[0]
predicted_res = model_res.predict(features_df)[0]

print("--- PREDICTION RESULTS ---")
print(f"Electrolyte Type: {{etype}}")
print(f"Electrolyte Thickness: {{thick}} um")
print(f"C-rate: {{crate}} C")
print(f"Temperature: {{temp}} °C")
print(f"Stack Pressure: {{pres}} MPa")
print(f"Cycle Number: {{cycle}}")
print(f"Predicted Capacity Retention: {{predicted_cap:.2f}} %")
print(f"Predicted Interfacial Resistance: {{predicted_res:.2f}} Ohm")

# Generate projection over cycles
cycle_range = [1, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
projection = []
for c in cycle_range:
    feats = pd.DataFrame([[etype_encoded, thick, crate, temp, pres, c]], 
                         columns=['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number'])
    p_cap = model_cap.predict(feats)[0]
    p_res = model_res.predict(feats)[0]
    projection.append((c, p_cap, p_res))

print("\\n--- DEGRADATION PROFILE ---")
print("Cycle | Capacity Retention (%) | Resistance (Ohm)")
for c, cap, res in projection:
    print(f"{{c:5d}} | {{cap:22.2f}} | {{res:16.2f}}")
"""
            return self._python_sandbox(code)
            
        elif "analyze the output" in task_lower:
            prev_output = ""
            for log in reversed(self.state.step_logs):
                if log.get("action") == "PythonSandbox":
                    prev_output = log.get("result", "")
                    break
                    
            if prev_output:
                if self.use_llm:
                    prompt = f"""
Analyze the python execution outputs for solid-state battery degradation prediction:
{prev_output}

Provide a summary analyzing:
1. State of Health (SOH) of the cell at the final cycle.
2. Safety concerns (high resistance indicates interface degradation or lithium contact loss).
3. Recommendations for parameters optimization (e.g. temp, pressure, thickness) to improve life.

Write the analysis in Markdown.
"""
                    res = self._call_llm(prompt)
                    if res:
                        return f"[Analysis Results]:\n{res}"
                
                # Rule-based fallback
                cap_match = re.search(r'Predicted Capacity Retention:\s*([\d\.]+)\s*%', prev_output)
                res_match = re.search(r'Predicted Interfacial Resistance:\s*([\d\.]+)\s*Ohm', prev_output)
                
                analysis = "[Analysis Results]:\n"
                if cap_match and res_match:
                    cap = float(cap_match.group(1))
                    res = float(res_match.group(1))
                    analysis += f"- Predicted Capacity Retention: **{cap:.2f}%**\n"
                    analysis += f"- Predicted Interfacial Resistance: **{res:.2f} Ohm**\n"
                    
                    if cap > 80.0:
                        analysis += "- **State of Health (SOH)**: Healthy. The cell retains more than 80% capacity.\n"
                    elif cap > 70.0:
                        analysis += "- **State of Health (SOH)**: Degraded. Cell requires monitoring, capacity is between 70% and 80%.\n"
                    else:
                        analysis += "- **State of Health (SOH)**: Failed/End-of-Life. Capacity is below 70%, which is standard automotive retirement limit.\n"
                        
                    params = self._parse_parameters_from_goal(self.state.goal)
                    if params['pressure_mpa'] < 2.0:
                        analysis += "- *Recommendation*: Increase stack pressure to ~5.0 MPa to improve electrical and ionic contact, which should slow degradation.\n"
                    if params['temperature_c'] > 45.0:
                        analysis += "- *Recommendation*: Improve thermal management. Temperatures above 45°C accelerate side reactions and electrolyte interface degradation.\n"
                    if params['c_rate'] > 1.5:
                        analysis += "- *Recommendation*: Lower the charging rate. C-rates above 1.5C lead to localized stress and quick capacity fade.\n"
                else:
                    analysis += "Error parsing prediction outputs. The simulation run completed successfully but returned unexpected layout."
                return analysis
            else:
                return "Error: No Python sandbox results found to analyze."
                
        elif "synthesize final comprehensive report" in task_lower or "synthesize report" in task_lower:
            web_info = ""
            prediction_info = ""
            analysis_info = ""
            
            for log in self.state.step_logs:
                action = log.get("action")
                result = log.get("result", "")
                
                if action == "WebSearch":
                    web_info = result
                elif action == "PythonSandbox":
                    prediction_info = result
                elif action == "AnalyzeSandbox":
                    analysis_info = result
            
            if self.use_llm:
                prompt = f"""
Compile a final evaluation report for the user goal: {self.state.goal}

Inputs gathered:
1. Literature context: {web_info}
2. Machine learning predictions: {prediction_info}
3. Analysis: {analysis_info}

Synthesize a comprehensive, professional research report in Markdown.
Start the report with: `# Orbit Agent Comprehensive Evaluation Report`
Include sections for Cell Configuration, Literature Review, Machine Learning Predictions, Degradation Analysis, and Recommendations.
"""
                res = self._call_llm(prompt)
                if res:
                    return f"Final report synthesized:\n\n{res}"

            params = self._parse_parameters_from_goal(self.state.goal)
            report = f"""# Orbit Agent Comprehensive Evaluation Report
**Goal**: {self.state.goal}

## Cell Configuration
- **Electrolyte Type**: {params['electrolyte_type']}
- **Electrolyte Thickness**: {params['electrolyte_thickness_um']} um
- **Operating Temperature**: {params['temperature_c']} C
- **Applied Stack Pressure**: {params['pressure_mpa']} MPa
- **Evaluated Cycle Number**: {params['cycle_number']} cycles

## Literature & Web Search Context
{web_info.replace('[Search Results]:', '').strip()}

## Machine Learning Model Predictions
The agent queried the trained Random Forest model (`battery_model.joblib`) via the Python execution sandbox.

```
{prediction_info.strip()}
```

## Degradation & Safety Analysis
{analysis_info.replace('[Analysis Results]:', '').strip()}

---
*Report generated dynamically by OrbitAgent's autonomous ReAct pipeline.*
"""
            return f"Final report synthesized:\n\n{report}"
            
        else:
            return self._query_kb(self.state.goal)

    def _web_search(self, query: str) -> str:
        time.sleep(0.5)
        try:
            search_results = wikipedia.search(query, results=1)
            if search_results:
                summary = wikipedia.summary(search_results[0], sentences=3)
                return f"[Search Results]: Wikipedia content for '{search_results[0]}': {summary}"
        except Exception as e:
            pass
            
        return self._query_kb(query)

    def _query_kb(self, query: str) -> str:
        time.sleep(0.3)
        query_lower = query.lower()
        results = []
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                results.append(value)
                
        if results:
            return "[Search Results]: " + " | ".join(results)
        else:
            return "[Search Results]: Solid-state lithium batteries are an emerging technology utilizing solid electrolytes (sulfide, oxide, polymer) to replace flammable liquid electrolytes, offering higher safety and energy densities."

    def _python_sandbox(self, code: str) -> str:
        temp_file = "temp_agent_sandbox.py"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        try:
            res = subprocess.run([sys.executable, temp_file], capture_output=True, text=True, timeout=15)
            output = res.stdout
            if res.stderr:
                output += "\n--- ERRORS ---\n" + res.stderr
        except subprocess.TimeoutExpired:
            output = "Process timed out after 15 seconds."
        except Exception as e:
            output = f"Execution failed: {str(e)}"
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return output

    def sense_node(self, tool_result: str):
        self.state.iteration_count += 1
        
        if "Final report synthesized" in tool_result:
            self.state.memory_summary = tool_result.replace("Final report synthesized:\n\n", "")
            return
            
        task = self.state.current_task.lower()
        
        if "search wikipedia" in task:
            self.state.memory_summary = f"Gathered literature context: {tool_result[:300]}..."
        elif "write and execute python sandbox" in task:
            self.state.memory_summary = f"Executed ML predictions. Predicted values generated successfully. Sandbox output contains predicted capacity retention and interfacial resistance across cycles."
        elif "analyze the output" in task:
            self.state.memory_summary = f"Completed degradation analysis of ML predictions. SOH evaluated and recommendations drafted."
        else:
            self.state.memory_summary = f"Memory updated with output from task: {self.state.current_task}. Data: {tool_result[:100]}..."

    def run_loop(self, user_goal: str, max_iterations: int = 6) -> str:
        self.initialize_session(user_goal, max_iterations=max_iterations)
        
        while not self.state.is_complete and self.state.iteration_count < self.state.max_iterations:
            # 1. Plan
            plan_status = self.plan_node()
            if plan_status["complete"]:
                break
                
            current_task = self.state.current_task
            current_plan = list(self.state.plan)
            
            # Map task to action string for logging
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
                
            # 2. Act
            action_result = self.act_node()
            
            # Log action
            step_log = {
                "iteration": self.state.iteration_count + 1,
                "plan": current_plan,
                "task": current_task,
                "action": action_name,
                "result": action_result,
                "memory_before": self.state.memory_summary
            }
            
            # 3. Sense
            self.sense_node(action_result)
            step_log["memory_after"] = self.state.memory_summary
            
            self.state.step_logs.append(step_log)
            
        return self.state.final_output

    def generate_submission_file(self, filename="submission.csv"):
        if not os.path.exists('battery_model.joblib'):
            from ml_pipeline import run_ml_pipeline
            run_ml_pipeline()
            
        import ml_pipeline
        ml_pipeline.run_ml_pipeline()
        return filename

if __name__ == "__main__":
    agent = OrbitAgent()
    user_goal = "Predict the degradation profile of Oxide electrolyte battery cell at 45°C under 3 MPa pressure after 1200 cycles."
    print(f"Running Orbit Agent for goal: '{user_goal}'")
    report = agent.run_loop(user_goal)
    print("\n================== AGENT FINAL REPORT ==================")
    print(report)
    print("========================================================")
