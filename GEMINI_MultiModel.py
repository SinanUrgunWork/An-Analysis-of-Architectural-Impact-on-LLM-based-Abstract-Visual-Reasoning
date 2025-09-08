import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from tool import tool_functions, execute_tool_function
import concurrent.futures
import threading
from queue import Queue
import re
import google.generativeai as genai
from google.protobuf.json_format import MessageToDict


INPUTS_FOLDER = "inputs"  
DELAY_BETWEEN_CALLS = 1  
DELAY_BETWEEN_FILES = 0.5  
MAX_WORKERS = 4  
BATCH_SIZE = 4  
API_RETRY_ATTEMPTS = 3  
API_RETRY_DELAY = 2.0  


load_dotenv()
MODEL_NAME = "gemini-1.5-flash"


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


log_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with log_lock:
        print(*args, **kwargs)

def clean_schema_for_gemini(schema):

    if not isinstance(schema, dict):
        return schema
    
    cleaned = {}
    

    unsupported_fields = {
        'additionalProperties', 
        '$schema', 
        '$id',
        'definitions',
        'allOf',
        'anyOf',
        'oneOf',
        'not',
        # Array validation fields
        'maxItems',
        'minItems',
        'uniqueItems',
        # String validation fields
        'maxLength',
        'minLength',
        'pattern',
        'format',
        # Number validation fields
        'minimum',
        'maximum', 
        'multipleOf',
        'exclusiveMaximum',
        'exclusiveMinimum',
        # Other advanced fields
        'const',
        'examples',
        'default',
        'readOnly',
        'writeOnly',
        'xml',
        'externalDocs',
        'deprecated'
    }
    
    for key, value in schema.items():
        if key in unsupported_fields:
            continue
            
        if isinstance(value, dict):
            cleaned[key] = clean_schema_for_gemini(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_schema_for_gemini(item) if isinstance(item, dict) else item for item in value]
        else:
            cleaned[key] = value
    
    return cleaned

def get_gemini_tools():
    """Convert tool_functions to Gemini format """
    gemini_tools = []
    for tool in tool_functions:
        original_params = tool["function"]["parameters"]
        
        cleaned_params = clean_schema_for_gemini(original_params)
        
        gemini_tool = {
            "function_declarations": [{
                "name": tool["function"]["name"],
                "description": tool["function"]["description"], 
                "parameters": cleaned_params
            }]
        }
        gemini_tools.append(gemini_tool)
    return gemini_tools

def extract_function_args(function_call):
    arguments = {}
    
    try:
        # Method 1: MessageToDict 
        if hasattr(function_call, '_pb'):
            call_dict = MessageToDict(function_call._pb)
            arguments = call_dict.get('args', {})
            if arguments and 'config_type' in arguments:
                return arguments
    except Exception:
        pass
    
    try:
        # Method 2: Direct args access
        if hasattr(function_call, 'args'):
            if hasattr(function_call.args, '_pb'):
                arguments = MessageToDict(function_call.args._pb)
                if arguments and 'config_type' in arguments:
                    return arguments
    except Exception:
        pass
    
    try:
        # Method 3: Manual field extraction
        if hasattr(function_call, 'args'):
            args = function_call.args
            
            # Extract config_type
            if hasattr(args, 'config_type'):
                arguments['config_type'] = str(args.config_type)
            
            # Extract objects array
            if hasattr(args, 'objects'):
                objects = []
                for obj in args.objects:
                    obj_dict = {}
                    if hasattr(obj, 'shape'):
                        obj_dict['shape'] = str(obj.shape)
                    if hasattr(obj, 'size'):
                        obj_dict['size'] = float(obj.size)
                    if hasattr(obj, 'color'):
                        obj_dict['color'] = int(obj.color)
                    if hasattr(obj, 'angle'):
                        obj_dict['angle'] = int(obj.angle)
                    objects.append(obj_dict)
                arguments['objects'] = objects
            
            # Extract optional fields
            if hasattr(args, 'grid_layout'):
                arguments['grid_layout'] = str(args.grid_layout)
            if hasattr(args, 'positions'):
                arguments['positions'] = list(args.positions)
            if hasattr(args, 'empty_positions'):
                arguments['empty_positions'] = list(args.empty_positions)
                
        if arguments and 'config_type' in arguments:
            return arguments
            
    except Exception:
        pass
    
    return arguments

def call_gemini_api(prompt, system_prompt, use_tools=False, temperature=0.1):
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4000,
                candidate_count=1,
            )
            
            if use_tools:
                gemini_tools = get_gemini_tools()
                response = model.generate_content(
                    full_prompt,
                    tools=gemini_tools,
                    generation_config=generation_config
                )
            else:
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
            
            # Enhanced response validation
            if not response or not response.candidates:
                raise ValueError("No response candidates from Gemini 1.5 Flash")
                
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("No content parts in response")
            
            return response
            
        except Exception as e:
            if attempt < API_RETRY_ATTEMPTS - 1:
                safe_print(f"Gemini API retry {attempt + 1}/{API_RETRY_ATTEMPTS}: {e}")
                time.sleep(API_RETRY_DELAY)
                continue
            else:
                raise e

class RavenAgentSystem:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.agent_analyses = {}
        self.worker_id = threading.current_thread().name
        self.timestamp = datetime.now().isoformat()
        
        # Load the puzzle data
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.raw_json_data = f.read()

    def analyze_config_types(self):
        """SLAVE AGENT 1: Analyze configuration patterns ONLY - Focus on layout structures"""
        safe_print(f"[{self.worker_id}] Slave Agent 1: Analyzing configuration types...")
        
        # SAME PROMPT as Claude version
        prompt = f"""
RAVEN'S MATRICES - CONFIGURATION PATTERN SPECIALIST

You are a specialist that ONLY analyzes configuration/layout patterns. Focus EXCLUSIVELY on how objects are arranged spatially.

MATRIX DATA:
{self.raw_json_data}

YOUR EXCLUSIVE TASK: Analyze configuration patterns across the 3x3 matrix.

AVAILABLE CONFIGURATION TYPES:
- "singleton_center": Single object in center
- "left_right": Two objects, positioned left and right  
- "up_down": Two objects, positioned top and bottom
- "out_in": Two objects, one outer and one inner
- "distribute_three": Three objects in triangular arrangement
- "grid_2x2": Four objects in 2x2 grid
- "grid_3x3": Nine objects in 3x3 grid

ANALYSIS STEPS:
1. Examine each panel's configuration type
2. Find row patterns in configurations
3. Find column patterns in configurations
4. Determine what configuration type panel 3_3 should have

IMPORTANT: Do NOT analyze shapes, colors, sizes, or angles. Focus ONLY on spatial arrangements and configurations.

Provide your analysis and conclude with: "RECOMMENDED CONFIG_TYPE: [type]"
"""

        system_prompt = "You are a configuration pattern specialist. Focus ONLY on spatial arrangements and layout patterns. Ignore shapes, colors, sizes, and rotations."

        response = call_gemini_api(prompt, system_prompt, use_tools=False)
        
        analysis = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    analysis += part.text
        
        if not analysis:
            analysis = "No analysis content in response"
        
        self.agent_analyses["config_agent"] = {
            "analysis": analysis,
            "specialization": "configuration_patterns",
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        time.sleep(DELAY_BETWEEN_CALLS)
        return analysis

    def analyze_shapes(self):
        """SLAVE AGENT 2: Analyze shape patterns ONLY - Focus on shape types and transformations"""
        safe_print(f"[{self.worker_id}] Slave Agent 2: Analyzing shape patterns...")
        
        # SAME PROMPT as Claude version
        prompt = f"""
RAVEN'S MATRICES - SHAPE PATTERN SPECIALIST

You are a specialist that ONLY analyzes shape patterns. Focus EXCLUSIVELY on shape types and shape transformations.

MATRIX DATA:
{self.raw_json_data}

YOUR EXCLUSIVE TASK: Analyze shape patterns across the 3x3 matrix.

AVAILABLE SHAPES:
- "triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"

ANALYSIS STEPS:
1. Identify shapes in each panel
2. Find row patterns for shapes
3. Find column patterns for shapes
4. Identify shape transformation rules
5. Determine what shapes panel 3_3 should contain

IMPORTANT: Do NOT analyze configurations, colors, sizes, or angles. Focus ONLY on shape types and shape sequence patterns.

Provide your analysis and conclude with: "RECOMMENDED SHAPES: [list of shapes]"
"""

        system_prompt = "You are a shape pattern specialist. Focus ONLY on shape types and shape transformations. Ignore configurations, colors, sizes, and rotations."

        response = call_gemini_api(prompt, system_prompt, use_tools=False)
        
        analysis = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    analysis += part.text
        
        if not analysis:
            analysis = "No analysis content in response"
        
        self.agent_analyses["shape_agent"] = {
            "analysis": analysis,
            "specialization": "shape_patterns",
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        time.sleep(DELAY_BETWEEN_CALLS)
        return analysis

    def analyze_colors(self):
        """SLAVE AGENT 3: Analyze color patterns ONLY - Focus on color values and progressions"""
        safe_print(f"[{self.worker_id}] Slave Agent 3: Analyzing color patterns...")
        
        # SAME PROMPT as Claude version
        prompt = f"""
RAVEN'S MATRICES - COLOR PATTERN SPECIALIST

You are a specialist that ONLY analyzes color patterns. Focus EXCLUSIVELY on color values and color progressions.

MATRIX DATA:
{self.raw_json_data}

YOUR EXCLUSIVE TASK: Analyze color patterns across the 3x3 matrix.

AVAILABLE COLORS:
- Integer 0-9 (0=white, 9=black)
- Integer 0-255 (grayscale value)
- String names: "white", "very_light_gray", "light_gray", "medium_light_gray", 
  "medium_gray", "medium_dark_gray", "dark_gray", "very_dark_gray", "almost_black", "black"

ANALYSIS STEPS:
1. Identify colors in each panel
2. Find row patterns for colors
3. Find column patterns for colors
4. Identify color progression rules (gradients, alternation, etc.)
5. Determine what colors panel 3_3 should have

IMPORTANT: Do NOT analyze configurations, shapes, sizes, or angles. Focus ONLY on color values and color sequence patterns.

Provide your analysis and conclude with: "RECOMMENDED COLORS: [list of colors]"
"""

        system_prompt = "You are a color pattern specialist. Focus ONLY on color values and color progressions. Ignore configurations, shapes, sizes, and rotations."

        response = call_gemini_api(prompt, system_prompt, use_tools=False)
        
        analysis = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    analysis += part.text
        
        if not analysis:
            analysis = "No analysis content in response"
        
        self.agent_analyses["color_agent"] = {
            "analysis": analysis,
            "specialization": "color_patterns",
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        time.sleep(DELAY_BETWEEN_CALLS)
        return analysis

    def analyze_sizes(self):
        """SLAVE AGENT 4: Analyze size patterns ONLY - Focus on size values and scaling"""
        safe_print(f"[{self.worker_id}] Slave Agent 4: Analyzing size patterns...")
        
        # SAME PROMPT as Claude version
        prompt = f"""
RAVEN'S MATRICES - SIZE PATTERN SPECIALIST

You are a specialist that ONLY analyzes size patterns. Focus EXCLUSIVELY on size values and size scaling patterns.

MATRIX DATA:
{self.raw_json_data}

YOUR EXCLUSIVE TASK: Analyze size patterns across the 3x3 matrix.

AVAILABLE SIZES:
- Float from 0.1 to 1.0


ANALYSIS STEPS:
1. Identify sizes in each panel
2. Find row patterns for sizes
3. Find column patterns for sizes
4. Identify size scaling rules (increasing, decreasing, alternating, etc.)
5. Determine what sizes panel 3_3 should have

IMPORTANT: Do NOT analyze configurations, shapes, colors, or angles. Focus ONLY on size values and size progression patterns.

Provide your analysis and conclude with: "RECOMMENDED SIZES: [list of sizes]"
"""

        system_prompt = "You are a size pattern specialist. Focus ONLY on size values and size scaling patterns. Ignore configurations, shapes, colors, and rotations."

        response = call_gemini_api(prompt, system_prompt, use_tools=False)
        
        analysis = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    analysis += part.text
        
        if not analysis:
            analysis = "No analysis content in response"
        
        self.agent_analyses["size_agent"] = {
            "analysis": analysis,
            "specialization": "size_patterns",
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        time.sleep(DELAY_BETWEEN_CALLS)
        return analysis

    def analyze_angles(self):
        """SLAVE AGENT 5: Analyze angle patterns ONLY - Focus on rotation values and transformations"""
        safe_print(f"[{self.worker_id}] Slave Agent 5: Analyzing angle patterns...")
        
        # SAME PROMPT as Claude version
        prompt = f"""
RAVEN'S MATRICES - ROTATION PATTERN SPECIALIST

You are a specialist that ONLY analyzes rotation/angle patterns. Focus EXCLUSIVELY on rotation values and rotational transformations.

MATRIX DATA:
{self.raw_json_data}

YOUR EXCLUSIVE TASK: Analyze rotation patterns across the 3x3 matrix.

AVAILABLE ANGLES:
- Rotation in degrees (-180 to 180)


ANALYSIS STEPS:
1. Identify rotations/angles in each panel
2. Find row patterns for rotations
3. Find column patterns for rotations
4. Identify rotation transformation rules (incremental rotation, mirroring, etc.)
5. Determine what angles panel 3_3 should have

IMPORTANT: Do NOT analyze configurations, shapes, colors, or sizes. Focus ONLY on rotation values and rotational sequence patterns.

Provide your analysis and conclude with: "RECOMMENDED ANGLES: [list of angles]"
"""

        system_prompt = "You are a rotation pattern specialist. Focus ONLY on rotation values and rotational transformations. Ignore configurations, shapes, colors, and sizes."

        response = call_gemini_api(prompt, system_prompt, use_tools=False)
        
        analysis = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    analysis += part.text
        
        if not analysis:
            analysis = "No analysis content in response"
        
        self.agent_analyses["angle_agent"] = {
            "analysis": analysis,
            "specialization": "rotation_patterns",
            "analysis_length": len(analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        time.sleep(DELAY_BETWEEN_CALLS)
        return analysis

    def save_agent_analyses(self):
        """Save all agent analyses to JSON with unique filename"""
        # Generate unique filename based on input JSON
        base_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
        number_match = re.search(r'\d+', base_name)
        if number_match:
            question_number = number_match.group()
            filename = f"agent_analyses_question{question_number}.json"
        else:
            filename = f"agent_analyses_{base_name}.json"
        
        # Calculate detailed agent analysis metrics
        total_analysis_length = sum(data["analysis_length"] for data in self.agent_analyses.values())
        agent_specializations = [data["specialization"] for data in self.agent_analyses.values()]
        
        agent_data = {
            "puzzle_file": self.json_file_path,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_multi_agent_specialist_analysis",
            "agent_analyses": self.agent_analyses,
            "analysis_summary": {
                "total_agents": len(self.agent_analyses),
                "agents_completed": list(self.agent_analyses.keys()),
                "total_analysis_length": total_analysis_length,
                "average_analysis_length": total_analysis_length / len(self.agent_analyses) if self.agent_analyses else 0,
                "specializations": agent_specializations,
                "agent_diversity_score": len(set(agent_specializations)),
                "analysis_completeness": len(self.agent_analyses) == 5,  # Expected 5 agents
                "specialist_coverage": {
                    "config_patterns": "config_agent" in self.agent_analyses,
                    "shape_patterns": "shape_agent" in self.agent_analyses,
                    "color_patterns": "color_agent" in self.agent_analyses,
                    "size_patterns": "size_agent" in self.agent_analyses,
                    "rotation_patterns": "angle_agent" in self.agent_analyses
                }
            }
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False)
        
        safe_print(f"[{self.worker_id}] Agent analyses saved to: {filename}")
        return filename

    def master_solver(self):
        """MASTER LLM: Synthesizes slave analyses and generates solution - NEVER SEES RAW INPUT DATA"""
        safe_print(f"[{self.worker_id}] Master Agent: Synthesizing solution from slave analyses...")
        
        # Compile all agent insights - MASTER ONLY GETS THESE, NOT RAW DATA
        agent_insights = ""
        agent_insight_stats = {
            "total_insights_length": 0,
            "agent_contributions": {}
        }
        
        for agent_name, data in self.agent_analyses.items():
            analysis_text = data.get('analysis', '') or ''
            specialization = data.get('specialization', 'unknown')
            agent_insights += f"\n=== {agent_name.upper()} ({specialization.upper()}) ===\n{analysis_text}\n"
            
            agent_insight_stats["agent_contributions"][agent_name] = {
                "analysis_length": len(analysis_text),
                "specialization": specialization,
                "contribution_ratio": len(analysis_text) / agent_insight_stats["total_insights_length"] if agent_insight_stats["total_insights_length"] > 0 else 0
            }
            agent_insight_stats["total_insights_length"] += len(analysis_text)

        # Update contribution ratios
        for agent_data in agent_insight_stats["agent_contributions"].values():
            agent_data["contribution_ratio"] = agent_data["analysis_length"] / agent_insight_stats["total_insights_length"] if agent_insight_stats["total_insights_length"] > 0 else 0

        # SAME MASTER PROMPT as Claude version - NO RAW INPUT DATA, ONLY SLAVE ANALYSES
        prompt = f"""RAVEN'S MATRIX MASTER SOLVER

You are the master coordinator that synthesizes specialist analyses to solve a 3x3 Raven's matrix puzzle.

You do NOT have access to the raw puzzle data. Instead, you have detailed analyses from 5 specialist agents who each focused on one specific aspect:

SPECIALIST AGENT ANALYSES:
{agent_insights}

YOUR TASK:
1. SYNTHESIZE INSIGHTS: Combine all specialist recommendations
2. RESOLVE CONFLICTS: If agents disagree, determine the most logical solution
3. INTEGRATE PATTERNS: Merge configuration, shape, color, size, and angle patterns
4. GENERATE SOLUTION: Call generate_visual_panel with the exact parameters recommended by specialists

Based on the specialist analyses above, determine the complete solution for panel 3_3.

ANALYSIS FRAMEWORK:
1. EXAMINE SPECIALIST INSIGHTS: Review each agent's recommendations
2. SYNTHESIZE PATTERNS: Combine all specialist findings into coherent solution
3. RESOLVE CONFLICTS: If specialists disagree, determine most logical approach
4. INTEGRATE SOLUTION: Merge config_type + shapes + colors + sizes + angles
5. GENERATE SOLUTION: Call generate_visual_panel with synthesized parameters

Use the generate_visual_panel function with parameters based ENTIRELY on what the specialists recommend:
- Use the config_type that the Configuration Specialist recommended
- Use the shapes that the Shape Specialist recommended  
- Use the colors that the Color Specialist recommended
- Use the sizes that the Size Specialist recommended
- Use the angles that the Rotation Specialist recommended

Follow the 5 steps above, then call the function with the synthesized specialist recommendations."""

        # SAME SYSTEM PROMPT as Claude version
        system_prompt = """You are an expert Raven's Progressive Matrices master solver with access to a powerful visual generation tool.

You synthesize analyses from specialist agents who have examined configuration patterns, shape patterns, color patterns, size patterns, and rotation patterns.

The tool can create ANY arrangement of shapes with precise control over:
- Layout configurations
- Object positions (specific indices in grids)
- Shape types (7 different shapes + line)
- Colors (10 grayscale levels with multiple naming conventions)
- Sizes (continuous scale from 0.1 to 1.0)
- Rotations (any angle from -180 to 180)

Think step by step to synthesize all specialist insights, then generate the solution using ALL available tool capabilities.
You MUST call the generate_visual_panel function - do not provide examples or pseudo-code.
Use the exact parameter values and terminology from the specialist analyses."""

        time.sleep(DELAY_BETWEEN_CALLS)
        
        response = call_gemini_api(prompt, system_prompt, use_tools=True)

        # Extract response content and function calls
        cot_reasoning = ""
        function_calls = []
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    cot_reasoning += part.text
                elif hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)

        if function_calls:
            for call in function_calls:
                function_name = call.name
                arguments = extract_function_args(call)

                # Generate unique filename for master CoT based on input JSON
                base_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
                number_match = re.search(r'\d+', base_name)
                if number_match:
                    question_number = number_match.group()
                    master_cot_filename = f"master_synthesis_cot_reasoning_question{question_number}.json"
                    llm_output_filename = f"llm_output_question{question_number}.json"
                else:
                    master_cot_filename = f"master_synthesis_cot_reasoning_{base_name}.json"
                    llm_output_filename = f"llm_output_{base_name}.json"

                # Save master CoT reasoning with agent insights and detailed analysis
                master_cot_data = {
                    "puzzle_file": self.json_file_path,
                    "worker_id": self.worker_id,
                    "timestamp": self.timestamp,
                    "model_used": MODEL_NAME,
                    "processing_type": "gemini15_multi_agent_cot",
                    "solving_session": {
                        "raw_llm_response": cot_reasoning,
                        "cot_reasoning": cot_reasoning,
                        "reasoning_length": len(cot_reasoning),
                        "function_called": function_name,
                        "function_arguments": arguments,
                        "success": True,
                        "extraction_method": "gemini15_multi_agent_synthesis"
                    },
                    "gemini15_multi_agent_system": {
                        "master_sees_raw_data": False,
                        "master_only_sees_agent_analyses": True,
                        "agent_analyses_used": list(self.agent_analyses.keys()),
                        "agent_specializations": [analysis["specialization"] for analysis in self.agent_analyses.values()],
                        "total_agent_insights_length": agent_insight_stats["total_insights_length"],
                        "synthesis_approach": "specialist_coordination",
                        "agent_insight_statistics": agent_insight_stats,
                        "architecture_type": "slave_specialist_master_synthesis"
                    },
                    "reasoning_analysis": {
                        # Basic step analysis
                        "has_step1_patterns": any(keyword in cot_reasoning.lower() for keyword in ["examine", "specialist", "step 1", "insights"]),
                        "has_step2_rule": any(keyword in cot_reasoning.lower() for keyword in ["synthesize", "combine", "step 2", "patterns"]),
                        "has_step3_apply": any(keyword in cot_reasoning.lower() for keyword in ["resolve", "conflicts", "step 3", "integrate"]),
                        "has_step4_generate": any(keyword in cot_reasoning.lower() for keyword in ["generate", "solution", "step 4", "parameters"]),
                        "has_step5_solution": any(keyword in cot_reasoning.lower() for keyword in ["function", "call", "step 5", "implement"]),
                        
                        # Multi-agent specific analysis
                        "synthesizes_config": "config" in cot_reasoning.lower(),
                        "synthesizes_shapes": any(shape in cot_reasoning.lower() for shape in ["triangle", "square", "pentagon", "hexagon", "circle"]),
                        "synthesizes_colors": any(color in cot_reasoning.lower() for color in ["black", "gray", "white", "color"]),
                        "synthesizes_sizes": "size" in cot_reasoning.lower(),
                        "synthesizes_angles": "angle" in cot_reasoning.lower() or "rotation" in cot_reasoning.lower(),
                        "mentions_specialist_insights": any(agent in cot_reasoning.lower() for agent in ["config", "shape", "color", "size", "angle", "specialist"]),
                        
                        # Content quality analysis
                        "mentions_rows_columns": "row" in cot_reasoning.lower() and "column" in cot_reasoning.lower(),
                        "mentions_positions": "position" in cot_reasoning.lower() or "grid" in cot_reasoning.lower(),
                        "cot_captured_successfully": len(cot_reasoning) > 50,
                        "systematic_approach": "systematic" in cot_reasoning.lower() or "analysis" in cot_reasoning.lower(),
                        "pattern_terminology": any(term in cot_reasoning.lower() for term in ["pattern", "rule", "progression", "sequence"]),
                        "transformation_analysis": any(term in cot_reasoning.lower() for term in ["transform", "change", "rotation", "scaling"]),
                        
                        # Quality metrics
                        "step_structure_quality": cot_reasoning.lower().count("step"),
                        "reasoning_depth_score": len(cot_reasoning) / 100,  # Basic depth metric
                        "contains_framework_keywords": any(keyword in cot_reasoning.lower() for keyword in ["framework", "analysis", "systematic", "examine"]),
                        "gemini15_multi_agent_processing": True,
                        
                        # Master synthesis quality
                        "master_synthesis_quality": sum([
                            "synthesize" in cot_reasoning.lower(),
                            "combine" in cot_reasoning.lower(),
                            "integrate" in cot_reasoning.lower(),
                            "specialist" in cot_reasoning.lower(),
                            "analysis" in cot_reasoning.lower()
                        ]),
                        "reasoning_completeness_score": sum([
                            "step 1" in cot_reasoning.lower() or "examine" in cot_reasoning.lower(),
                            "step 2" in cot_reasoning.lower() or "synthesize" in cot_reasoning.lower(),
                            "step 3" in cot_reasoning.lower() or "resolve" in cot_reasoning.lower(),
                            "step 4" in cot_reasoning.lower() or "integrate" in cot_reasoning.lower(),
                            "step 5" in cot_reasoning.lower() or "generate" in cot_reasoning.lower(),
                            "specialist" in cot_reasoning.lower()
                        ]),
                        "synthesis_coordination_score": sum([
                            any(agent in cot_reasoning.lower() for agent in ["config", "shape", "color", "size", "angle"]),
                            "specialist" in cot_reasoning.lower(),
                            "recommendation" in cot_reasoning.lower(),
                            "combine" in cot_reasoning.lower(),
                            "coordinate" in cot_reasoning.lower()
                        ])
                    }
                }

                with open(master_cot_filename, "w", encoding="utf-8") as cot_file:
                    json.dump(master_cot_data, cot_file, indent=2, ensure_ascii=False)

                # Save LLM function arguments for debugging (separate from CoT)
                with open(llm_output_filename, "w", encoding="utf-8") as out:
                    json.dump({
                        "worker_id": self.worker_id,
                        "function_arguments": arguments,
                        "note": f"Gemini 1.5 Flash multi-agent master synthesis CoT reasoning details are in {master_cot_filename}"
                    }, out, indent=2, ensure_ascii=False)

                # Execute the tool with source filename for dynamic naming
                result = execute_tool_function(function_name, arguments, self.json_file_path)
                
                # Update master CoT file with execution result
                master_cot_data["execution_result"] = {
                    "status": result.get("status", "unknown"),
                    "filename": result.get("filename", ""),
                    "execution_successful": result.get("status") == "success"
                }
                
                with open(master_cot_filename, "w", encoding="utf-8") as cot_file:
                    json.dump(master_cot_data, cot_file, indent=2, ensure_ascii=False)
                
                return result
                    
        else:
            # Handle case where function was not called
            error_msg = "Master LLM did not call the tool function"
            if cot_reasoning:
                error_msg += f" - Response: {cot_reasoning}"
            
            # Generate unique error filename
            base_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
            number_match = re.search(r'\d+', base_name)
            if number_match:
                question_number = number_match.group()
                error_filename = f"master_synthesis_cot_reasoning_question{question_number}.json"
            else:
                error_filename = f"master_synthesis_cot_reasoning_{base_name}.json"
            
            # Save failed CoT attempt with detailed analysis
            failed_cot_data = {
                "puzzle_file": self.json_file_path,
                "worker_id": self.worker_id,
                "timestamp": self.timestamp,
                "model_used": MODEL_NAME,
                "processing_type": "gemini15_multi_agent_cot",
                "solving_session": {
                    "raw_llm_response": cot_reasoning,
                    "cot_reasoning": cot_reasoning,
                    "reasoning_length": len(cot_reasoning),
                    "function_called": None,
                    "function_arguments": None,
                    "success": False,
                    "error_message": error_msg,
                    "extraction_method": "gemini15_multi_agent_synthesis_failed"
                },
                "gemini15_multi_agent_system": {
                    "master_sees_raw_data": False,
                    "master_only_sees_agent_analyses": True,
                    "agent_analyses_used": list(self.agent_analyses.keys()),
                    "synthesis_approach": "specialist_coordination"
                },
                "reasoning_analysis": {
                    "has_reasoning": len(cot_reasoning) > 0,
                    "reasoning_but_no_function_call": len(cot_reasoning) > 0,
                    "likely_pseudo_code": "function" in cot_reasoning.lower() or "call" in cot_reasoning.lower(),
                    "mentions_specialist_insights": any(agent in cot_reasoning.lower() for agent in ["config", "shape", "color", "size", "angle", "specialist"]),
                    "synthesis_attempt": "synthesize" in cot_reasoning.lower() or "combine" in cot_reasoning.lower(),
                    "cot_captured_successfully": len(cot_reasoning) > 50,
                    "gemini15_multi_agent_processing": True,
                    "reasoning_depth_score": len(cot_reasoning) / 100,
                    "failure_analysis": {
                        "has_reasoning_content": len(cot_reasoning) > 0,
                        "mentions_function_concepts": any(term in cot_reasoning.lower() for term in ["function", "call", "generate", "tool"]),
                        "contains_parameters": any(param in cot_reasoning.lower() for param in ["config_type", "objects", "shape", "color", "size", "angle"]),
                        "synthesis_indicators": sum([
                            "synthesize" in cot_reasoning.lower(),
                            "combine" in cot_reasoning.lower(),
                            "integrate" in cot_reasoning.lower(),
                            "specialist" in cot_reasoning.lower()
                        ])
                    }
                }
            }
            
            with open(error_filename, "w", encoding="utf-8") as cot_file:
                json.dump(failed_cot_data, cot_file, indent=2, ensure_ascii=False)
            
            return {"status": "error", "message": error_msg}


def get_json_files_from_inputs():
    """Get all questionText*.json files from inputs folder, sorted by number"""
    if not os.path.exists(INPUTS_FOLDER):
        safe_print(f"Error: {INPUTS_FOLDER} folder not found!")
        return []
    
    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            if os.path.getsize(filepath) > 0:  # Check if file is not empty
                json_files.append(filepath)
    
    # Sort by the number in filename
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    json_files.sort(key=extract_number)
    return json_files


def solve_raven_from_json(json_file_path):
    """
    Gemini 1.5 Flash Multi-agent RAVEN solver: Slaves analyze raw data, Master synthesizes slave outputs
    """
    try:
        if not os.path.exists(json_file_path):
            return {"status": "error", "message": f"JSON file not found: {json_file_path}"}
        
        safe_print(f"[{threading.current_thread().name}] Starting Gemini 1.5 Flash Multi-Agent Raven Solver...")
        agent_system = RavenAgentSystem(json_file_path)
        
        # PHASE 1: Slave agents analyze raw input data (each focuses on ONE aspect)
        safe_print(f"[{threading.current_thread().name}] Phase 1: Slave agents analyzing raw data...")
        agent_system.analyze_config_types()    # Slave 1: Configuration patterns only
        agent_system.analyze_shapes()          # Slave 2: Shape patterns only
        agent_system.analyze_colors()          # Slave 3: Color patterns only
        agent_system.analyze_sizes()           # Slave 4: Size patterns only
        agent_system.analyze_angles()          # Slave 5: Angle patterns only
        
        # Save all slave analyses
        agent_system.save_agent_analyses()
        
        # PHASE 2: Master synthesizes slave analyses (never sees raw data)
        safe_print(f"[{threading.current_thread().name}] Phase 2: Master synthesizing slave insights...")
        result = agent_system.master_solver()
        
        return result

    except Exception as e:
        # Generate unique error filename
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        number_match = re.search(r'\d+', base_name)
        if number_match:
            question_number = number_match.group()
            error_filename = f"gemini15_multi_agent_error_question{question_number}.json"
        else:
            error_filename = f"gemini15_multi_agent_error_{base_name}.json"
        
        # Save exception details
        error_data = {
            "puzzle_file": json_file_path,
            "worker_id": threading.current_thread().name,
            "timestamp": datetime.now().isoformat(),
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_multi_agent_cot",
            "solving_session": {
                "success": False,
                "exception": str(e),
                "error_type": type(e).__name__
            }
        }
        
        with open(error_filename, "w", encoding="utf-8") as error_file:
            json.dump(error_data, error_file, indent=2, ensure_ascii=False)
            
        return {"status": "error", "message": str(e)}


def process_single_file(json_file_path, file_index, total_files):
    """Process a single JSON file with progress tracking"""
    filename = os.path.basename(json_file_path)
    
    try:
        result = solve_raven_from_json(json_file_path)
        
        if result.get("status") == "success":
            safe_print(f"[{file_index}/{total_files}] SUCCESS: {filename} -> {result['filename']}")
            return {
                "file": filename,
                "status": "success",
                "output": result['filename'],
                "message": result.get('message', '')
            }
        else:
            safe_print(f"[{file_index}/{total_files}] FAILED: {filename}: {result.get('message')}")
            return {
                "file": filename,
                "status": "failed",
                "output": None,
                "message": result.get('message', '')
            }
            
    except Exception as e:
        error_msg = str(e)
        safe_print(f"[{file_index}/{total_files}] ERROR: {filename}: {error_msg}")
        return {
            "file": filename,
            "status": "error",
            "output": None,
            "message": error_msg
        }


def process_files_in_parallel(json_files):
    """Process multiple files in parallel using ThreadPoolExecutor"""
    results_summary = []
    total_files = len(json_files)
    
    safe_print(f"\nProcessing {total_files} files in batches of {BATCH_SIZE}")
    safe_print(f"Using {MAX_WORKERS} parallel workers")
    safe_print("-" * 60)
    
    # Process files in batches
    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = json_files[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
        
        safe_print(f"\n=== Batch {batch_num}/{total_batches} (Files {batch_start + 1}-{batch_end}) ===")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all files in the batch
            future_to_file = {
                executor.submit(process_single_file, filepath, i+batch_start+1, total_files): filepath 
                for i, filepath in enumerate(batch_files)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result()
                    results_summary.append(result)
                except Exception as e:
                    filename = os.path.basename(filepath)
                    results_summary.append({
                        "file": filename,
                        "status": "error",
                        "output": None,
                        "message": str(e)
                    })
        
        # Progress update
        processed_so_far = min(batch_end, total_files)
        success_so_far = len([r for r in results_summary if r["status"] == "success"])
        safe_print(f"\nProgress: {processed_so_far}/{total_files} files ({success_so_far} successful)")
        
        # Small delay between batches
        if batch_end < total_files:
            safe_print(f"Waiting {DELAY_BETWEEN_FILES}s before next batch...")
            time.sleep(DELAY_BETWEEN_FILES)
    
    return results_summary


def process_all_json_files():
    """Process all JSON files in the inputs folder with parallel processing"""
    json_files = get_json_files_from_inputs()
    
    if not json_files:
        safe_print("No questionText*.json files found in inputs folder!")
        return
    
    safe_print(f"Found {len(json_files)} JSON files to process")
    
    start_time = time.time()
    
    # Process files in parallel
    results_summary = process_files_in_parallel(json_files)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate statistics
    total_files = len(json_files)
    successful = len([r for r in results_summary if r["status"] == "success"])
    failed = len([r for r in results_summary if r["status"] != "success"])
    
    # Save final summary
    summary_data = {
        "processing_session": {
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_multi_agent_cot_parallel",
            "architecture": "slave_specialist_master_synthesis",
            "delay_between_files": DELAY_BETWEEN_FILES,
            "parallel_workers": MAX_WORKERS,
            "batch_size": BATCH_SIZE,
            "total_time_seconds": elapsed_time,
            "total_time_minutes": elapsed_time / 60,
            "average_time_per_file": elapsed_time / total_files if total_files else 0,
            "success_rate": successful / total_files if total_files else 0,
            "api_retry_attempts": API_RETRY_ATTEMPTS
        },
        "gemini15_multi_agent_architecture": {
            "slave_agents": {
                "config_agent": "configuration_patterns",
                "shape_agent": "shape_patterns", 
                "color_agent": "color_patterns",
                "size_agent": "size_patterns",
                "angle_agent": "rotation_patterns"
            },
            "master_agent": "synthesis_coordinator",
            "data_flow": "slaves_see_raw_data -> master_sees_only_analyses",
            "specialization_approach": "single_aspect_focus_per_agent",
            "model_version": "gemini-1.5-flash"
        },
        "results": results_summary
    }
    
    with open("gemini15_multi_agent_batch_summary_parallel.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    safe_print(f"\n{'='*60}")
    safe_print("GEMINI 1.5 FLASH MULTI-AGENT BATCH PROCESSING COMPLETE")
    safe_print(f"{'='*60}")
    safe_print(f"Architecture: Slave Specialists + Master Synthesizer")
    safe_print(f"Model: {MODEL_NAME}")
    safe_print(f"Total files processed: {total_files}")
    safe_print(f"Successful: {successful}")
    safe_print(f"Failed: {failed}")
    safe_print(f"Success rate: {successful/total_files*100:.1f}%")
    safe_print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    safe_print(f"Average time per file: {summary_data['processing_session']['average_time_per_file']:.2f}s")
    safe_print(f"API retry attempts: {API_RETRY_ATTEMPTS}")
    safe_print("Summary saved to: gemini15_multi_agent_batch_summary_parallel.json")
    
    # List successful outputs
    successful_outputs = [r["output"] for r in results_summary if r["status"] == "success"]
    if successful_outputs:
        safe_print(f"\nGenerated files:")
        for i, output in enumerate(successful_outputs[:10]):  # Show first 10
            safe_print(f"  - {output}")
        if len(successful_outputs) > 10:
            safe_print(f"  ... and {len(successful_outputs) - 10} more")
    
    return results_summary


def main():
    safe_print("Architecture: 5 Slave Specialists + 1 Master Synthesizer")
    safe_print(f"Inputs folder: {INPUTS_FOLDER}")
    safe_print(f"Parallel workers: {MAX_WORKERS}")
    safe_print(f"Batch size: {BATCH_SIZE}")
    safe_print(f"Delay between batches: {DELAY_BETWEEN_FILES}s")
    safe_print(f"Model: {MODEL_NAME}")
    safe_print(f"API retry attempts: {API_RETRY_ATTEMPTS}")
    safe_print("")
    safe_print("System Design:")
    safe_print("- Slave Agent 1: Configuration Pattern Specialist")
    safe_print("- Slave Agent 2: Shape Pattern Specialist") 
    safe_print("- Slave Agent 3: Color Pattern Specialist")
    safe_print("- Slave Agent 4: Size Pattern Specialist")
    safe_print("- Slave Agent 5: Rotation Pattern Specialist")
    safe_print("- Master Agent: Synthesis Coordinator ")
    
    if not os.getenv("GEMINI_API_KEY"):
        safe_print("[!] Missing GEMINI_API_KEY in .env file")
        return
    
    # Process all JSON files in inputs folder
    results = process_all_json_files()
    
    if not results:
        return
    
    # Show final statistics
    success_count = len([r for r in results if r["status"] == "success"])
    total_count = len(results)
    
    safe_print(f"\nFinal Results: {success_count}/{total_count} files processed successfully")
    
    if success_count > 0:
        safe_print("Check the generated question*LLMAnswer.png files!")
        try:
            # Show the first successful result
            first_success = next(r for r in results if r["status"] == "success")
            if first_success["output"]:
                Image.open(first_success["output"]).show()
        except:
            pass


if __name__ == "__main__":
    main()