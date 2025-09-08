import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from tool import tool_functions, execute_tool_function
import threading
from queue import Queue
import re
from openai import OpenAI


INPUTS_FOLDER = "inputs"  
DELAY_BETWEEN_CALLS = 2  
DELAY_BETWEEN_FILES = 1  
MAX_RECOMMENDATION_RETRIES = 5  


load_dotenv()
MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "gpt-4.1-mini")


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


log_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with log_lock:
        print(*args, **kwargs)

def safe_api_call(func, *args, **kwargs):
    """ API call wrapper with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = func(*args, **kwargs)
            # Validate response has content
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content and len(content.strip()) > 10:
                    return response
                else:
                    safe_print(f"WARNING: Empty/short response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
            return response
        except Exception as e:
            safe_print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise e
    return None

def extract_recommendation(analysis_text, recommendation_type):
    """Extract single recommendation from analysis text"""
    if not analysis_text:
        return None
    
    patterns = {
        "CONFIG_TYPE": [
            r"RECOMMENDED CONFIG_TYPE:\s*([^\n\r,]+)",
            r"CONFIG_TYPE:\s*([^\n\r,]+)",
            r"configuration(?:\s+type)?:\s*([^\n\r,]+)"
        ],
        "SHAPES": [
            r"RECOMMENDED SHAPES?:\s*([^\n\r,\[\]]+)",
            r"SHAPES?:\s*([^\n\r,\[\]]+)",
            r"shape:\s*([^\n\r,]+)"
        ],
        "COLORS": [
            r"RECOMMENDED COLORS?:\s*([^\n\r,\[\]]+)",
            r"COLORS?:\s*([^\n\r,\[\]]+)",
            r"color:\s*([^\n\r,]+)"
        ],
        "SIZES": [
            r"RECOMMENDED SIZES?:\s*([^\n\r,\[\]]+)",
            r"SIZES?:\s*([^\n\r,\[\]]+)",
            r"size:\s*([^\n\r,]+)"
        ],
        "ANGLES": [
            r"RECOMMENDED ANGLES?:\s*([^\n\r,\[\]]+)",
            r"ANGLES?:\s*([^\n\r,\[\]]+)",
            r"angle:\s*([^\n\r,]+)"
        ]
    }
    
    if recommendation_type not in patterns:
        return None
    
    for pattern in patterns[recommendation_type]:
        match = re.search(pattern, analysis_text, re.IGNORECASE)
        if match:
            result = match.group(1).strip().strip('"\'[]')
            # Clean up common separators and take only the first value
            result = result.split(',')[0].split(';')[0].split('|')[0].strip()
            if result and result.lower() not in ['none', 'null', '']:
                return result
    
    return None

class RavenAgentSystem:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.agent_analyses = {}
        self.worker_id = threading.current_thread().name
        self.timestamp = datetime.now().isoformat()
        
        # Load the puzzle data
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.raw_json_data = f.read()
        
        # Validate JSON data
        if not self.raw_json_data or len(self.raw_json_data.strip()) < 10:
            raise ValueError(f"Invalid or empty JSON data in {json_file_path}")

    def analyze_config_types_with_retry(self):
        """SLAVE AGENT 1: Analyze configuration patterns with retry until recommendation found"""
        safe_print(f"[{self.worker_id}] Slave Agent 1: Analyzing configuration types...")
        
        for attempt in range(MAX_RECOMMENDATION_RETRIES):
            safe_print(f"[{self.worker_id}] Config Agent - Attempt {attempt + 1}/{MAX_RECOMMENDATION_RETRIES}")
            
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

IMPORTANT: 
- Do NOT analyze shapes, colors, sizes, or angles. Focus ONLY on spatial arrangements.
- You MUST recommend exactly ONE configuration type for panel 3_3.
- Your response MUST end with: "RECOMMENDED CONFIG_TYPE: [single_exact_type_name]"

Provide your analysis and conclude with exactly one configuration type recommendation.
"""

            try:
                response = safe_api_call(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a configuration pattern specialist. Focus ONLY on spatial arrangements and layout patterns. You MUST provide exactly ONE configuration type recommendation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                analysis = ""
                if response and response.choices and response.choices[0].message:
                    analysis = response.choices[0].message.content or ""
                
                # Extract recommendation
                recommendation = extract_recommendation(analysis, "CONFIG_TYPE")
                
                if recommendation:
                    safe_print(f"[{self.worker_id}] Config analysis successful with recommendation: {recommendation}")
                    
                    self.agent_analyses["config_agent"] = {
                        "analysis": analysis,
                        "recommendation": recommendation,
                        "specialization": "configuration_patterns",
                        "analysis_length": len(analysis),
                        "timestamp": datetime.now().isoformat(),
                        "attempts_needed": attempt + 1,
                        "quality_check": {
                            "has_meaningful_content": len(analysis) > 50,
                            "has_recommendation": True,
                            "recommendation_value": recommendation,
                            "extraction_successful": True
                        }
                    }
                    
                    time.sleep(DELAY_BETWEEN_CALLS)
                    return analysis
                else:
                    safe_print(f"[{self.worker_id}] Config analysis attempt {attempt + 1} failed - no single recommendation found")
                    if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                safe_print(f"[{self.worker_id}] Config agent attempt {attempt + 1} error: {e}")
                if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                    time.sleep(2)
                    continue
        
        # All attempts failed
        error_analysis = f"ERROR: Config analysis failed after {MAX_RECOMMENDATION_RETRIES} attempts - no single valid recommendation found"
        safe_print(f"[{self.worker_id}] {error_analysis}")
        
        self.agent_analyses["config_agent"] = {
            "analysis": error_analysis,
            "recommendation": None,
            "specialization": "configuration_patterns",
            "analysis_length": len(error_analysis),
            "timestamp": datetime.now().isoformat(),
            "attempts_needed": MAX_RECOMMENDATION_RETRIES,
            "quality_check": {
                "has_meaningful_content": False,
                "has_recommendation": False,
                "extraction_successful": False,
                "error_occurred": True
            }
        }
        return error_analysis

    def analyze_shapes_with_retry(self):
        """SLAVE AGENT 2: Analyze shape patterns with retry until recommendation found"""
        safe_print(f"[{self.worker_id}] Slave Agent 2: Analyzing shape patterns...")
        
        for attempt in range(MAX_RECOMMENDATION_RETRIES):
            safe_print(f"[{self.worker_id}] Shape Agent - Attempt {attempt + 1}/{MAX_RECOMMENDATION_RETRIES}")
            
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

IMPORTANT: 
- Do NOT analyze configurations, colors, sizes, or angles. Focus ONLY on shape types.
- You MUST recommend exactly ONE shape for panel 3_3.
- Your response MUST end with: "RECOMMENDED SHAPE: [single_shape_name]"

Provide your analysis and conclude with exactly one shape recommendation.
"""

            try:
                response = safe_api_call(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a shape pattern specialist. Focus ONLY on shape types and transformations. You MUST provide exactly ONE shape recommendation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                analysis = ""
                if response and response.choices and response.choices[0].message:
                    analysis = response.choices[0].message.content or ""
                
                # Extract recommendation
                recommendation = extract_recommendation(analysis, "SHAPES")
                
                if recommendation:
                    safe_print(f"[{self.worker_id}] Shape analysis successful with recommendation: {recommendation}")
                    
                    self.agent_analyses["shape_agent"] = {
                        "analysis": analysis,
                        "recommendation": recommendation,
                        "specialization": "shape_patterns",
                        "analysis_length": len(analysis),
                        "timestamp": datetime.now().isoformat(),
                        "attempts_needed": attempt + 1,
                        "quality_check": {
                            "has_meaningful_content": len(analysis) > 50,
                            "has_recommendation": True,
                            "recommendation_value": recommendation,
                            "extraction_successful": True
                        }
                    }
                    
                    time.sleep(DELAY_BETWEEN_CALLS)
                    return analysis
                else:
                    safe_print(f"[{self.worker_id}] Shape analysis attempt {attempt + 1} failed - no single recommendation found")
                    if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                safe_print(f"[{self.worker_id}] Shape agent attempt {attempt + 1} error: {e}")
                if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                    time.sleep(2)
                    continue
        
        # All attempts failed
        error_analysis = f"ERROR: Shape analysis failed after {MAX_RECOMMENDATION_RETRIES} attempts - no single valid recommendation found"
        safe_print(f"[{self.worker_id}] {error_analysis}")
        
        self.agent_analyses["shape_agent"] = {
            "analysis": error_analysis,
            "recommendation": None,
            "specialization": "shape_patterns",
            "analysis_length": len(error_analysis),
            "timestamp": datetime.now().isoformat(),
            "attempts_needed": MAX_RECOMMENDATION_RETRIES,
            "quality_check": {
                "has_meaningful_content": False,
                "has_recommendation": False,
                "extraction_successful": False,
                "error_occurred": True
            }
        }
        return error_analysis

    def analyze_colors_with_retry(self):
        """SLAVE AGENT 3: Analyze color patterns with retry until recommendation found"""
        safe_print(f"[{self.worker_id}] Slave Agent 3: Analyzing color patterns...")
        
        for attempt in range(MAX_RECOMMENDATION_RETRIES):
            safe_print(f"[{self.worker_id}] Color Agent - Attempt {attempt + 1}/{MAX_RECOMMENDATION_RETRIES}")
            
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

IMPORTANT: 
- Do NOT analyze configurations, shapes, sizes, or angles. Focus ONLY on color values.
- You MUST recommend exactly ONE color for panel 3_3.
- Your response MUST end with: "RECOMMENDED COLOR: [single_color_value]"

Provide your analysis and conclude with exactly one color recommendation.
"""

            try:
                response = safe_api_call(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a color pattern specialist. Focus ONLY on color values and progressions. You MUST provide exactly ONE color recommendation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                analysis = ""
                if response and response.choices and response.choices[0].message:
                    analysis = response.choices[0].message.content or ""
                
                # Extract recommendation
                recommendation = extract_recommendation(analysis, "COLORS")
                
                if recommendation:
                    safe_print(f"[{self.worker_id}] Color analysis successful with recommendation: {recommendation}")
                    
                    self.agent_analyses["color_agent"] = {
                        "analysis": analysis,
                        "recommendation": recommendation,
                        "specialization": "color_patterns",
                        "analysis_length": len(analysis),
                        "timestamp": datetime.now().isoformat(),
                        "attempts_needed": attempt + 1,
                        "quality_check": {
                            "has_meaningful_content": len(analysis) > 50,
                            "has_recommendation": True,
                            "recommendation_value": recommendation,
                            "extraction_successful": True
                        }
                    }
                    
                    time.sleep(DELAY_BETWEEN_CALLS)
                    return analysis
                else:
                    safe_print(f"[{self.worker_id}] Color analysis attempt {attempt + 1} failed - no single recommendation found")
                    if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                safe_print(f"[{self.worker_id}] Color agent attempt {attempt + 1} error: {e}")
                if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                    time.sleep(2)
                    continue
        
        # All attempts failed
        error_analysis = f"ERROR: Color analysis failed after {MAX_RECOMMENDATION_RETRIES} attempts - no single valid recommendation found"
        safe_print(f"[{self.worker_id}] {error_analysis}")
        
        self.agent_analyses["color_agent"] = {
            "analysis": error_analysis,
            "recommendation": None,
            "specialization": "color_patterns",
            "analysis_length": len(error_analysis),
            "timestamp": datetime.now().isoformat(),
            "attempts_needed": MAX_RECOMMENDATION_RETRIES,
            "quality_check": {
                "has_meaningful_content": False,
                "has_recommendation": False,
                "extraction_successful": False,
                "error_occurred": True
            }
        }
        return error_analysis

    def analyze_sizes_with_retry(self):
        """SLAVE AGENT 4: Analyze size patterns with retry until recommendation found"""
        safe_print(f"[{self.worker_id}] Slave Agent 4: Analyzing size patterns...")
        
        for attempt in range(MAX_RECOMMENDATION_RETRIES):
            safe_print(f"[{self.worker_id}] Size Agent - Attempt {attempt + 1}/{MAX_RECOMMENDATION_RETRIES}")
            
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

IMPORTANT: 
- Do NOT analyze configurations, shapes, colors, or angles. Focus ONLY on size values.
- You MUST provide a clear recommendation at the end.
- Your response MUST end with: "RECOMMENDED SIZES: [size1, size2, ...]" or "RECOMMENDED SIZES: [single_size]"

Provide your analysis and conclude with the required recommendation format.
"""

            try:
                response = safe_api_call(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a size pattern specialist. Focus ONLY on size values and scaling patterns. You MUST provide exactly ONE size recommendation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                analysis = ""
                if response and response.choices and response.choices[0].message:
                    analysis = response.choices[0].message.content or ""
                
                # Extract recommendation
                recommendation = extract_recommendation(analysis, "SIZES")
                
                if recommendation:
                    safe_print(f"[{self.worker_id}] Size analysis successful with recommendation: {recommendation}")
                    
                    self.agent_analyses["size_agent"] = {
                        "analysis": analysis,
                        "recommendation": recommendation,
                        "specialization": "size_patterns",
                        "analysis_length": len(analysis),
                        "timestamp": datetime.now().isoformat(),
                        "attempts_needed": attempt + 1,
                        "quality_check": {
                            "has_meaningful_content": len(analysis) > 50,
                            "has_recommendation": True,
                            "recommendation_value": recommendation,
                            "extraction_successful": True
                        }
                    }
                    
                    time.sleep(DELAY_BETWEEN_CALLS)
                    return analysis
                else:
                    safe_print(f"[{self.worker_id}] Size analysis attempt {attempt + 1} failed - no single recommendation found")
                    if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                safe_print(f"[{self.worker_id}] Size agent attempt {attempt + 1} error: {e}")
                if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                    time.sleep(2)
                    continue
        
        # All attempts failed
        error_analysis = f"ERROR: Size analysis failed after {MAX_RECOMMENDATION_RETRIES} attempts - no single valid recommendation found"
        safe_print(f"[{self.worker_id}] {error_analysis}")
        
        self.agent_analyses["size_agent"] = {
            "analysis": error_analysis,
            "recommendation": None,
            "specialization": "size_patterns",
            "analysis_length": len(error_analysis),
            "timestamp": datetime.now().isoformat(),
            "attempts_needed": MAX_RECOMMENDATION_RETRIES,
            "quality_check": {
                "has_meaningful_content": False,
                "has_recommendation": False,
                "extraction_successful": False,
                "error_occurred": True
            }
        }
        return error_analysis

    def analyze_angles_with_retry(self):
        """SLAVE AGENT 5: Analyze angle patterns with retry until recommendation found"""
        safe_print(f"[{self.worker_id}] Slave Agent 5: Analyzing angle patterns...")
        
        for attempt in range(MAX_RECOMMENDATION_RETRIES):
            safe_print(f"[{self.worker_id}] Angle Agent - Attempt {attempt + 1}/{MAX_RECOMMENDATION_RETRIES}")
            
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

IMPORTANT: 
- Do NOT analyze configurations, shapes, colors, or sizes. Focus ONLY on rotation values.
- You MUST recommend exactly ONE angle for panel 3_3.
- Your response MUST end with: "RECOMMENDED ANGLE: [single_angle_value]"

Provide your analysis and conclude with exactly one angle recommendation.
"""

            try:
                response = safe_api_call(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a rotation pattern specialist. Focus ONLY on rotation values and transformations. You MUST provide exactly ONE angle recommendation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                analysis = ""
                if response and response.choices and response.choices[0].message:
                    analysis = response.choices[0].message.content or ""
                
                # Extract recommendation
                recommendation = extract_recommendation(analysis, "ANGLES")
                
                if recommendation:
                    safe_print(f"[{self.worker_id}] Angle analysis successful with recommendation: {recommendation}")
                    
                    self.agent_analyses["angle_agent"] = {
                        "analysis": analysis,
                        "recommendation": recommendation,
                        "specialization": "rotation_patterns",
                        "analysis_length": len(analysis),
                        "timestamp": datetime.now().isoformat(),
                        "attempts_needed": attempt + 1,
                        "quality_check": {
                            "has_meaningful_content": len(analysis) > 50,
                            "has_recommendation": True,
                            "recommendation_value": recommendation,
                            "extraction_successful": True
                        }
                    }
                    
                    time.sleep(DELAY_BETWEEN_CALLS)
                    return analysis
                else:
                    safe_print(f"[{self.worker_id}] Angle analysis attempt {attempt + 1} failed - no single recommendation found")
                    if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                        time.sleep(2)
                        continue
                
            except Exception as e:
                safe_print(f"[{self.worker_id}] Angle agent attempt {attempt + 1} error: {e}")
                if attempt < MAX_RECOMMENDATION_RETRIES - 1:
                    time.sleep(2)
                    continue
        
        # All attempts failed
        error_analysis = f"ERROR: Angle analysis failed after {MAX_RECOMMENDATION_RETRIES} attempts - no single valid recommendation found"
        safe_print(f"[{self.worker_id}] {error_analysis}")
        
        self.agent_analyses["angle_agent"] = {
            "analysis": error_analysis,
            "recommendation": None,
            "specialization": "rotation_patterns",
            "analysis_length": len(error_analysis),
            "timestamp": datetime.now().isoformat(),
            "attempts_needed": MAX_RECOMMENDATION_RETRIES,
            "quality_check": {
                "has_meaningful_content": False,
                "has_recommendation": False,
                "extraction_successful": False,
                "error_occurred": True
            }
        }
        return error_analysis

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
        
        # Calculate detailed agent analysis metrics with quality validation
        total_analysis_length = 0
        successful_agents = 0
        failed_agents = 0
        total_attempts = 0
        successful_recommendations = 0
        
        for data in self.agent_analyses.values():
            analysis_len = data.get("analysis_length", 0)
            total_analysis_length += analysis_len
            total_attempts += data.get("attempts_needed", 0)
            
            if data.get("quality_check", {}).get("extraction_successful", False):
                successful_agents += 1
            else:
                failed_agents += 1
                
            if data.get("recommendation"):
                successful_recommendations += 1
        
        agent_specializations = [data["specialization"] for data in self.agent_analyses.values()]
        
        agent_data = {
            "puzzle_file": self.json_file_path,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "model_used": MODEL_NAME,
            "processing_type": "multi_agent_with_retry_logic",
            "agent_analyses": self.agent_analyses,
            "analysis_summary": {
                "total_agents": len(self.agent_analyses),
                "successful_agents": successful_agents,
                "failed_agents": failed_agents,
                "successful_recommendations": successful_recommendations,
                "agents_completed": list(self.agent_analyses.keys()),
                "total_analysis_length": total_analysis_length,
                "total_attempts_needed": total_attempts,
                "average_attempts_per_agent": total_attempts / len(self.agent_analyses) if self.agent_analyses else 0,
                "specializations": agent_specializations,
                "analysis_completeness": len(self.agent_analyses) == 5,
                "recommendation_success_rate": successful_recommendations / len(self.agent_analyses) if self.agent_analyses else 0,
                "quality_metrics": {
                    "all_agents_successful": successful_agents == 5,
                    "all_recommendations_found": successful_recommendations == 5,
                    "retry_efficiency": total_attempts / (len(self.agent_analyses) * MAX_RECOMMENDATION_RETRIES) if self.agent_analyses else 0
                }
            }
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False)
        
        safe_print(f"[{self.worker_id}] Agent analyses saved to: {filename}")
        safe_print(f"[{self.worker_id}] Recommendations: {successful_recommendations}/{len(self.agent_analyses)} found")
        return filename

    def master_solver(self):
        """MASTER LLM: Synthesizes slave analyses and generates solution"""
        safe_print(f"[{self.worker_id}] Master Agent: Synthesizing solution from slave analyses...")
        
        # Check if we have all agent recommendations
        missing_recommendations = []
        for agent_name, data in self.agent_analyses.items():
            if not data.get("recommendation"):
                missing_recommendations.append(agent_name)
        
        if missing_recommendations:
            error_msg = f"Cannot proceed - missing recommendations from: {missing_recommendations}"
            safe_print(f"[{self.worker_id}] ERROR: {error_msg}")
            return {"status": "error", "message": error_msg}
        
        # Compile all agent insights with recommendations
        agent_insights = ""
        for agent_name, data in self.agent_analyses.items():
            analysis_text = data.get('analysis', '') or ''
            recommendation = data.get('recommendation', 'No recommendation')
            specialization = data.get('specialization', 'unknown')
            attempts = data.get('attempts_needed', 1)
            
            agent_insights += f"""
=== {agent_name.upper()} ({specialization.upper()}) - Attempts: {attempts} ===
RECOMMENDATION: {recommendation}

ANALYSIS:
{analysis_text}
"""
        
        # MASTER PROMPT
        prompt = f"""RAVEN'S MATRIX MASTER SOLVER

You are the master coordinator that synthesizes specialist analyses to solve a 3x3 Raven's matrix puzzle.

You have detailed analyses from 5 specialist agents who each focused on one specific aspect:

SPECIALIST AGENT ANALYSES WITH RECOMMENDATIONS:
{agent_insights}

YOUR TASK:
1. EXTRACT RECOMMENDATIONS: Use the exact recommendations from each specialist
2. SYNTHESIZE SOLUTION: Combine all specialist recommendations into a complete solution
3. RESOLVE CONFLICTS: If agents have conflicting recommendations, choose the most logical
4. GENERATE SOLUTION: Call generate_visual_panel with the synthesized parameters

SYNTHESIS STEPS:
1. Extract the single CONFIG_TYPE recommendation from the Configuration Agent
2. Extract the single SHAPE recommendation from the Shape Agent
3. Extract the single COLOR recommendation from the Color Agent  
4. Extract the single SIZE recommendation from the Size Agent
5. Extract the single ANGLE recommendation from the Angle Agent
6. Call generate_visual_panel with these exact single parameters

Use the generate_visual_panel function with parameters based ENTIRELY on what the specialists recommend.
Each agent provides exactly ONE value per feature.
You MUST call the function - do not provide examples or pseudo-code."""

        system_prompt = """You are an expert Raven's Progressive Matrices master solver with access to a powerful visual generation tool.

You synthesize analyses from specialist agents who have examined different pattern aspects and provided specific recommendations.

Think step by step to synthesize all specialist recommendations, then generate the solution using the generate_visual_panel function.
You MUST call the generate_visual_panel function - do not provide examples or pseudo-code."""

        try:
            time.sleep(DELAY_BETWEEN_CALLS)
            
            response = safe_api_call(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                tools=tool_functions,
                tool_choice="required",
                temperature=0.1,
                max_tokens=3000
            )

            if not response:
                return {"status": "error", "message": "Master API call failed after retries"}

            message = response.choices[0].message
            cot_reasoning = message.content if message and message.content else ""

            if message.tool_calls:
                for call in message.tool_calls:
                    function_name = call.function.name
                    try:
                        arguments = json.loads(call.function.arguments)
                    except json.JSONDecodeError as e:
                        return {"status": "error", "message": f"Failed to parse function arguments: {e}"}

                    # Generate unique filename for master CoT
                    base_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
                    number_match = re.search(r'\d+', base_name)
                    if number_match:
                        question_number = number_match.group()
                        master_cot_filename = f"master_synthesis_cot_reasoning_question{question_number}.json"
                        llm_output_filename = f"llm_output_question{question_number}.json"
                    else:
                        master_cot_filename = f"master_synthesis_cot_reasoning_{base_name}.json"
                        llm_output_filename = f"llm_output_{base_name}.json"

                    # Save master CoT reasoning
                    master_cot_data = {
                        "puzzle_file": self.json_file_path,
                        "worker_id": self.worker_id,
                        "timestamp": self.timestamp,
                        "model_used": MODEL_NAME,
                        "processing_type": "multi_agent_with_retry_synthesis",
                        "solving_session": {
                            "raw_llm_response": message.content,
                            "cot_reasoning": cot_reasoning,
                            "reasoning_length": len(cot_reasoning),
                            "function_called": function_name,
                            "function_arguments": arguments,
                            "tool_call_id": call.id,
                            "success": True
                        },
                        "agent_recommendations_used": {
                            agent_name: data.get("recommendation") 
                            for agent_name, data in self.agent_analyses.items()
                        },
                        "synthesis_quality": {
                            "all_recommendations_available": all(data.get("recommendation") for data in self.agent_analyses.values()),
                            "synthesis_reasoning_captured": len(cot_reasoning) > 50,
                            "function_call_successful": True
                        }
                    }

                    with open(master_cot_filename, "w", encoding="utf-8") as cot_file:
                        json.dump(master_cot_data, cot_file, indent=2, ensure_ascii=False)

                    # Save LLM function arguments
                    with open(llm_output_filename, "w", encoding="utf-8") as out:
                        json.dump({
                            "worker_id": self.worker_id,
                            "function_arguments": arguments,
                            "note": f"Master synthesis details in {master_cot_filename}"
                        }, out, indent=2, ensure_ascii=False)

                    # Execute the tool
                    result = execute_tool_function(function_name, arguments, self.json_file_path)
                    
                    # Update master CoT file with execution result
                    master_cot_data["execution_result"] = {
                        "status": result.get("status", "unknown"),
                        "filename": result.get("filename", ""),
                        "execution_successful": result.get("status") == "success"
                    }
                    
                    with open(master_cot_filename, "w", encoding="utf-8") as cot_file:
                        json.dump(master_cot_data, cot_file, indent=2, ensure_ascii=False)
                    
                    safe_print(f"[{self.worker_id}] Master synthesis complete")
                    return result
                        
            else:
                error_msg = "Master LLM did not call the tool function"
                safe_print(f"[{self.worker_id}] ERROR: {error_msg}")
                return {"status": "error", "message": error_msg}
        
        except Exception as e:
            safe_print(f"[{self.worker_id}] Master solver error: {e}")
            return {"status": "error", "message": f"Master solver exception: {str(e)}"}


def get_json_files_from_inputs():
    """Get all questionText*.json files from inputs folder, sorted by number"""
    if not os.path.exists(INPUTS_FOLDER):
        safe_print(f"Error: {INPUTS_FOLDER} folder not found!")
        return []
    
    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            json_files.append(filepath)
    
    # Sort by the number in filename
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    json_files.sort(key=extract_number)
    return json_files


def solve_raven_from_json(json_file_path):
    """Multi-agent RAVEN solver with retry logic for recommendations"""
    try:
        if not os.path.exists(json_file_path):
            return {"status": "error", "message": f"JSON file not found: {json_file_path}"}
        
        safe_print(f"[{threading.current_thread().name}] Starting Multi-Agent Raven Solver with Retry Logic...")
        agent_system = RavenAgentSystem(json_file_path)
        
        # PHASE 1: Slave agents analyze with retry until recommendations found
        safe_print(f"[{threading.current_thread().name}] Phase 1: Slave agents analyzing with retry logic...")
        agent_system.analyze_config_types_with_retry()
        agent_system.analyze_shapes_with_retry()
        agent_system.analyze_colors_with_retry()
        agent_system.analyze_sizes_with_retry()
        agent_system.analyze_angles_with_retry()
        
        # Save all slave analyses
        analysis_file = agent_system.save_agent_analyses()
        
        # PHASE 2: Master synthesizes slave analyses
        safe_print(f"[{threading.current_thread().name}] Phase 2: Master synthesizing recommendations...")
        result = agent_system.master_solver()
        
        return result

    except Exception as e:
        safe_print(f"Exception in solve_raven_from_json: {e}")
        return {"status": "error", "message": str(e)}


def process_single_file(json_file_path, file_index, total_files):
    """Process a single JSON file with progress tracking"""
    filename = os.path.basename(json_file_path)
    
    safe_print(f"\n[{file_index}/{total_files}] Processing: {filename}")
    
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


def process_all_json_files():
    """Process all JSON files sequentially (no batching)"""
    json_files = get_json_files_from_inputs()
    
    if not json_files:
        safe_print("No questionText*.json files found in inputs folder!")
        return
    
    safe_print(f"Found {len(json_files)} JSON files to process")
    safe_print("Processing files sequentially with retry logic for recommendations...")
    
    start_time = time.time()
    results_summary = []
    
    for i, json_file_path in enumerate(json_files):
        result = process_single_file(json_file_path, i + 1, len(json_files))
        results_summary.append(result)
        
        # Delay between files
        if i < len(json_files) - 1:
            safe_print(f"Waiting {DELAY_BETWEEN_FILES}s before next file...")
            time.sleep(DELAY_BETWEEN_FILES)
    
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
            "processing_type": "multi_agent_with_retry_sequential",
            "max_recommendation_retries": MAX_RECOMMENDATION_RETRIES,
            "total_time_seconds": elapsed_time,
            "average_time_per_file": elapsed_time / total_files if total_files else 0,
            "success_rate": successful / total_files if total_files else 0
        },
        "results": results_summary
    }
    
    with open("multi_agent_retry_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    safe_print(f"\n{'='*60}")
    safe_print(f"Total files processed: {total_files}")
    safe_print(f"Successful: {successful}")
    safe_print(f"Failed: {failed}")
    safe_print(f"Success rate: {successful/total_files*100:.1f}%")
    safe_print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    safe_print(f"Average time per file: {elapsed_time/total_files:.2f}s")
    safe_print("Summary saved to: multi_agent_retry_summary.json")
    
    return results_summary


def main():
    """Multi-agent RAVEN solver with recommendation retry logic"""
    safe_print("Architecture: 5 Specialist Agents + 1 Master Synthesizer")
    safe_print(f"Max recommendation retries: {MAX_RECOMMENDATION_RETRIES}")
    safe_print(f"Processing mode: Sequential (no batching)")
    safe_print(f"Model: {MODEL_NAME}")
    safe_print("")

    
    if not os.getenv("OPENAI_API_KEY"):
        safe_print("[!] Missing OPENAI_API_KEY in .env file")
        return
    
    # Process all JSON files
    results = process_all_json_files()
    
    if not results:
        return
    
    # Show final statistics
    success_count = len([r for r in results if r["status"] == "success"])
    total_count = len(results)
    
    safe_print(f"\nFinal Results: {success_count}/{total_count} files processed successfully")
    
    if success_count > 0:
        safe_print("Check the generated question*LLMAnswer.png files!")


if __name__ == "__main__":
    main()
