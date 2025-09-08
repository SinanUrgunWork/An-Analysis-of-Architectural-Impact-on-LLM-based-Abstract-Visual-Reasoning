import os
import json
import time
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from collections import OrderedDict
from tool import tool_functions, execute_tool_function
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from google.protobuf.json_format import MessageToDict

INPUTS_FOLDER = "inputs"
CHECK_INTERVAL = 10
MAX_ATTEMPTS = 3
CONFIDENCE_THRESHOLD = 8.0
DELAY_BETWEEN_REFLECTIONS = 0.05

CHUNK_SIZE = 200
MAX_CONCURRENT_FILES = 12

API_RETRY_ATTEMPTS = 2
API_RETRY_DELAY = 1.0
API_TIMEOUT = 30

load_dotenv()
MODEL_NAME = "gemini-1.5-flash"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_gemini_tools():
    return [{
        "function_declarations": [{
            "name": "generate_visual_panel",
            "description": "Generate a RAVEN matrix answer panel with specified configuration and objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_type": {
                        "type": "string",
                        "description": "Panel configuration type",
                        "enum": ["singleton_center", "left_right", "up_down", "out_in", 
                                "distribute_three", "distribute_four", "grid_2x2", 
                                "distribute_nine", "grid_3x3"]
                    },
                    "objects": {
                        "type": "array",
                        "description": "Array of objects to place in the panel",
                        "items": {
                            "type": "object",
                            "properties": {
                                "shape": {
                                    "type": "string",
                                    "description": "Shape type",
                                    "enum": ["triangle", "square", "pentagon", "hexagon", 
                                            "heptagon", "circle", "line", "none"]
                                },
                                "size": {
                                    "type": "number",
                                    "description": "Size of the shape (0.1 to 1.0)"
                                },
                                "color": {
                                    "type": "integer",
                                    "description": "Color value (0=white to 9=black)"
                                },
                                "angle": {
                                    "type": "integer",
                                    "description": "Rotation angle (-180 to 180)"
                                }
                            },
                            "required": ["shape", "size", "color", "angle"]
                        }
                    },
                    "grid_layout": {
                        "type": "string",
                        "description": "Optional grid layout",
                        "enum": ["1x1", "1x2", "2x1", "2x2", "3x3", "1x3", "3x1"]
                    },
                    "positions": {
                        "type": "array",
                        "description": "Optional array of position indices",
                        "items": {
                            "type": "integer"
                        }
                    },
                    "empty_positions": {
                        "type": "array",
                        "description": "Optional array of empty position indices",
                        "items": {
                            "type": "integer"
                        }
                    }
                },
                "required": ["config_type", "objects"]
            }
        }]
    }]


def extract_function_args(function_call):
    arguments = {}
    
    try:
        if hasattr(function_call, '_pb'):
            call_dict = MessageToDict(function_call._pb)
            arguments = call_dict.get('args', {})
            if arguments and 'config_type' in arguments:
                return arguments
    except Exception as e:
        pass
    
    try:
        if hasattr(function_call, 'args'):
            if hasattr(function_call.args, '_pb'):
                arguments = MessageToDict(function_call.args._pb)
                if arguments and 'config_type' in arguments:
                    return arguments
    except Exception as e:
        pass
    
    try:
        if hasattr(function_call, 'args'):
            args = function_call.args
            
            if hasattr(args, 'config_type'):
                arguments['config_type'] = str(args.config_type)
            
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
            
            if hasattr(args, 'grid_layout'):
                arguments['grid_layout'] = str(args.grid_layout)
            if hasattr(args, 'positions'):
                arguments['positions'] = list(args.positions)
            if hasattr(args, 'empty_positions'):
                arguments['empty_positions'] = list(args.empty_positions)
                
        if arguments and 'config_type' in arguments:
            return arguments
            
    except Exception as e:
        pass
    
    try:
        func_str = str(function_call)
        config_match = re.search(r'config_type["\']?\s*:\s*["\']([^"\']+)["\']', func_str)
        if config_match:
            arguments['config_type'] = config_match.group(1)
            
    except Exception as e:
        pass
    
    return arguments


class ReflectionCache:
    def __init__(self, max_size=1500):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.lock = threading.Lock()
    
    def get_key(self, reasoning, function_call):
        combined = f"{reasoning}|||{json.dumps(function_call, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, reasoning, function_call):
        with self.lock:
            key = self.get_key(reasoning, function_call)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self.cache[key]
            
            self.stats['misses'] += 1
            return None
    
    def set(self, reasoning, function_call, reflection_result):
        with self.lock:
            key = self.get_key(reasoning, function_call)
            
            if key in self.cache:
                self.cache[key] = reflection_result
                self.cache.move_to_end(key)
                return
            
            if len(self.cache) >= self.max_size:
                removed_key = next(iter(self.cache))
                del self.cache[removed_key]
                self.stats['evictions'] += 1
            
            self.cache[key] = reflection_result
    
    def get_stats(self):
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': f"{hit_rate:.2%}",
                'total_requests': total_requests
            }


reflection_cache = ReflectionCache()


def get_json_files_from_inputs():
    if not os.path.exists(INPUTS_FOLDER):
        print(f"Error: {INPUTS_FOLDER} folder not found!")
        return []
    
    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            if os.path.getsize(filepath) > 0:
                json_files.append(filepath)
    
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    json_files.sort(key=extract_number)
    return json_files


def chunk_files(json_files, chunk_size):
    chunks = []
    for i in range(0, len(json_files), chunk_size):
        chunk = json_files[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def call_gemini_with_tools(user_prompt, system_prompt, model_name=MODEL_NAME):
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            model = genai.GenerativeModel(model_name)
            gemini_tools = get_gemini_tools()
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = model.generate_content(
                full_prompt,
                tools=gemini_tools,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                    candidate_count=1,
                )
            )
            
            cot_reasoning = ""
            function_calls = []
            
            if not response or not response.candidates:
                raise ValueError("No response candidates from Gemini")
                
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("No content parts in response")
            
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    cot_reasoning += part.text
                elif hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
            
            if not cot_reasoning:
                cot_reasoning = "No reasoning content in response"
            
            function_call = None
            function_called = None
            if function_calls:
                function_called = function_calls[0].name
                function_call = extract_function_args(function_calls[0])
            
            return {
                "cot_reasoning": cot_reasoning,
                "function_called": function_called,
                "function_call": function_call,
                "success": bool(function_call and function_call.get('config_type'))
            }
            
        except Exception as e:
            if attempt < API_RETRY_ATTEMPTS - 1:
                print(f"Gemini API retry {attempt + 1}/{API_RETRY_ATTEMPTS}: {e}")
                time.sleep(API_RETRY_DELAY)
                continue
            else:
                print(f"Gemini API failed after {API_RETRY_ATTEMPTS} attempts: {e}")
                return {
                    "cot_reasoning": f"API Error: {str(e)}",
                    "function_called": None,
                    "function_call": None,
                    "success": False,
                    "error": str(e)
                }


def perform_self_reflection(reasoning, function_call, question_number):
    cached_reflection = reflection_cache.get(reasoning, function_call)
    if cached_reflection is not None:
        return cached_reflection
    
    json_file_path = os.path.join(INPUTS_FOLDER, f"questionText{question_number}.json")
    if not os.path.exists(json_file_path):
        return {
            "status": "error",
            "confidence_score": 0,
            "recommendation": "REJECT",
            "error": "Original puzzle file not found"
        }
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_json_data = f.read()
    
    reflection_prompt = f"""SELF-REFLECTION ON RAVEN'S MATRICES SOLUTION

You are critically evaluating your own solution to a Raven's Progressive Matrices puzzle.

ORIGINAL MATRIX DATA:
{raw_json_data}

YOUR PREVIOUS REASONING:
{reasoning}

YOUR PREVIOUS FUNCTION CALL:
{json.dumps(function_call, indent=2)}

REFLECTION TASKS:
1. PATTERN VERIFICATION: Re-examine the matrix data. Did you correctly identify all patterns?
   - Check row patterns (left to right)
   - Check column patterns (top to bottom)
   - Check diagonal patterns if applicable
   - Verify your pattern description matches the actual data

2. LOGIC VALIDATION: Is your reasoning sound?
   - Are there any logical gaps or assumptions?
   - Did you consider alternative interpretations?
   - Is the pattern you identified consistent across all rows/columns?

3. FUNCTION CALL ACCURACY: Does your function call correctly implement your reasoning?
   - Do the parameters match your described solution?
   - Are all required parameters included (config_type, objects)?
   - Are the values within valid ranges?
   - For each object: shape, size (0.1-1.0), color (0-9), angle (-180 to 180)?

4. ALTERNATIVE SOLUTIONS: Could there be other valid interpretations?
   - What other patterns might exist?
   - How confident are you in your current solution?

5. ERROR DETECTION: Look for potential mistakes:
   - Calculation errors
   - Misinterpreted shapes, colors, or positions
   - Overlooked transformations
   - Wrong config_type selection

PROVIDE YOUR REFLECTION IN THIS EXACT FORMAT:
CONFIDENCE_SCORE: [1-10, where 10 is completely confident]
PATTERN_ACCURACY: [CORRECT/UNCERTAIN/INCORRECT - with brief explanation]
LOGIC_SOUNDNESS: [SOUND/QUESTIONABLE/FLAWED - with brief explanation]
FUNCTION_ACCURACY: [ACCURATE/NEEDS_ADJUSTMENT/INCORRECT - with brief explanation]
ALTERNATIVE_PATTERNS: [List any other possible interpretations]
CRITICAL_ISSUES: [List any problems you found, or "NONE" if no issues]
RECOMMENDATION: [ACCEPT/REVISE/REJECT - with explanation]
REVISED_REASONING: [If recommending revision, provide improved reasoning, otherwise "N/A"]
REVISED_FUNCTION_CALL: [If recommending revision, provide improved function call in JSON format, otherwise "N/A"]"""

    try:
        time.sleep(DELAY_BETWEEN_REFLECTIONS)
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            reflection_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2500,
            )
        )
        
        reflection_text = ""
        if response and response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    reflection_text += part.text
        
        if not reflection_text:
            reflection_text = "No reflection content in response"
        
        reflection_result = parse_reflection_response(reflection_text)
        reflection_cache.set(reasoning, function_call, reflection_result)
        
        return reflection_result
        
    except Exception as e:
        print(f"Error in self-reflection: {e}")
        return {
            "status": "error",
            "confidence_score": 5.0,
            "recommendation": "ACCEPT",
            "error": str(e)
        }


def parse_reflection_response(reflection_text):
    try:
        confidence_match = re.search(r'CONFIDENCE_SCORE:\s*(\d+(?:\.\d+)?)', reflection_text, re.IGNORECASE)
        confidence_score = float(confidence_match.group(1)) if confidence_match else 6.0
        
        pattern_match = re.search(r'PATTERN_ACCURACY:\s*(\w+)', reflection_text, re.IGNORECASE)
        pattern_accuracy = pattern_match.group(1).upper() if pattern_match else "CORRECT"
        
        logic_match = re.search(r'LOGIC_SOUNDNESS:\s*(\w+)', reflection_text, re.IGNORECASE)
        logic_soundness = logic_match.group(1).upper() if logic_match else "SOUND"
        
        function_match = re.search(r'FUNCTION_ACCURACY:\s*(\w+)', reflection_text, re.IGNORECASE)
        function_accuracy = function_match.group(1).upper() if function_match else "ACCURATE"
        
        recommendation_match = re.search(r'RECOMMENDATION:\s*(\w+)', reflection_text, re.IGNORECASE)
        recommendation = recommendation_match.group(1).upper() if recommendation_match else "ACCEPT"
        
        revised_reasoning = None
        revised_function_call = None
        
        revised_reasoning_match = re.search(r'REVISED_REASONING:\s*(.*?)(?=REVISED_FUNCTION_CALL:|$)', reflection_text, re.DOTALL | re.IGNORECASE)
        if revised_reasoning_match and revised_reasoning_match.group(1).strip() not in ["N/A", "n/a", "NA"]:
            revised_reasoning = revised_reasoning_match.group(1).strip()
        
        revised_function_match = re.search(r'REVISED_FUNCTION_CALL:\s*(.*?)(?=\n\n|$)', reflection_text, re.DOTALL | re.IGNORECASE)
        if revised_function_match and revised_function_match.group(1).strip() not in ["N/A", "n/a", "NA"]:
            try:
                json_match = re.search(r'\{.*\}', revised_function_match.group(1), re.DOTALL)
                if json_match:
                    revised_function_call = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {
            "status": "success",
            "confidence_score": confidence_score,
            "pattern_accuracy": pattern_accuracy,
            "logic_soundness": logic_soundness,
            "function_accuracy": function_accuracy,
            "recommendation": recommendation,
            "revised_reasoning": revised_reasoning,
            "revised_function_call": revised_function_call,
            "full_reflection": reflection_text
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "confidence_score": 6.0,
            "recommendation": "ACCEPT",
            "full_reflection": reflection_text
        }


def process_single_file_with_reflection(json_file_path, max_trials=3):
    filename = os.path.basename(json_file_path)
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_json_data = f.read()
        
        number_match = re.search(r'(\d+)', filename)
        question_number = number_match.group(1) if number_match else filename
        
        user_prompt = f"""Solve this 3x3 Raven's matrix. Find the pattern and generate panel 3_3.

{raw_json_data}

Please follow this ANALYSIS FRAMEWORK step by step and show your reasoning:

1. EXAMINE STRUCTURE: Look at each panel's config_type and object count
2. IDENTIFY PATTERNS: Find systematic changes across rows and columns  
3. DETECT TRANSFORMATIONS: Shape changes, color progression, size scaling, rotation
4. APPLY RULE: Determine what panel 3_3 should be
5. GENERATE SOLUTION: Call generate_visual_panel with exact parameters

Think through each step carefully, then call the generate_visual_panel function with:
- config_type: "singleton_center", "left_right", "up_down", "out_in", "distribute_three", "distribute_four", "grid_2x2", "distribute_nine", "grid_3x3"
- objects: array with shape, size, color, angle

Available shapes: "triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"
Colors: 0-9 (0=white, 9=black)
Sizes: 0.1-1.0 
Angles: -180 to 180 

For grid_3x3: provide exactly 9 objects, use "none" for empty spots.

Show your step-by-step reasoning, then call the function."""

        system_prompt = """You are an expert Raven's Progressive Matrices solver with access to a powerful visual generation tool.

IMPORTANT: You must provide step-by-step reasoning first, then call the generate_visual_panel function.

The tool can create ANY arrangement of shapes with precise control over:
- Layout configurations 
- Object positions (specific indices in grids)
- Shape types (7 different shapes + line)
- Colors (10 grayscale levels with multiple naming conventions)
- Sizes (continuous scale from 0.1 to 1.0)
- Rotations (any angle from -180 to 180)

Process:
1. Analyze the matrix patterns step by step
2. Show your reasoning for each step
3. Determine the solution
4. Call generate_visual_panel function with exact parameters

You MUST call the generate_visual_panel function after providing your analysis - do not provide examples or pseudo-code.
Use the exact parameter values and terminology from the input data."""

        result = call_gemini_with_tools(user_prompt, system_prompt)
        
        if not result["success"]:
            cot_filename = f"simple_cot_reasoning_question{question_number}.json"
            cot_data = {
                "puzzle_file": json_file_path,
                "model_used": MODEL_NAME,
                "processing_type": "gemini15_with_reflection",
                "solving_session": {
                    "raw_llm_response": result["cot_reasoning"],
                    "cot_reasoning": result["cot_reasoning"],
                    "reasoning_length": len(result["cot_reasoning"]),
                    "function_called": result["function_called"],
                    "function_arguments": result["function_call"],
                    "success": result["success"],
                    "extraction_method": "gemini15_enhanced",
                    "error": result.get("error", "No function call found")
                }
            }
            
            with open(cot_filename, "w", encoding="utf-8") as f:
                json.dump(cot_data, f, indent=2, ensure_ascii=False)
            
            return {
                "question_number": question_number,
                "status": "no_function_call",
                "cot_reasoning": result["cot_reasoning"],
                "cot_length": len(result["cot_reasoning"]),
                "error": result.get("error", "No function call found")
            }
        
        cot_reasoning = result["cot_reasoning"]
        function_call = result["function_call"]
        
        print(f"[Question {question_number}] CoT length: {len(cot_reasoning)} chars")
        print(f"[Question {question_number}] Function call captured: {result['function_called']}")
        
        final_function_call = function_call
        final_reasoning = cot_reasoning
        was_revised = False
        reflection_result = None
        status = "low_confidence"
        final_confidence = 0
        trial = 1

        while trial <= max_trials:
            reflection_result = perform_self_reflection(final_reasoning, final_function_call, question_number)
            confidence = reflection_result.get('confidence_score', 0)
            recommendation = reflection_result.get('recommendation', 'REJECT')
            print(f"[Question {question_number}] Reflection Trial {trial}: Conf={confidence} | Reco={recommendation}")

            if confidence >= CONFIDENCE_THRESHOLD and recommendation == "ACCEPT":
                status = "accepted"
                final_confidence = confidence
                break
            elif recommendation == "REVISE" and reflection_result.get('revised_function_call'):
                final_function_call = reflection_result['revised_function_call']
                final_reasoning = reflection_result.get('revised_reasoning', final_reasoning)
                was_revised = True
                trial += 1
                continue
            else:
                final_confidence = confidence
                status = "low_confidence"
                break
        else:
            status = "max_trials_reached"
        
        cot_filename = f"simple_cot_reasoning_question{question_number}.json"
        cot_data = {
            "puzzle_file": json_file_path,
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_with_reflection",
            "solving_session": {
                "raw_llm_response": final_reasoning,
                "cot_reasoning": final_reasoning,
                "reasoning_length": len(final_reasoning),
                "function_called": result["function_called"],
                "function_arguments": final_function_call,
                "success": bool(final_function_call),
                "extraction_method": "gemini15_enhanced",
                "confidence_score": final_confidence,
                "validation_method": "Self-Reflection",
                "status": status
            },
            "reasoning_analysis": {
                "has_step1_patterns": any(keyword in final_reasoning.lower() for keyword in ["examine", "structure", "step 1"]),
                "has_step2_rule": any(keyword in final_reasoning.lower() for keyword in ["identify", "pattern", "step 2"]),
                "has_step3_apply": any(keyword in final_reasoning.lower() for keyword in ["detect", "transformation", "step 3"]),
                "has_step4_generate": any(keyword in final_reasoning.lower() for keyword in ["apply", "rule", "step 4"]),
                "has_step5_solution": any(keyword in final_reasoning.lower() for keyword in ["generate", "solution", "step 5"]),
                "mentions_rows_columns": "row" in final_reasoning.lower() and "column" in final_reasoning.lower(),
                "mentions_shapes": any(shape in final_reasoning.lower() for shape in ["triangle", "square", "pentagon", "hexagon", "circle"]),
                "mentions_colors": any(color in final_reasoning.lower() for color in ["black", "gray", "white", "color"]),
                "mentions_positions": "position" in final_reasoning.lower() or "grid" in final_reasoning.lower(),
                "cot_captured_successfully": len(final_reasoning) > 50,
                "gemini15_processing": True
            },
            "self_reflection_analysis": {
                "confidence_score": final_confidence,
                "pattern_accuracy": reflection_result.get('pattern_accuracy', 'N/A'),
                "logic_soundness": reflection_result.get('logic_soundness', 'N/A'),
                "function_accuracy": reflection_result.get('function_accuracy', 'N/A'),
                "recommendation": reflection_result.get('recommendation', 'N/A'),
                "reflection_trials": trial,
                "was_revised": was_revised
            }
        }
        
        with open(cot_filename, "w", encoding="utf-8") as f:
            json.dump(cot_data, f, indent=2, ensure_ascii=False)
        print(f"[Question {question_number}] CoT saved to: {cot_filename}")
        
        if final_function_call is not None:
            final_function_call["source_filename"] = json_file_path
        execute_result = execute_tool_function("generate_visual_panel", final_function_call, json_file_path)
        
        return {
            "question_number": question_number,
            "status": status,
            "confidence_score": final_confidence,
            "cot_reasoning": final_reasoning,
            "cot_length": len(final_reasoning),
            "function_call": final_function_call,
            "reflection_summary": reflection_result,
            "execution_result": execute_result,
            "output_file": execute_result.get("filename") if execute_result.get("status") == "success" else None,
            "was_revised": was_revised,
            "reflection_trials": trial
        }
        
    except Exception as e:
        print(f"[Question {question_number}] ERROR: {str(e)}")
        return {
            "question_number": question_number,
            "status": "processing_error",
            "error": str(e)
        }


def process_chunk_parallel(chunk_files, chunk_id):
    print(f"\n{'='*50}")
    print(f"PROCESSING CHUNK {chunk_id} ({len(chunk_files)} files)")
    print(f"{'='*50}")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as executor:
        future_to_file = {
            executor.submit(process_single_file_with_reflection, json_file_path): json_file_path 
            for json_file_path in chunk_files
        }
        
        for future in as_completed(future_to_file):
            json_file_path = future_to_file[future]
            filename = os.path.basename(json_file_path)
            try:
                result = future.result()
                results.append(result)
                print(f"[Chunk {chunk_id}] {filename} completed")
            except Exception as e:
                print(f"[Chunk {chunk_id}] {filename} error: {e}")
                number_match = re.search(r'(\d+)', filename)
                question_number = number_match.group(1) if number_match else filename
                results.append({
                    "question_number": question_number,
                    "status": "processing_error",
                    "error": str(e)
                })
    
    results.sort(key=lambda x: int(x["question_number"]) if str(x["question_number"]).isdigit() else 0)
    
    print(f"Chunk {chunk_id} completed: {len(results)} results")
    return results


def main():
    print("=== Gemini 1.5 Flash RAVEN Matrix Solver with Self-Reflection ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}/10")
    print(f"Max trials: {MAX_ATTEMPTS}")
    print(f"Chunk size: {CHUNK_SIZE} files per chunk")
    print(f"Max concurrent files per chunk: {MAX_CONCURRENT_FILES}")
    print(f"Delay between reflections: {DELAY_BETWEEN_REFLECTIONS}s")
    print(f"API retry attempts: {API_RETRY_ATTEMPTS}")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Missing GEMINI_API_KEY in .env file")
        return
    
    json_files = get_json_files_from_inputs()
    if not json_files:
        print("No questionText*.json files found in inputs folder!")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    file_chunks = chunk_files(json_files, CHUNK_SIZE)
    print(f"Split into {len(file_chunks)} chunks of ~{CHUNK_SIZE} files each")
    
    all_results = []
    start_time = time.time()
    
    for chunk_id, current_chunk in enumerate(file_chunks, 1):
        chunk_results = process_chunk_parallel(current_chunk, chunk_id)
        all_results.extend(chunk_results)
        
        print(f"Completed {chunk_id}/{len(file_chunks)} chunks")
        print(f"Total results so far: {len(all_results)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    successful = len([r for r in all_results if r.get("status") == "accepted"])
    low_confidence = len([r for r in all_results if r.get("status") == "low_confidence"])
    no_function = len([r for r in all_results if r.get("status") == "no_function_call"])
    errors = len([r for r in all_results if r.get("status") in ["processing_error"]])
    revised = len([r for r in all_results if r.get("was_revised", False)])
    
    confidence_scores = [r.get("confidence_score", 0) for r in all_results if r.get("confidence_score", 0) > 0]
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    summary_data = {
        "processing_session": {
            "total_files": len(json_files),
            "total_chunks": len(file_chunks),
            "chunk_size": CHUNK_SIZE,
            "max_concurrent_files": MAX_CONCURRENT_FILES,
            "successful": successful,
            "low_confidence": low_confidence,
            "no_function_call": no_function,
            "errors": errors,
            "revised_solutions": revised,
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_with_reflection",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "average_confidence": avg_confidence,
            "success_rate": successful / len(json_files) if json_files else 0,
            "total_processing_time": processing_time,
            "average_time_per_question": processing_time / len(json_files) if json_files else 0
        },
        "optimization_stats": {
            "reflection_cache_stats": reflection_cache.get_stats(),
            "api_stability": {
                "retry_attempts": API_RETRY_ATTEMPTS,
                "retry_delay": API_RETRY_DELAY,
                "model_version": "gemini-1.5-flash"
            }
        },
        "results": all_results
    }
    
    summary_filename = f"gemini15_processing_summary_{int(time.time())}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"Total files: {len(json_files)}")
    print(f"Total chunks processed: {len(file_chunks)}")
    print(f"Successful (high confidence): {successful} ({successful/len(json_files)*100:.1f}%)")
    print(f"Low confidence: {low_confidence}")
    print(f"No function call: {no_function}")
    print(f"Errors: {errors}")
    print(f"Revised solutions: {revised}")
    print(f"Average confidence: {avg_confidence:.1f}/10")
    
    print(f"\nPerformance:")
    print(f"Total processing time: {processing_time/60:.1f} minutes")
    print(f"Average time per question: {processing_time/len(json_files):.1f}s")
    
    cache_stats = reflection_cache.get_stats()
    print(f"\nReflection cache stats:")
    print(f"  Hit rate: {cache_stats['hit_rate']}")
    print(f"  Total requests: {cache_stats['total_requests']}")
    print(f"  Cache efficiency: {cache_stats['size']}/{cache_stats['max_size']}")
    
    print(f"\nSummary saved to: {summary_filename}")


if __name__ == "__main__":
    main()
