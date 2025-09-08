import os
import google.generativeai as genai
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import re
import traceback
from google.protobuf.json_format import MessageToDict
from tool import execute_tool_function
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Dict, List, Any, Optional
import logging
import random


INPUTS_FOLDER = "inputs"
MAX_CONCURRENT_FILES = 12  
API_DELAY = 0.2  
MAX_RETRIES = 2  
RETRY_DELAY = 1.0  
REQUEST_TIMEOUT = 60  


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('processing.log', encoding='utf-8')  # File output
    ]
)
logger = logging.getLogger(__name__)


load_dotenv()
MODEL_NAME = "gemini-1.5-flash"

class EnhancedProgressCounter:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.retried = 0
        self.failed_permanently = 0
        self.lock = threading.Lock()
        self.detailed_results = {}
        self.start_time = time.time()  
    
    def update(self, filename: str, success: bool = False, retry_count: int = 0, error_msg: str = None):
        with self.lock:
            self.completed += 1
            if success:
                self.successful += 1
            if retry_count > 0:
                self.retried += 1
            if error_msg and retry_count >= MAX_RETRIES:
                self.failed_permanently += 1
                
            self.detailed_results[filename] = {
                'success': success,
                'retry_count': retry_count,
                'error': error_msg
            }
            
            if self.completed % 5 == 0:  
                success_rate = (self.successful / self.completed * 100) if self.completed > 0 else 0
                progress_msg = f"Progress: {self.completed}/{self.total} ({self.completed/self.total*100:.1f}%) - Success: {self.successful} ({success_rate:.1f}%), Retried: {self.retried}, Failed: {self.failed_permanently}"
                print(progress_msg)
                logger.info(progress_msg)
                
                # Show current processing speed
                elapsed = time.time() - self.start_time
                files_per_minute = (self.completed / elapsed) * 60 if elapsed > 0 else 0
                logger.info(f"Speed: {files_per_minute:.1f} files/minute")

def robust_extract_function_args(function_call) -> Optional[Dict[str, Any]]:
    extraction_methods = [
        _extract_via_message_to_dict,
        _extract_via_direct_access,
        _extract_via_manual_parsing,
        _extract_via_string_parsing
    ]
    
    for method in extraction_methods:
        try:
            result = method(function_call)
            if result and 'config_type' in result and 'objects' in result:
                logger.debug(f"Successfully extracted args using {method.__name__}")
                return result
        except Exception as e:
            logger.debug(f"{method.__name__} failed: {e}")
            continue
    
    logger.warning("All extraction methods failed")
    return None

def _extract_via_message_to_dict(function_call) -> Dict[str, Any]:
    """Method 1: MessageToDict extraction"""
    if hasattr(function_call, '_pb'):
        call_dict = MessageToDict(function_call._pb)
        return call_dict.get('args', {})
    return {}

def _extract_via_direct_access(function_call) -> Dict[str, Any]:
    """Method 2: Direct field access"""
    arguments = {}
    
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
                for field in ['shape', 'size', 'color', 'angle']:
                    if hasattr(obj, field):
                        value = getattr(obj, field)
                        if field == 'size':
                            obj_dict[field] = float(value)
                        elif field in ['color', 'angle']:
                            obj_dict[field] = int(value)
                        else:
                            obj_dict[field] = str(value)
                if len(obj_dict) == 4:  # All required fields present
                    objects.append(obj_dict)
            arguments['objects'] = objects
        
        # Extract optional fields
        for field in ['grid_layout', 'positions', 'empty_positions']:
            if hasattr(args, field):
                value = getattr(args, field)
                if field == 'grid_layout':
                    arguments[field] = str(value)
                else:
                    arguments[field] = list(value)
    
    return arguments

def _extract_via_manual_parsing(function_call) -> Dict[str, Any]:
    """Method 3: Manual field iteration"""
    arguments = {}
    
    # Common protobuf method names to skip
    SKIP_FIELDS = ['DESCRIPTOR', 'ByteSize', 'Clear', 'ClearField', 'CopyFrom', 'FindInitializationErrors',
                   'FromString', 'HasField', 'IsInitialized', 'ListFields', 'MergeFrom', 'MergeFromString',
                   'ParseFromString', 'SerializeToString', 'SetInParent', 'WhichOneof']
    
    if hasattr(function_call, 'args'):
        args = function_call.args
        for field_name in dir(args):
            if not field_name.startswith('_') and field_name not in SKIP_FIELDS:
                try:
                    value = getattr(args, field_name)
                    if hasattr(value, '_pb'):
                        arguments[field_name] = MessageToDict(value._pb)
                    elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        arguments[field_name] = list(value)
                    else:
                        arguments[field_name] = value
                except:
                    continue
    
    return arguments

def _extract_via_string_parsing(function_call) -> Dict[str, Any]:
    """Method 4: String representation parsing"""
    try:
        func_str = str(function_call)
        arguments = {}
        
        # config_type extraction
        config_match = re.search(r'config_type["\']?\s*:\s*["\']([^"\']+)["\']', func_str)
        if config_match:
            arguments['config_type'] = config_match.group(1)
        
        
        return arguments
    except:
        return {}

def validate_and_truncate_json(raw_json_data: str, max_chars: int = 15000) -> str:
    """Validate JSON data and truncate if too large to prevent API errors"""
    if len(raw_json_data) > max_chars:
        logger.warning(f"JSON data too large ({len(raw_json_data)} chars), truncating to {max_chars}")
        # Try to keep complete panels
        truncated = raw_json_data[:max_chars]
        # Find last complete panel
        last_panel_idx = truncated.rfind('"3_')
        if last_panel_idx > max_chars // 2:  # If we have at least half the data
            truncated = truncated[:last_panel_idx]
        truncated += "\n... (truncated for API limits)"
        return truncated
    return raw_json_data

def call_gemini_api_with_retry(model, prompt, tools, max_attempts=MAX_RETRIES):
    """API call with error handling"""
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            # Check prompt size
            if len(prompt) > 25000:  # Gemini has token limits
                logger.warning(f"Prompt very large ({len(prompt)} chars), may cause API errors")
            
            response = model.generate_content(
                prompt,
                tools=tools,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                    candidate_count=1,  # Ensure single response
                ),
                # request_options={'timeout': REQUEST_TIMEOUT}  # Add timeout if supported
            )
            
            # Enhanced response validation
            if not response:
                raise ValueError("No response object from Gemini API")
            
            if not hasattr(response, 'candidates') or not response.candidates:
                raise ValueError("No candidates in response")
                
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not candidate.content:
                raise ValueError("No content in response candidate")
                
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                raise ValueError("No parts in response content")
            
            # Check for blocked or filtered responses
            if hasattr(candidate, 'finish_reason'):
                if candidate.finish_reason == 3:  # SAFETY
                    raise ValueError("Response blocked due to safety filters")
                elif candidate.finish_reason == 4:  # RECITATION  
                    raise ValueError("Response blocked due to recitation filters")
            
            logger.debug(f"API call successful on attempt {attempt + 1}")
            return response
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Simplified retry strategies 
            if "empty response" in error_msg.lower() or "no content" in error_msg.lower():
                wait_time = RETRY_DELAY + random.uniform(0, 0.5)
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                wait_time = RETRY_DELAY * 2 + random.uniform(0, 1)  # Only 2x for rate limits
            else:
                wait_time = RETRY_DELAY + random.uniform(0, 0.3)
            
            if attempt < max_attempts - 1:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_attempts}): {error_msg}. Retrying in {wait_time:.2f}s")
                time.sleep(wait_time)
                continue
    
    logger.error(f"API call failed permanently after {max_attempts} attempts. Last error: {last_error}")
    raise last_error

def process_single_file_enhanced(json_file_path: str, progress_counter: EnhancedProgressCounter, thread_id: int) -> Dict[str, Any]:
    """file processing with error handling and retry logic"""
    filename = os.path.basename(json_file_path)
    retry_count = 0
    
    while retry_count <= MAX_RETRIES:
        try:
            # Read file with validation
            if not os.path.exists(json_file_path):
                raise FileNotFoundError(f"File not found: {json_file_path}")
                
            with open(json_file_path, "r", encoding="utf-8") as f:
                puzzle_data = json.load(f)
            
            # Validate JSON structure
            if not puzzle_data:
                raise ValueError("Empty JSON data")

            # Validate and truncate JSON data to prevent API errors
            raw_json_data = json.dumps(puzzle_data, indent=2, ensure_ascii=False)
            raw_json_data = validate_and_truncate_json(raw_json_data)
            
            number_match = re.search(r'(\d+)', filename)
            file_number = number_match.group(1) if number_match else filename

            #prompt with function call emphasis
            user_prompt = f"""MANDATORY: You MUST solve this Raven's matrix AND call generate_visual_panel function.

{raw_json_data}

CRITICAL INSTRUCTIONS:
- You MUST analyze the pattern step-by-step
- You MUST call generate_visual_panel function with the solution
- DO NOT just provide text analysis - FUNCTION CALL IS MANDATORY

Analysis Framework:
1. EXAMINE STRUCTURE: Look at each panel's config_type and object count
2. IDENTIFY PATTERNS: Find systematic changes across rows and columns  
3. DETECT TRANSFORMATIONS: Shape changes, color progression, size scaling, rotation
4. APPLY RULE: Determine what panel 3_3 should be
5. GENERATE SOLUTION: Call generate_visual_panel with exact parameters

MANDATORY FUNCTION CALL PARAMETERS:
- config_type: Must be one of: "singleton_center", "left_right", "up_down", "out_in", "distribute_three", "distribute_four", "grid_2x2", "distribute_nine", "grid_3x3"
- objects: Array with shape, size, color, angle for each object

Available shapes: "triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"
Colors: 0-9 (0=white, 9=black)
Sizes: 0.1-1.0 
Angles: -180 to 180 

EXAMPLE FUNCTION CALL FORMAT (adapt with your solution):
generate_visual_panel(config_type="grid_3x3", objects=[{{"shape": "triangle", "size": 0.5, "color": 5, "angle": 0}}, ...])

FAILURE TO CALL THE FUNCTION = INCOMPLETE RESPONSE"""

            system_prompt = """You are an expert Raven's Progressive Matrices solver. Your task has TWO MANDATORY PARTS:

1. ANALYZE the 3x3 matrix pattern step by step
2. CALL the generate_visual_panel function with your solution

CRITICAL: Every response MUST end with a generate_visual_panel function call. Text analysis alone is insufficient.

The generate_visual_panel tool specifications:
- config_type: Layout configuration (9 options available)
- objects: Array of shape objects with 4 required properties each
- For grid layouts: provide exact number of objects needed
- Use "none" shape for empty positions in grids

MANDATORY WORKFLOW:
1. Study the given 8 panels (1_1 through 3_2)
2. Identify the pattern rule
3. Determine what panel 3_3 should contain
4. Call generate_visual_panel function with exact parameters

INCOMPLETE RESPONSE = No function call = FAILURE"""

            # Setup model per thread
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(MODEL_NAME)
            gemini_tools = get_gemini_tools()

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Simplified, faster rate limiting
            base_delay = API_DELAY + (thread_id * 0.1) + random.uniform(0, 0.2)
            time.sleep(base_delay)
            
            logger.debug(f"Thread {thread_id} processing {filename} (attempt {retry_count + 1}) - delay: {base_delay:.2f}s")

            # API call with retry
            response = call_gemini_api_with_retry(model, full_prompt, gemini_tools, MAX_RETRIES)

            # Enhanced response processing with aggressive function call detection
            cot_reasoning = ""
            function_calls = []
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        cot_reasoning += part.text
                    elif hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)

            # Aggressive validation for function calls
            if not cot_reasoning:
                cot_reasoning = "No reasoning content in response"
                logger.warning(f"Empty reasoning for {filename}")
                
            if not function_calls:
                logger.warning(f"No function calls in response for {filename}")
                # Check if reasoning mentions function call but actual call is missing
                reasoning_lower = cot_reasoning.lower()
                if any(keyword in reasoning_lower for keyword in ["generate_visual_panel", "function", "call"]):
                    logger.warning(f"Reasoning mentions function call but no actual call found for {filename}")
                    
                # If this is not the first attempt consider it a failure
                if retry_count > 0:
                    raise ValueError("Model failed to generate function call after retry")

            # Enhanced CoT data structure with more detailed metrics
            cot_data = {
                "puzzle_file": json_file_path,
                "model_used": MODEL_NAME,
                "processing_type": "gemini_parallel_enhanced_v2",
                "thread_id": thread_id,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat(),
                "json_data_size": len(raw_json_data),
                "solving_session": {
                    "raw_llm_response": cot_reasoning,
                    "cot_reasoning": cot_reasoning,
                    "reasoning_length": len(cot_reasoning),
                    "function_called": None,
                    "function_arguments": None,
                    "success": False,
                    "extraction_method": "enhanced_robust_extraction_v2",
                    "function_calls_count": len(function_calls),
                    "response_quality": "good" if len(cot_reasoning) > 200 and function_calls else "poor"
                },
                "quality_metrics": {
                    "has_structured_analysis": any(keyword in cot_reasoning.lower() 
                                                 for keyword in ["step 1", "step 2", "step 3", "step 4", "step 5", "examine", "identify", "detect", "apply", "generate"]),
                    "mentions_patterns": any(keyword in cot_reasoning.lower() 
                                           for keyword in ["pattern", "rule", "transformation", "systematic", "progression"]),
                    "mentions_matrix_elements": any(element in cot_reasoning.lower() 
                                                  for element in ["row", "column", "panel", "shape", "color", "size", "config_type"]),
                    "mentions_function_call": any(keyword in cot_reasoning.lower()
                                                for keyword in ["generate_visual_panel", "function", "call"]),
                    "reasoning_depth": "high" if len(cot_reasoning) > 500 else "medium" if len(cot_reasoning) > 200 else "low",
                    "has_function_call": len(function_calls) > 0,
                    "comprehensive_analysis": len(cot_reasoning) > 300 and len(function_calls) > 0
                }
            }

            success = False
            if function_calls:
                for call in function_calls:
                    function_name = call.name
                    
                    # Enhanced argument extraction
                    arguments = robust_extract_function_args(call)
                    
                    if arguments:
                        arguments["source_filename"] = json_file_path
                        
                        # Execute tool function
                        tool_result = execute_tool_function(function_name, arguments, source_filename=json_file_path)
                        
                        cot_data["solving_session"]["function_called"] = function_name
                        cot_data["solving_session"]["function_arguments"] = arguments
                        cot_data["solving_session"]["success"] = True
                        cot_data["solving_session"]["tool_result"] = tool_result
                        success = True
                        break
                    else:
                        logger.warning(f"Failed to extract arguments for {filename}")

            # Write result file with error handling
            cot_filename = f"enhanced_cot_reasoning_question{file_number}.json"
            try:
                with open(cot_filename, "w", encoding="utf-8") as f:
                    json.dump(cot_data, f, indent=2, ensure_ascii=False)
            except Exception as write_error:
                logger.error(f"Failed to write CoT file for {filename}: {write_error}")

            progress_counter.update(filename, success, retry_count)

            return {
                "filename": filename,
                "status": "success" if success else "no_function_call",
                "cot_length": len(cot_reasoning),
                "has_function_call": bool(function_calls),
                "function_calls_count": len(function_calls),
                "retry_count": retry_count,
                "thread_id": thread_id,
                "quality_score": _calculate_quality_score(cot_data)
            }

        except Exception as e:
            retry_count += 1
            error_msg = f"{str(e)} (Attempt {retry_count}/{MAX_RETRIES + 1})"
            
            if retry_count <= MAX_RETRIES:
                logger.warning(f"Retrying {filename} due to error: {error_msg}")
                time.sleep(RETRY_DELAY)  # Simple fixed delay
                continue
            else:
                logger.error(f"PERMANENT FAILURE for {filename}: {error_msg}")
                progress_counter.update(filename, False, retry_count, error_msg)
                
                return {
                    "filename": filename,
                    "status": "permanent_failure",
                    "cot_length": 0,
                    "has_function_call": False,
                    "error_message": error_msg,
                    "retry_count": retry_count,
                    "thread_id": thread_id
                }

def _calculate_quality_score(cot_data: Dict[str, Any]) -> float:
    """Calculate quality score for the processing result"""
    metrics = cot_data.get("quality_metrics", {})
    score = 0.0
    
    if metrics.get("has_structured_analysis", False):
        score += 0.3
    if metrics.get("mentions_patterns", False):
        score += 0.2
    if metrics.get("mentions_matrix_elements", False):
        score += 0.2
    if metrics.get("has_function_call", False):
        score += 0.3
    
    # Reasoning depth bonus
    depth = metrics.get("reasoning_depth", "low")
    if depth == "high":
        score += 0.1
    elif depth == "medium":
        score += 0.05
    
    return min(score, 1.0)

def get_gemini_tools():
    """Define tools in Gemini's expected format directly"""
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

def process_all_files_enhanced():
    """Enhanced parallel processing with comprehensive error handling"""
    json_files = get_json_files_from_inputs()

    if not json_files:
        print("No input JSON files found!")
        logger.error("No input JSON files found!")
        return []

    total_files = len(json_files)
    print(f"Starting enhanced processing of {total_files} files")
    print(f"Configuration: max_workers={MAX_CONCURRENT_FILES}, api_delay={API_DELAY}s, max_retries={MAX_RETRIES}")
    logger.info(f"Starting enhanced processing of {total_files} files")

    start_time = time.time()
    progress_counter = EnhancedProgressCounter(total_files)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as executor:
        future_to_file = {
            executor.submit(process_single_file_enhanced, json_file_path, progress_counter, i): json_file_path 
            for i, json_file_path in enumerate(json_files)
        }
        
        for future in as_completed(future_to_file):
            try:
                result = future.result(timeout=300)  # 5 minute timeout per file
                results.append(result)
            except Exception as e:
                json_file_path = future_to_file[future]
                filename = os.path.basename(json_file_path)
                logger.error(f"Executor exception for {filename}: {e}")
                results.append({
                    "filename": filename,
                    "status": "executor_exception",
                    "error_message": str(e)
                })

    end_time = time.time()
    total_time = end_time - start_time

    # statistics
    successful = len([r for r in results if r.get("status") == "success"])
    partial_success = len([r for r in results if r.get("status") == "no_function_call"])
    failed = total_files - successful - partial_success
    
    avg_quality = sum(r.get("quality_score", 0) for r in results) / len(results) if results else 0

    # Detailed summary
    summary_data = {
        "processing_session": {
            "total_files": total_files,
            "successful": successful,
            "partial_success": partial_success,  
            "failed": failed,
            "success_rate": successful / total_files,
            "completion_rate": (successful + partial_success) / total_files,
            "average_quality_score": avg_quality,
            "model_used": MODEL_NAME,
            "processing_type": "gemini_parallel_enhanced_v2",
            "configuration": {
                "max_concurrent_files": MAX_CONCURRENT_FILES,
                "api_delay": API_DELAY,
                "max_retries": MAX_RETRIES,
                "retry_delay": RETRY_DELAY
            },
            "timing": {
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "average_time_per_file": total_time / total_files,
                "estimated_speedup": "3-5x vs sequential with reliability"
            }
        },
        "detailed_results": results,
        "quality_analysis": {
            "high_quality_results": len([r for r in results if r.get("quality_score", 0) > 0.8]),
            "medium_quality_results": len([r for r in results if 0.5 <= r.get("quality_score", 0) <= 0.8]),
            "low_quality_results": len([r for r in results if r.get("quality_score", 0) < 0.5])
        }
    }

    timestamp = int(time.time())
    summary_filename = f"enhanced_processing_summary_{timestamp}.json"
    
    try:
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to: {summary_filename}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")

    # Final report
    print(f"\n{'='*80}")
    print(f"Total files: {total_files}")
    print(f"Fully successful: {successful} ({successful/total_files*100:.1f}%)")
    print(f"Partial success: {partial_success} ({partial_success/total_files*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_files*100:.1f}%)")
    print(f"Average quality score: {avg_quality:.3f}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average per file: {total_time/total_files:.2f}s")
    
    logger.info("Enhanced processing completed successfully")
    logger.info(f"Results: {successful}/{total_files} successful, quality: {avg_quality:.3f}")

    return results

def get_json_files_from_inputs():
    """Get JSON files from inputs folder with validation"""
    if not os.path.exists(INPUTS_FOLDER):
        print(f"Error: Inputs folder not found: {INPUTS_FOLDER}")
        logger.error(f"Inputs folder not found: {INPUTS_FOLDER}")
        return []

    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            if os.path.getsize(filepath) > 0:  # Check file is not empty
                json_files.append(filepath)
            else:
                print(f"Warning: Empty file skipped: {filename}")
                logger.warning(f"Empty file skipped: {filename}")

    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0

    json_files.sort(key=extract_number)
    print(f"Found {len(json_files)} valid JSON files")
    logger.info(f"Found {len(json_files)} valid JSON files")
    return json_files

def main():
    """Enhanced main function with comprehensive setup"""
    print(f"Model: {MODEL_NAME}")
    print(f"Max Concurrent Files: {MAX_CONCURRENT_FILES}")
    print(f"API Delay: {API_DELAY}s")
    print(f"Max Retries: {MAX_RETRIES}")
    
    # Environment validation
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("Missing GEMINI_API_KEY in environment")
        print("Please set GEMINI_API_KEY in your .env file")
        return

    if not os.path.exists(INPUTS_FOLDER):
        logger.error(f"Inputs folder not found: {INPUTS_FOLDER}")
        print(f"Please create the '{INPUTS_FOLDER}' folder and add your JSON files")
        return

    try:
        # Process files
        results = process_all_files_enhanced()
        
        if results:
            successful_count = len([r for r in results if r.get("status") == "success"])
            logger.info(f"Final result: {successful_count}/{len(results)} files processed successfully")
            print(f"Processing complete: {successful_count}/{len(results)} files successful")
        else:
            logger.error("No files were processed")
            print("No files were processed - check your input files")
            
    except Exception as e:
        logger.error(f"Main processing error: {e}")
        print(f"Processing failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
