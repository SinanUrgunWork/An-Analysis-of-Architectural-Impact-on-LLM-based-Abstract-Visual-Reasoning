import os
from openai import OpenAI
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from tool import tool_functions
import re


INPUTS_FOLDER = "inputs"  
DELAY_BETWEEN_FILES = 0.5  


load_dotenv()
MODEL_NAME = "gpt-4.1-mini"  


def get_json_files_from_inputs():
    """Get all questionText*.json files from inputs folder, sorted by number"""
    if not os.path.exists(INPUTS_FOLDER):
        print(f"Error: {INPUTS_FOLDER} folder not found!")
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


def process_single_file(json_file_path, file_index, total_files):
    """Process a single JSON file"""
    filename = os.path.basename(json_file_path)
    
    try:
        # Load the puzzle data
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_json_data = f.read()
        
        number_match = re.search(r'\d+', filename)
        file_number = number_match.group() if number_match else filename
        
        # Create prompts
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
- Layout configurations ("singleton_center", "left_right", "up_down", "out_in", "distribute_three", "distribute_four", "grid_2x2", "distribute_nine", "grid_3x3")
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

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print(f"[{file_index}/{total_files}] Processing: {filename}")
        
        # Make API call
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tool_functions,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4000
        )
        
        message = response.choices[0].message
        
        # Extract CoT reasoning
        cot_reasoning = message.content or ""
        if not cot_reasoning:
            cot_reasoning = "No reasoning content in response"
        
        print(f"[{file_index}/{total_files}] CoT length: {len(cot_reasoning)} chars")
        
        # Save CoT file
        cot_filename = f"simple_cot_reasoning_question{file_number}.json"
        
        cot_data = {
            "puzzle_file": json_file_path,
            "model_used": MODEL_NAME,
            "processing_type": "openai_sequential_api",
            "solving_session": {
                "raw_llm_response": cot_reasoning,
                "cot_reasoning": cot_reasoning,
                "reasoning_length": len(cot_reasoning),
                "function_called": None,
                "function_arguments": None,
                "success": False,
                "extraction_method": "sequential_api_content"
            },
            "reasoning_analysis": {
                "has_step1_patterns": any(keyword in cot_reasoning.lower() for keyword in ["examine", "structure", "step 1"]),
                "has_step2_rule": any(keyword in cot_reasoning.lower() for keyword in ["identify", "pattern", "step 2"]),
                "has_step3_apply": any(keyword in cot_reasoning.lower() for keyword in ["detect", "transformation", "step 3"]),
                "has_step4_generate": any(keyword in cot_reasoning.lower() for keyword in ["apply", "rule", "step 4"]),
                "has_step5_solution": any(keyword in cot_reasoning.lower() for keyword in ["generate", "solution", "step 5"]),
                "mentions_rows_columns": "row" in cot_reasoning.lower() and "column" in cot_reasoning.lower(),
                "mentions_shapes": any(shape in cot_reasoning.lower() for shape in ["triangle", "square", "pentagon", "hexagon", "circle"]),
                "mentions_colors": any(color in cot_reasoning.lower() for color in ["black", "gray", "white", "color"]),
                "mentions_positions": "position" in cot_reasoning.lower() or "grid" in cot_reasoning.lower(),
                "cot_captured_successfully": len(cot_reasoning) > 50,
                "sequential_api_processing": True
            }
        }
        
        if message.tool_calls:
            print(f"[{file_index}/{total_files}] Processing {len(message.tool_calls)} function calls...")
            
            for call in message.tool_calls:
                function_name = call.function.name
                arguments = json.loads(call.function.arguments)
                
                # Add source filename for tool execution
                arguments["source_filename"] = json_file_path
                
                cot_data["solving_session"]["function_called"] = function_name
                cot_data["solving_session"]["function_arguments"] = arguments
                cot_data["solving_session"]["success"] = True
                
                print(f"[{file_index}/{total_files}] Function call captured: {function_name}")
                print(f"[{file_index}/{total_files}] Arguments saved")
                break
        else:
            print(f"[{file_index}/{total_files}] No function calls found")
        
        # Save CoT data
        with open(cot_filename, "w", encoding="utf-8") as f:
            json.dump(cot_data, f, indent=2, ensure_ascii=False)
        
        print(f"[{file_index}/{total_files}] CoT saved to: {cot_filename}")
        
        return {
            "filename": filename,
            "status": "success" if message.tool_calls else "no_function_call",
            "cot_length": len(cot_reasoning),
            "has_function_call": bool(message.tool_calls),
            "processing_time": 0  # We'll add this if needed
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"[{file_index}/{total_files}] ERROR: {filename}: {error_msg}")
        
        return {
            "filename": filename,
            "status": "error",
            "cot_length": 0,
            "has_function_call": False,
            "error_message": error_msg
        }


def process_all_files_sequential():
    """Process all files sequentially"""
    json_files = get_json_files_from_inputs()
    
    if not json_files:
        print("No geminiAnswer*.json files found!")
        return
    
    total_files = len(json_files)
    print(f"Found {total_files} files to process")
    print(f"Processing sequentially with {DELAY_BETWEEN_FILES}s delay between files")
    
    start_time = time.time()
    results = []
    
    for i, json_file_path in enumerate(json_files, 1):
        # Process file
        result = process_single_file(json_file_path, i, total_files)
        results.append(result)
        
        # Show progress every 10 files
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            successful_so_far = len([r for r in results if r["status"] == "success"])
            avg_time_per_file = elapsed_time / i
            estimated_remaining = (total_files - i) * avg_time_per_file
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {i}/{total_files} files ({i/total_files*100:.1f}%)")
            print(f"Successful: {successful_so_far}")
            print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
            print(f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
            print(f"Average time per file: {avg_time_per_file:.1f}s")
            print("-" * 25)
        
        # Brief delay between files
        if i < total_files:
            time.sleep(DELAY_BETWEEN_FILES)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate final statistics
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] != "success"])
    
    # Save final summary
    summary_data = {
        "processing_session": {
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "model_used": MODEL_NAME,
            "processing_type": "openai_sequential_api",
            "delay_between_files": DELAY_BETWEEN_FILES,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "average_time_per_file": total_time / total_files if total_files else 0,
            "success_rate": successful / total_files if total_files else 0
        },
        "results": results
    }
    
    summary_filename = f"sequential_processing_summary_{int(time.time())}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful} ({successful/total_files*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_files*100:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes, {total_time/3600:.1f} hours)")
    print(f"Average time per file: {summary_data['processing_session']['average_time_per_file']:.2f}s")
    print(f"Summary saved to: {summary_filename}")
    
    # CoT analysis summary
    cot_files_found = 0
    cot_high_quality = 0
    for i in range(1, total_files + 1):
        cot_file = f"simple_cot_reasoning_question{i}.json"
        if os.path.exists(cot_file):
            cot_files_found += 1
            try:
                with open(cot_file, "r", encoding="utf-8") as f:
                    cot_data = json.load(f)
                    reasoning_length = cot_data.get("solving_session", {}).get("reasoning_length", 0)
                    if reasoning_length > 200:  # High quality threshold
                        cot_high_quality += 1
            except:
                pass
    
    print(f"\nCoT Analysis:")
    print(f"CoT files created: {cot_files_found}/{total_files}")
    print(f"High-quality reasoning (>200 chars): {cot_high_quality}/{cot_files_found if cot_files_found > 0 else 1}")
    
    return results


def main():
    """OpenAI Sequential API implementation for RAVEN solver"""
    
    print(f"Model: {MODEL_NAME}")
    print("Processing files one by one (no batch API)")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in .env file")
        return
    
    # Process all files sequentially
    results = process_all_files_sequential()
    
    if results:
        successful_count = len([r for r in results if r["status"] == "success"])
        print(f"\nFinal Results: {successful_count}/{len(results)} files processed successfully")


if __name__ == "__main__":
    main()
