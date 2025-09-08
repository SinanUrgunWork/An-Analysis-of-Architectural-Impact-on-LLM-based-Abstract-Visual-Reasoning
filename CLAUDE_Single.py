import os
import anthropic
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from tool import tool_functions
import re


INPUTS_FOLDER = "inputs"  # Folder containing JSON files
BATCH_CHECK_INTERVAL = 5  # Check progress every 5 seconds


load_dotenv()
MODEL_NAME = "claude-3-5-haiku-20241022"


def convert_tools_to_claude_format():
    """Convert OpenAI tool format to Claude tool format"""
    claude_tools = []
    
    for tool in tool_functions:
        if tool["type"] == "function":
            func = tool["function"]
            
            # Convert to Claude tool format
            claude_tool = {
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"]
            }
            
            claude_tools.append(claude_tool)
    
    return claude_tools


def get_json_files_from_inputs():
    if not os.path.exists(INPUTS_FOLDER):
        print(f"Error: {INPUTS_FOLDER} folder not found!")
        return []
    
    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            json_files.append(filepath)
    
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    json_files.sort(key=extract_number)
    return json_files


def create_batch_jsonl_file(json_files):
    """Create JSONL file for batch processing """
    
    batch_requests = []
    
    for json_file_path in json_files:
        try:
            # Load the puzzle data
            with open(json_file_path, "r", encoding="utf-8") as f:
                raw_json_data = f.read()
            
            filename = os.path.basename(json_file_path)
            number_match = re.search(r'\d+', filename)
            file_number = number_match.group() if number_match else filename
            
            # Create individual request for each file
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
- Shape types (7 different shapes)
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

            # Create batch request structure 
            batch_request = {
                "custom_id": f"puzzle_{file_number}",
                "method": "POST",
                "url": "/v1/messages",
                "body": {
                    "model": MODEL_NAME,
                    "system": system_prompt,
                    "messages": [
                        {"role": "user", "content": user_prompt}
                    ],
                    "tools": convert_tools_to_claude_format(),
                    "tool_choice": {"type": "auto"},
                    "max_tokens": 4000
                }
            }
            
            batch_requests.append({
                "request": batch_request,
                "file_info": {
                    "filepath": json_file_path,
                    "filename": filename,
                    "file_number": file_number
                }
            })
            
        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")
            continue
    
    # Write JSONL file for batch processing
    batch_filename = f"batch_requests.jsonl"
    
    with open(batch_filename, "w", encoding="utf-8") as f:
        for item in batch_requests:
            f.write(json.dumps(item["request"]) + "\n")
    
    print(f"Created batch file: {batch_filename}")
    print(f"Total requests in batch: {len(batch_requests)}")
    
    return batch_filename, batch_requests


def submit_batch_job(batch_filename, file_mapping):
    """Submit batch job """
    
    print(f"\nPreparing batch processing: {batch_filename}")
    
    try:
        # Simulate batch job creation
        batch_id = f"claude_batch_{int(time.time())}"
        
        print(f"Batch job created successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Status: validating")
        print(f"Created at: {datetime.now().strftime('%H:%M:%S')}")
        
        # Save batch info for tracking
        batch_info = {
            "batch_id": batch_id,
            "status": "validating",
            "created_at": int(time.time()),
            "batch_filename": batch_filename,
            "total_requests": len(file_mapping),
            "file_mapping": {item["request"]["custom_id"]: item["file_info"] for item in file_mapping}
        }
        
        batch_info_filename = f"batch_info_{batch_id}.json"
        with open(batch_info_filename, "w", encoding="utf-8") as f:
            json.dump(batch_info, f, indent=2, ensure_ascii=False)
        
        print(f"Batch info saved to: {batch_info_filename}")
        
        return batch_id, batch_info_filename
        
    except Exception as e:
        print(f"Error submitting batch job: {e}")
        return None, None


def monitor_batch_job(batch_id, batch_info_filename, file_mapping):
    """Monitor batch job progress """
    
    print(f"\nProcessing batch job: {batch_id}")
    print(f"Processing files sequentially...")
    print(f"Note: Claude API processes requests individually")
    
    # Initialize Anthropic client
    try:
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")
        return None
    
    start_time = time.time()
    total_requests = len(file_mapping)
    completed_requests = 0
    failed_requests = 0
    results = []
    
    print(f"\n--- Starting Sequential Processing ---")
    print(f"Total files to process: {total_requests}")
    
    for item in file_mapping:
        request_data = item["request"]
        file_info = item["file_info"]
        
        custom_id = request_data["custom_id"]
        filename = file_info["filename"]
        file_number = file_info["file_number"]
        
        print(f"\nProcessing {filename} ({completed_requests + 1}/{total_requests})...")
        
        try:
            # Extract request details
            system_prompt = request_data["body"]["system"]
            user_message = request_data["body"]["messages"][0]["content"]
            # tools are already in Claude format from the batch request
            
            # Make API call to Claude
            response = client.messages.create(
                model=MODEL_NAME,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=convert_tools_to_claude_format(),
                tool_choice={"type": "auto"},
                max_tokens=4000
            )
            
            # Convert response to OpenAI-like format for compatibility
            claude_response = {
                "custom_id": custom_id,
                "response": {
                    "choices": [{
                        "message": {
                            "content": "",
                            "tool_calls": []
                        }
                    }]
                }
            }
            
            # Extract content and tool calls
            if hasattr(response, 'content') and response.content:
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        claude_response["response"]["choices"][0]["message"]["content"] += content_block.text
                    elif hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        tool_call = {
                            "id": content_block.id,
                            "type": "function",
                            "function": {
                                "name": content_block.name,
                                "arguments": json.dumps(content_block.input)
                            }
                        }
                        claude_response["response"]["choices"][0]["message"]["tool_calls"].append(tool_call)
            
            results.append(claude_response)
            completed_requests += 1
            
            print(f"SUCCESS: {filename} processed")
            
            # Show progress
            elapsed_minutes = (time.time() - start_time) / 60
            print(f"Progress: {completed_requests}/{total_requests} completed ({elapsed_minutes:.1f} minutes elapsed)")
            
            # Brief pause between requests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            
            # Create error response
            error_response = {
                "custom_id": custom_id,
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }
            results.append(error_response)
            failed_requests += 1
            
            # Continue to next file
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n--- Sequential Processing Complete ---")
    print(f"Total processed: {completed_requests + failed_requests}")
    print(f"Successful: {completed_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    class MockBatch:
        def __init__(self, batch_id, results):
            self.id = batch_id
            self.status = "completed"
            self.output_file_id = f"results_{batch_id}"
            self.request_counts = {
                "total": completed_requests + failed_requests,
                "completed": completed_requests,
                "failed": failed_requests
            }
            self.results = results
    
    # Save results to file 
    results_filename = f"batch_results_{batch_id}.jsonl"
    with open(results_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    return MockBatch(batch_id, results)


def download_and_process_results(batch, batch_info_filename):
    """Process batch results """
    
    print(f"\nProcessing batch results...")
    
    try:
        # Load batch info
        with open(batch_info_filename, "r", encoding="utf-8") as f:
            batch_info = json.load(f)
        
        file_mapping = batch_info["file_mapping"]
        
        # Results are already available from the batch object
        results_filename = f"batch_results_{batch.id}.jsonl"
        
        print(f"Results saved to: {results_filename}")
        
        # Process each result
        results = []
        successful_results = 0
        failed_results = 0
        
        with open(results_filename, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    custom_id = result["custom_id"]
                    
                    if custom_id not in file_mapping:
                        print(f"Warning: Unknown custom_id {custom_id}")
                        continue
                    
                    file_info = file_mapping[custom_id]
                    file_number = file_info["file_number"]
                    filename = file_info["filename"]
                    filepath = file_info["filepath"]
                    
                    print(f"\nProcessing result for {filename}...")
                    
                    if result.get("error"):
                        print(f"API Error for {filename}: {result['error']}")
                        failed_results += 1
                        continue
                    
                    response = result["response"]
                    message = response["choices"][0]["message"]
                    
                    # Extract CoT reasoning
                    cot_reasoning = message.get("content", "")
                    if not cot_reasoning:
                        cot_reasoning = "No reasoning content in batch response"
                    
                    print(f"CoT reasoning length: {len(cot_reasoning)} characters")
                    
                    # Save CoT file
                    cot_filename = f"simple_cot_reasoning_question{file_number}.json"
                    
                    cot_data = {
                        "puzzle_file": filepath,
                        "model_used": MODEL_NAME,
                        "processing_type": "claude_batch_api",
                        "batch_info": {
                            "batch_id": batch.id,
                            "custom_id": custom_id
                        },
                        "solving_session": {
                            "raw_llm_response": cot_reasoning,
                            "cot_reasoning": cot_reasoning,
                            "reasoning_length": len(cot_reasoning),
                            "function_called": None,
                            "function_arguments": None,
                            "success": False,
                            "extraction_method": "claude_api_content"
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
                            "claude_api_processing": True
                        }
                    }
                    
                    # Process function calls if present
                    if message.get("tool_calls"):
                        print(f"Processing {len(message['tool_calls'])} function calls...")
                        
                        for call in message["tool_calls"]:
                            function_name = call["function"]["name"]
                            arguments = json.loads(call["function"]["arguments"])
                            
                            # Add source filename for tool execution
                            arguments["source_filename"] = filepath
                            
                            cot_data["solving_session"]["function_called"] = function_name
                            cot_data["solving_session"]["function_arguments"] = arguments
                            cot_data["solving_session"]["success"] = True
                            
                            # Note: Function arguments saved for manual execution
                            print(f"Function call captured: {function_name}")
                            print(f"Arguments saved for manual execution")
                            
                            successful_results += 1
                            break
                    else:
                        print(f"No function calls found in response")
                        failed_results += 1
                    
                    # Save CoT data
                    with open(cot_filename, "w", encoding="utf-8") as f:
                        json.dump(cot_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"CoT saved to: {cot_filename}")
                    
                    results.append({
                        "custom_id": custom_id,
                        "filename": filename,
                        "status": "success" if message.get("tool_calls") else "no_function_call",
                        "cot_length": len(cot_reasoning),
                        "has_function_call": bool(message.get("tool_calls"))
                    })
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    failed_results += 1
                    continue
        
        # Final summary
        total_processed = successful_results + failed_results
        
        print(f"\n{'='*80}")
        print("BATCH API PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total results processed: {total_processed}")
        print(f"Successful (with function calls): {successful_results}")
        print(f"Failed or incomplete: {failed_results}")
        print(f"Success rate: {successful_results/total_processed*100:.1f}%" if total_processed > 0 else "N/A")
        print(f"Results saved to: {results_filename}")
        
        # Save processing summary
        summary = {
            "batch_id": batch.id,
            "total_processed": total_processed,
            "successful": successful_results,
            "failed": failed_results,
            "results": results
        }
        
        summary_filename = f"batch_processing_summary_{batch.id}.json"
        with open(summary_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {summary_filename}")
        
        return results
        
    except Exception as e:
        print(f"Error downloading/processing results: {e}")
        return []


def main():
    """Claude 3.5 Haiku implementation for RAVEN solver"""
    
    print(f"Model: {MODEL_NAME}")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[!] Missing ANTHROPIC_API_KEY in .env file")
        return
    
    # Step 1: Prepare batch file
    print("\n=== Step 1: Preparing Batch File ===")
    json_files = get_json_files_from_inputs()
    
    if not json_files:
        print("No questionText*.json files found!")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    batch_filename, file_mapping = create_batch_jsonl_file(json_files)
    if not batch_filename:
        print("Failed to create batch file")
        return
    
    # Step 2: Submit batch job
    print("\n=== Step 2: Submitting Batch Job ===")
    batch_id, batch_info_filename = submit_batch_job(batch_filename, file_mapping)
    
    if not batch_id:
        print("Failed to submit batch job")
        return
    
    # Step 3: Monitor batch progress
    print("\n=== Step 3: Processing Files ===")
    completed_batch = monitor_batch_job(batch_id, batch_info_filename, file_mapping)
    
    if not completed_batch:
        print("Batch job failed or was cancelled")
        return
    
    # Step 4: Process results
    print("\n=== Step 4: Processing Results ===")
    results = download_and_process_results(completed_batch, batch_info_filename)
    
    print(f"\nBatch API processing completed!")
    
    if results:
        successful_count = len([r for r in results if r["status"] == "success"])
        print(f"Final: {successful_count}/{len(results)} files processed successfully")


if __name__ == "__main__":
    main()
