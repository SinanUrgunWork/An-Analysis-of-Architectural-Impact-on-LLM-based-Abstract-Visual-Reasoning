import os
import json
import time
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from collections import OrderedDict
from tool import tool_functions, execute_tool_function
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity


INPUTS_FOLDER = "inputs"
BATCH_INPUT_FILE_PREFIX = "batch_requests_chunk"
BATCH_OUTPUT_FILE_PREFIX = "batch_results_chunk"
BATCH_ERROR_FILE_PREFIX = "batch_errors_chunk"
CHECK_INTERVAL = 10
MAX_ATTEMPTS = 3
SIMILARITY_THRESHOLD = 0.8  
DELAY_BETWEEN_ATTEMPTS = 0.1


BATCH_CHUNK_SIZE = 150  
MAX_CONCURRENT_BATCHES = 2 


EMBEDDING_MODEL = "text-embedding-3-small"
REFERENCE_EMBEDDINGS_FILE = "reference_solution_embeddings.json"


load_dotenv()
MODEL_NAME = "gpt-4.1-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingCache:
    """Cache for embeddings to avoid recomputing identical solutions"""
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get_key(self, text):
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text):
        """Get embedding from cache if exists"""
        key = self.get_key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            return self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def set(self, text, embedding):
        """Store embedding in cache"""
        key = self.get_key(text)
        
        if key in self.cache:
            self.cache[key] = embedding
            self.cache.move_to_end(key)
            return
        
        if len(self.cache) >= self.max_size:
            removed_key = next(iter(self.cache))
            del self.cache[removed_key]
            self.stats['evictions'] += 1
        
        self.cache[key] = embedding
    
    def get_stats(self):
        """Return cache statistics"""
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


class ReferenceSolutionManager:
    """Manages reference solution embeddings for similarity comparison"""
    def __init__(self, reference_file=REFERENCE_EMBEDDINGS_FILE):
        self.reference_file = reference_file
        self.reference_embeddings = []
        self.reference_metadata = []
        self.load_reference_embeddings()
    
    def load_reference_embeddings(self):
        """Load precomputed reference embeddings"""
        if os.path.exists(self.reference_file):
            try:
                with open(self.reference_file, 'r') as f:
                    data = json.load(f)
                    self.reference_embeddings = [np.array(item['embedding']) for item in data['references']]
                    self.reference_metadata = [item['metadata'] for item in data['references']]
                print(f"Loaded {len(self.reference_embeddings)} reference embeddings")
            except Exception as e:
                print(f"Error loading reference embeddings: {e}")
                self.initialize_default_references()
        else:
            print("No reference embeddings file found creating default references")
            self.initialize_default_references()
    
    def initialize_default_references(self):
        """Initialize with some default high quality solution patterns"""
        default_patterns = [
            {
                "reasoning": "Step 1: Examining structure shows consistent pattern. Step 2: Pattern analysis reveals systematic transformations. Step 3: Row progression follows clear rule. Step 4: Column consistency confirmed. Step 5: Generated solution applies discovered pattern.",
                "function_call": {"config_type": "singleton_center", "objects": [{"shape": "triangle", "size": 0.5, "color": 5, "angle": 0}]},
                "metadata": {"type": "high_quality_systematic", "confidence": 9.0}
            },
            {
                "reasoning": "Systematic analysis: 1) Structure examination shows grid patterns 2) Transformation detection reveals rotation and color changes 3) Position analysis confirms spatial relationships 4) Pattern application generates consistent solution",
                "function_call": {"config_type": "grid_3x3", "objects": [{"shape": "square", "size": 0.4, "color": 3, "angle": 45}]},
                "metadata": {"type": "structured_approach", "confidence": 8.5}
            },
            {
                "reasoning": "Pattern identification across rows and columns shows clear progression. Shape transformations follow logical sequence. Color variations indicate systematic rule. Position changes confirm spatial pattern. Solution maintains consistency.",
                "function_call": {"config_type": "distribute_four", "objects": [{"shape": "circle", "size": 0.6, "color": 2, "angle": 0}]},
                "metadata": {"type": "pattern_based", "confidence": 8.8}
            }
        ]
        
        print("Creating embeddings for default reference patterns")
        for i, pattern in enumerate(default_patterns):
            try:
                combined_text = self.create_solution_text(pattern["reasoning"], pattern["function_call"])
                embedding = get_embedding(combined_text)
                self.reference_embeddings.append(embedding)
                self.reference_metadata.append(pattern["metadata"])
                print(f"Created reference embedding {i+1}/{len(default_patterns)}")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error creating reference embedding {i}: {e}")
        
        # Save to file
        self.save_reference_embeddings()
    
    def create_solution_text(self, reasoning, function_call):
        """Create combined text representation of solution"""
        return f"REASONING: {reasoning}\n\nFUNCTION_CALL: {json.dumps(function_call, sort_keys=True)}"
    
    def save_reference_embeddings(self):
        """Save reference embeddings to file"""
        try:
            data = {
                "model": EMBEDDING_MODEL,
                "created": datetime.now().isoformat(),
                "references": [
                    {
                        "embedding": embedding.tolist(),
                        "metadata": metadata
                    }
                    for embedding, metadata in zip(self.reference_embeddings, self.reference_metadata)
                ]
            }
            with open(self.reference_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.reference_embeddings)} reference embeddings to {self.reference_file}")
        except Exception as e:
            print(f"Error saving reference embeddings: {e}")
    
    def add_successful_solution(self, reasoning, function_call, confidence_score):
        """Add a new successful solution to references"""
        if confidence_score >= 8.0:  
            try:
                combined_text = self.create_solution_text(reasoning, function_call)
                embedding = get_embedding(combined_text)
                self.reference_embeddings.append(embedding)
                self.reference_metadata.append({
                    "type": "successful_solution",
                    "confidence": confidence_score,
                    "added": datetime.now().isoformat()
                })
                
                # Periodically save updates
                if len(self.reference_embeddings) % 10 == 0:
                    self.save_reference_embeddings()
                    
            except Exception as e:
                print(f"Error adding successful solution to references: {e}")
    
    def compute_similarity(self, reasoning, function_call):
        """Compute similarity score against reference solutions"""
        if not self.reference_embeddings:
            print("No reference embeddings available")
            return 0.0
        
        try:
            combined_text = self.create_solution_text(reasoning, function_call)
            current_embedding = get_embedding(combined_text)
            
            # Compute similarities against all references
            similarities = []
            for ref_embedding in self.reference_embeddings:
                similarity = cosine_similarity(
                    current_embedding.reshape(1, -1),
                    ref_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)
            
            # Return maximum similarity
            max_similarity = max(similarities)
            best_ref_idx = similarities.index(max_similarity)
            
            return {
                "max_similarity": max_similarity,
                "best_reference": self.reference_metadata[best_ref_idx] if best_ref_idx < len(self.reference_metadata) else None,
                "all_similarities": similarities
            }
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return {"max_similarity": 0.0, "error": str(e)}


# Global instances
embedding_cache = EmbeddingCache()
reference_manager = ReferenceSolutionManager()


def get_embedding(text):
    """Get embedding for text with caching"""
    # Check cache first
    cached_embedding = embedding_cache.get(text)
    if cached_embedding is not None:
        return cached_embedding
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        embedding_cache.set(text, embedding)
        return embedding
        
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(1536)  # Return zero vector as fallback


def perform_embedding_validation(reasoning, function_call, question_number):
    """Perform embedding-based validation of the reasoning and solution"""
    try:
        # Compute similarity
        similarity_result = reference_manager.compute_similarity(reasoning, function_call)
        max_similarity = similarity_result.get("max_similarity", 0.0)
        
        # Determine recommendation based on similarity threshold
        if max_similarity >= SIMILARITY_THRESHOLD:
            recommendation = "ACCEPT"
            status = "accepted"
        else:
            recommendation = "REJECT"
            status = "low_similarity"
        
        # Convert similarity to confidence-like score (0-10 scale)
        confidence_score = max_similarity * 10
        
        return {
            "status": "success",
            "confidence_score": confidence_score,
            "similarity_score": max_similarity,
            "recommendation": recommendation,
            "validation_status": status,
            "best_reference": similarity_result.get("best_reference"),
            "threshold_used": SIMILARITY_THRESHOLD,
            "embedding_model": EMBEDDING_MODEL
        }
        
    except Exception as e:
        print(f"Error in embedding validation: {e}")
        return {
            "status": "error",
            "confidence_score": 0,
            "similarity_score": 0.0,
            "recommendation": "REJECT",
            "error": str(e)
        }


def generate_new_attempt(question_number, attempt_count):
    """Generate a new attempt for the question"""
    # Load original puzzle data
    json_file_path = os.path.join(INPUTS_FOLDER, f"questionText{question_number}.json")
    if not os.path.exists(json_file_path):
        return None, None
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_json_data = f.read()
    
    temperature = min(0.3 + (attempt_count * 0.1), 0.7)  # Increase temperature for later attempts
    
    user_prompt = f"""Solve this 3x3 Raven's matrix. Find the pattern and generate panel 3_3.

{raw_json_data}

Please follow this ANALYSIS FRAMEWORK step by step and show your reasoning (Attempt {attempt_count+1}):

1. EXAMINE STRUCTURE: Look at each panel's config_type and object count
2. IDENTIFY PATTERNS: Find systematic changes across rows and columns  
3. DETECT TRANSFORMATIONS: Shape changes, color progression, size scaling, rotation
4. APPLY RULE: Determine what panel 3_3 should be
5. GENERATE SOLUTION: Call generate_visual_panel with exact parameters

Think through each step carefully and consider multiple possible patterns before deciding. 
Then call the generate_visual_panel function with:
- config_type: "singleton_center", "left_right", "up_down", "out_in", "distribute_three", "distribute_four", "grid_2x2", "distribute_nine", "grid_3x3"
- objects: array with shape, size, color, angle

Available shapes: "triangle", "square", "pentagon", "hexagon", "heptagon", "circle", "line", "none"
Colors: 0-9 (0=white, 9=black)
Sizes: 0.1-1.0 
Angles: -180 to 180 

For grid_3x3: provide exactly 9 objects, use "none" for empty spots.

Show your detailed step-by-step reasoning, then call the function."""

    system_prompt = """You are an expert Raven's Progressive Matrices solver with access to a powerful visual generation tool.

IMPORTANT: You must provide detailed step-by-step reasoning first, then call the generate_visual_panel function.

The tool can create ANY arrangement of shapes with precise control over:
- Layout configurations 
- Object positions (specific indices in grids)
- Shape types (7 different shapes + line)
- Colors (10 grayscale levels with multiple naming conventions)
- Sizes (continuous scale from 0.1 to 1.0)
- Rotations (any angle from -180 to 180)

Process:
1. Analyze the matrix patterns step by step
2. Show your detailed reasoning for each step
3. Determine the solution with confidence
4. Call generate_visual_panel function with exact parameters

You MUST call the generate_visual_panel function after providing your analysis."""
    
    try:
        time.sleep(DELAY_BETWEEN_ATTEMPTS)
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tool_functions,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=4000
        )
        
        message = response.choices[0].message
        reasoning = message.content or ""
        
        # Extract function call
        function_call = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.function.name == "generate_visual_panel":
                function_call = json.loads(tool_call.function.arguments)
        
        return reasoning, function_call
        
    except Exception as e:
        print(f"Error generating new attempt: {e}")
        return None, None


def get_json_files_from_inputs():
    """Get all geminiAnswer*.json files from inputs folder, sorted by number"""
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


def chunk_files(json_files, chunk_size):
    """Split files into chunks for batch processing"""
    chunks = []
    for i in range(0, len(json_files), chunk_size):
        chunk = json_files[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def create_batch_request_file(json_files, chunk_id):
    """Create JSONL file with batch requests for a specific chunk"""
    requests = []
    batch_input_file = f"{BATCH_INPUT_FILE_PREFIX}_{chunk_id}.jsonl"
    
    for json_file_path in json_files:
        filename = os.path.basename(json_file_path)
        
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_json_data = f.read()
        
        number_match = re.search(r'\d+', filename)
        file_number = number_match.group() if number_match else filename
        
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
        
        request = {
            "custom_id": f"chunk{chunk_id}-request-{file_number}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "tools": tool_functions,
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": 4000
            }
        }
        
        requests.append(request)
    
    # Write JSONL file
    with open(batch_input_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch request file {batch_input_file} with {len(requests)} requests")
    return batch_input_file, len(requests)


def submit_batch(batch_input_file, chunk_id):
    """Submit batch to OpenAI"""
    try:
        batch_input_file_obj = client.files.create(
            file=open(batch_input_file, "rb"),
            purpose="batch"
        )
        
        batch = client.batches.create(
            input_file_id=batch_input_file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"RAVEN matrix solver batch chunk {chunk_id}"
            }
        )
        
        print(f"Batch chunk {chunk_id} submitted with ID: {batch.id}")
        return batch.id
        
    except Exception as e:
        print(f"Error submitting batch chunk {chunk_id}: {e}")
        return None


def wait_for_batch_completion(batch_id, chunk_id):
    """Wait for batch to complete"""
    print(f"Waiting for batch chunk {chunk_id} ({batch_id}) to complete...")
    
    while True:
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
        except Exception as e:
            print(f"Error checking batch {chunk_id} status: {e}")
            return None
        
        print(f"Batch chunk {chunk_id} status: {status}")
        
        if status == "completed":
            print(f"Batch chunk {chunk_id} completed successfully!")
            return batch
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch chunk {chunk_id} {status}!")
            if hasattr(batch, 'errors'):
                print(f"Errors: {batch.errors}")
            return None
        
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            total = counts.total
            completed = counts.completed
            failed = counts.failed
            print(f"Chunk {chunk_id} progress: {completed}/{total} completed, {failed} failed")
        
        time.sleep(CHECK_INTERVAL)


def download_batch_results(batch, chunk_id):
    """Download batch results"""
    try:
        batch_output_file = f"{BATCH_OUTPUT_FILE_PREFIX}_{chunk_id}.jsonl"
        batch_error_file = f"{BATCH_ERROR_FILE_PREFIX}_{chunk_id}.jsonl"
        
        # Download output file
        if batch.output_file_id:
            file_response = client.files.content(batch.output_file_id)
            with open(batch_output_file, 'wb') as f:
                f.write(file_response.content)
            print(f"Downloaded chunk {chunk_id} results to {batch_output_file}")
        
        # Download error file if exists
        if hasattr(batch, 'error_file_id') and batch.error_file_id:
            error_response = client.files.content(batch.error_file_id)
            with open(batch_error_file, 'wb') as f:
                f.write(error_response.content)
            print(f"Downloaded chunk {chunk_id} errors to {batch_error_file}")
        
        return batch_output_file
        
    except Exception as e:
        print(f"Error downloading batch chunk {chunk_id} results: {e}")
        return None


def process_batch_result_with_embedding(result, question_number, max_trials=3):
    """Process a single batch result with embedding based validation"""
    try:
        response = result['response']
        if response['status_code'] != 200:
            return {
                "question_number": question_number,
                "status": "api_error",
                "error": f"API returned status {response['status_code']}"
            }
        
        body = response['body']
        message = body['choices'][0]['message']
        initial_reasoning = message.get('content') or ""
        if not initial_reasoning:
            initial_reasoning = "No reasoning content in response"
        
        print(f"[Question {question_number}] Initial CoT length: {len(initial_reasoning)} chars")
        
        # Extract function call
        initial_function_call = None
        function_called = None
        if 'tool_calls' in message and message['tool_calls']:
            tool_call = message['tool_calls'][0]
            function_called = tool_call['function']['name']
            initial_function_call = json.loads(tool_call['function']['arguments'])
            print(f"[Question {question_number}] Function call captured: {function_called}")
        else:
            print(f"[Question {question_number}] No function calls found")
        
        if not initial_function_call:
            # Save CoT file for no function call case
            cot_filename = f"embedding_cot_reasoning_question{question_number}.json"
            cot_data = {
                "puzzle_file": f"inputs/questionText{question_number}.json",
                "model_used": MODEL_NAME,
                "processing_type": "chunked_batch_api_with_embedding",
                "solving_session": {
                    "raw_llm_response": initial_reasoning,
                    "cot_reasoning": initial_reasoning,
                    "reasoning_length": len(initial_reasoning),
                    "function_called": function_called,
                    "function_arguments": initial_function_call,
                    "success": False,
                    "extraction_method": "batch_api_content"
                }
            }
            
            with open(cot_filename, "w", encoding="utf-8") as f:
                json.dump(cot_data, f, indent=2, ensure_ascii=False)
            
            return {
                "question_number": question_number,
                "status": "no_function_call",
                "cot_reasoning": initial_reasoning,
                "cot_length": len(initial_reasoning)
            }
        
        # Start embedding validation loop
        current_reasoning = initial_reasoning
        current_function_call = initial_function_call
        final_similarity = 0.0
        final_confidence = 0.0
        status = "low_similarity"
        attempt_history = []
        
        # Regular embedding validation loop
        for attempt in range(max_trials):
            print(f"[Question {question_number}] Embedding validation attempt {attempt + 1}")
            
            validation_result = perform_embedding_validation(current_reasoning, current_function_call, question_number)
            similarity_score = validation_result.get('similarity_score', 0.0)
            confidence_score = validation_result.get('confidence_score', 0.0)
            recommendation = validation_result.get('recommendation', 'REJECT')
            
            attempt_history.append({
                "attempt": attempt + 1,
                "similarity_score": similarity_score,
                "confidence_score": confidence_score,
                "recommendation": recommendation,
                "reasoning_length": len(current_reasoning)
            })
            
            print(f"[Question {question_number}] Attempt {attempt + 1}: Similarity={similarity_score:.3f} | Confidence={confidence_score:.1f} | Recommendation={recommendation}")
            
            if similarity_score >= SIMILARITY_THRESHOLD and recommendation == "ACCEPT":
                final_similarity = similarity_score
                final_confidence = confidence_score
                status = "accepted"
                break
            elif attempt < max_trials - 1:  
                # Generate new attempt
                print(f"[Question {question_number}] Generating new attempt")
                new_reasoning, new_function_call = generate_new_attempt(question_number, attempt + 1)
                
                if new_reasoning and new_function_call:
                    current_reasoning = new_reasoning
                    current_function_call = new_function_call
                    print(f"[Question {question_number}] New attempt generated, length: {len(new_reasoning)} chars")
                else:
                    print(f"[Question {question_number}] Failed to generate new attempt")
                    final_similarity = similarity_score
                    final_confidence = confidence_score
                    status = "generation_failed"
                    break
            else:
                # Last attempt
                final_similarity = similarity_score
                final_confidence = confidence_score
                status = "max_trials_reached"
        
        # If we got a good solution add it to references
        if final_similarity >= SIMILARITY_THRESHOLD:
            reference_manager.add_successful_solution(current_reasoning, current_function_call, final_confidence)
        
        # Save CoT file
        cot_filename = f"embedding_cot_reasoning_question{question_number}.json"
        cot_data = {
            "puzzle_file": f"inputs/questionText{question_number}.json",
            "model_used": MODEL_NAME,
            "processing_type": "chunked_batch_api_with_embedding",
            "solving_session": {
                "raw_llm_response": current_reasoning,
                "cot_reasoning": current_reasoning,
                "reasoning_length": len(current_reasoning),
                "function_called": function_called,
                "function_arguments": current_function_call,
                "success": bool(current_function_call),
                "extraction_method": "batch_api_content",
                "final_similarity_score": final_similarity,
                "final_confidence_score": final_confidence,
                "status": status
            },
            "embedding_validation": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "embedding_model": EMBEDDING_MODEL,
                "max_trials": max_trials,
                "final_similarity": final_similarity,
                "final_confidence": final_confidence,
                "validation_status": status,
                "attempt_history": attempt_history,
                "best_reference": validation_result.get("best_reference")
            }
        }
        
        with open(cot_filename, "w", encoding="utf-8") as f:
            json.dump(cot_data, f, indent=2, ensure_ascii=False)
        print(f"[Question {question_number}] CoT saved to: {cot_filename}")
        
        # Execute tool function
        json_file_path = os.path.join(INPUTS_FOLDER, f"questionText{question_number}.json")
        if current_function_call is not None:
            current_function_call["source_filename"] = json_file_path
        execute_result = execute_tool_function("generate_visual_panel", current_function_call, json_file_path)
        
        return {
            "question_number": question_number,
            "status": status,
            "similarity_score": final_similarity,
            "confidence_score": final_confidence,
            "cot_reasoning": current_reasoning,
            "cot_length": len(current_reasoning),
            "function_call": current_function_call,
            "validation_summary": validation_result,
            "execution_result": execute_result,
            "output_file": execute_result.get("filename") if execute_result.get("status") == "success" else None,
            "total_attempts": len(attempt_history),
            "attempt_history": attempt_history
        }
        
    except Exception as e:
        print(f"[Question {question_number}] ERROR: {str(e)}")
        return {
            "question_number": question_number,
            "status": "processing_error",
            "error": str(e)
        }


def process_chunk_results(batch_output_file, chunk_id):
    """Process results from a specific chunk"""
    if not os.path.exists(batch_output_file):
        print(f"Chunk {chunk_id} output file {batch_output_file} not found!")
        return []
    
    results_to_process = []
    with open(batch_output_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                custom_id = result['custom_id']
                match = re.search(r'chunk\d+-request-(\d+)', custom_id)
                if match:
                    question_number = match.group(1)
                    results_to_process.append((result, question_number))
    
    print(f"Processing chunk {chunk_id} with {len(results_to_process)} results in parallel...")
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_question = {
            executor.submit(process_batch_result_with_embedding, result, qnum): qnum 
            for result, qnum in results_to_process
        }
        
        for future in as_completed(future_to_question):
            question_number = future_to_question[future]
            try:
                result = future.result()
                results.append(result)
                print(f" [Chunk {chunk_id}] Question {question_number} completed")
            except Exception as e:
                print(f" [Chunk {chunk_id}] Question {question_number} error: {e}")
                results.append({
                    "question_number": question_number,
                    "status": "processing_error",
                    "error": str(e)
                })
    
    results.sort(key=lambda x: int(x["question_number"]))
    return results


def process_single_chunk(chunk_files, chunk_id):
    """Process a single chunk from start to finish"""
    print(f"\n{'='*50}")
    print(f"PROCESSING CHUNK {chunk_id} ({len(chunk_files)} files)")
    print(f"{'='*50}")
    
    # Create batch request file
    batch_input_file, num_requests = create_batch_request_file(chunk_files, chunk_id)
    
    # Submit batch
    batch_id = submit_batch(batch_input_file, chunk_id)
    if not batch_id:
        print(f"Failed to submit batch chunk {chunk_id}")
        return []
    
    # Wait for completion
    batch = wait_for_batch_completion(batch_id, chunk_id)
    if not batch:
        print(f"Batch chunk {chunk_id} processing failed")
        return []
    
    # Download results
    batch_output_file = download_batch_results(batch, chunk_id)
    if not batch_output_file:
        print(f"Failed to download batch chunk {chunk_id} results")
        return []
    
    # Process results
    chunk_results = process_chunk_results(batch_output_file, chunk_id)
    
    # Cleanup
    if os.path.exists(batch_input_file):
        os.remove(batch_input_file)
        print(f"Cleaned up: {batch_input_file}")
    
    print(f"Chunk {chunk_id} completed: {len(chunk_results)} results")
    return chunk_results


def main():
    """Main function for chunked batch API processing with embedding validation"""
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Max trials: {MAX_ATTEMPTS}")
    print(f"Batch chunk size: {BATCH_CHUNK_SIZE} files per batch")
    print(f"Delay between attempts: {DELAY_BETWEEN_ATTEMPTS}s")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Missing OPENAI_API_KEY in .env file")
        return
    
    # Get input files
    json_files = get_json_files_from_inputs()
    if not json_files:
        print("No questionText*.json files found in inputs folder!")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    # Split into chunks
    file_chunks = chunk_files(json_files, BATCH_CHUNK_SIZE)
    print(f"Split into {len(file_chunks)} chunks of ~{BATCH_CHUNK_SIZE} files each")
    
    # Process chunks sequentially
    all_results = []
    start_time = time.time()
    
    for chunk_id, current_chunk in enumerate(file_chunks, 1):
        chunk_results = process_single_chunk(current_chunk, chunk_id)
        all_results.extend(chunk_results)
        
        print(f"Completed {chunk_id}/{len(file_chunks)} chunks")
        print(f"Total results so far: {len(all_results)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate statistics
    successful = len([r for r in all_results if r.get("status") == "accepted"])
    low_similarity = len([r for r in all_results if r.get("status") == "low_similarity"])
    max_trials_reached = len([r for r in all_results if r.get("status") == "max_trials_reached"])
    no_function = len([r for r in all_results if r.get("status") == "no_function_call"])
    errors = len([r for r in all_results if r.get("status") in ["api_error", "processing_error"]])
    
    similarity_scores = [r.get("similarity_score", 0.0) for r in all_results if r.get("similarity_score", 0.0) > 0]
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    confidence_scores = [r.get("confidence_score", 0) for r in all_results if r.get("confidence_score", 0) > 0]
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    total_attempts = sum([r.get("total_attempts", 1) for r in all_results])
    avg_attempts = total_attempts / len(all_results) if all_results else 0
    
    # Save summary
    summary_data = {
        "processing_session": {
            "total_files": len(json_files),
            "total_chunks": len(file_chunks),
            "chunk_size": BATCH_CHUNK_SIZE,
            "successful": successful,
            "low_similarity": low_similarity,
            "max_trials_reached": max_trials_reached,
            "no_function_call": no_function,
            "errors": errors,
            "model_used": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "processing_type": "chunked_batch_api_with_embedding",
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "average_similarity": avg_similarity,
            "average_confidence": avg_confidence,
            "success_rate": successful / len(json_files) if json_files else 0,
            "total_processing_time": processing_time,
            "average_time_per_question": processing_time / len(json_files) if json_files else 0,
            "total_attempts": total_attempts,
            "average_attempts_per_question": avg_attempts
        },
        "optimization_stats": {
            "embedding_cache_stats": embedding_cache.get_stats(),
            "reference_embeddings_count": len(reference_manager.reference_embeddings)
        },
        "results": all_results
    }
    
    summary_filename = f"embedding_batch_processing_summary_{int(time.time())}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save final reference embeddings
    reference_manager.save_reference_embeddings()
    
    # Print summary
    print("\n" + "="*60)
    print(f"Total files: {len(json_files)}")
    print(f"Total chunks processed: {len(file_chunks)}")
    print(f"Successful (high similarity): {successful} ({successful/len(json_files)*100:.1f}%)")
    print(f"Low similarity: {low_similarity}")
    print(f"Max trials reached: {max_trials_reached}")
    print(f"No function call: {no_function}")
    print(f"Errors: {errors}")
    print(f"Average similarity: {avg_similarity:.3f} (threshold: {SIMILARITY_THRESHOLD})")
    print(f"Average confidence: {avg_confidence:.1f}/10")
    print(f"Average attempts per question: {avg_attempts:.1f}")
    
    print(f"\nPerformance:")
    print(f"Total processing time: {processing_time/60:.1f} minutes")
    print(f"Average time per question: {processing_time/len(json_files):.1f}s")
    
    cache_stats = embedding_cache.get_stats()
    print(f"\nEmbedding cache stats:")
    print(f"  Hit rate: {cache_stats['hit_rate']}")
    print(f"  Total requests: {cache_stats['total_requests']}")
    print(f"  Cache size: {cache_stats['size']}")
    
    print(f"Reference embeddings: {len(reference_manager.reference_embeddings)}")
    print(f"\nSummary saved to: {summary_filename}")


if __name__ == "__main__":
    main()
