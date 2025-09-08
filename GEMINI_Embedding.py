import os
import json
import time
import re
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
from collections import OrderedDict
from tool import tool_functions, execute_tool_function
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from google.protobuf.json_format import MessageToDict
import threading


INPUTS_FOLDER = "inputs"
CHECK_INTERVAL = 10
MAX_ATTEMPTS = 3
SIMILARITY_THRESHOLD = 0.8  # Embedding similarity threshold
DELAY_BETWEEN_ATTEMPTS = 0.1


CHUNK_SIZE = 200  
MAX_CONCURRENT_FILES = 12  

# === EMBEDDING SETTINGS ===
EMBEDDING_MODEL = "text-embedding-3-small" 
REFERENCE_EMBEDDINGS_FILE = "reference_solution_embeddings.json"

# === GEMINI API SETTINGS ===
API_RETRY_ATTEMPTS = 2
API_RETRY_DELAY = 1.0


load_dotenv()
MODEL_NAME = "gemini-1.5-flash"  


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # For embeddings


def get_gemini_tools():
    """Define tools in Gemini's  format"""
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
    """Enhanced function extraction optimized for Gemini """
    arguments = {}
    
    try:
        # Method 1: MessageToDict 
        if hasattr(function_call, '_pb'):
            call_dict = MessageToDict(function_call._pb)
            arguments = call_dict.get('args', {})
            if arguments and 'config_type' in arguments:
                return arguments
    except Exception as e:
        pass
    
    try:
        # Method 2: Direct args access
        if hasattr(function_call, 'args'):
            if hasattr(function_call.args, '_pb'):
                arguments = MessageToDict(function_call.args._pb)
                if arguments and 'config_type' in arguments:
                    return arguments
    except Exception as e:
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
            
    except Exception as e:
        pass
    
    return arguments


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
        self.lock = threading.Lock()  # Thread safety
    
    def get_key(self, text):
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text):
        """Get embedding from cache if exists"""
        with self.lock:
            key = self.get_key(text)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats['hits'] += 1
                return self.cache[key]
            
            self.stats['misses'] += 1
            return None
    
    def set(self, text, embedding):
        """Store embedding in cache"""
        with self.lock:
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


class ReferenceSolutionManager:
    """Manages reference solution embeddings for similarity comparison"""
    def __init__(self, reference_file=REFERENCE_EMBEDDINGS_FILE):
        self.reference_file = reference_file
        self.reference_embeddings = []
        self.reference_metadata = []
        self.load_reference_embeddings()
    
    def load_reference_embeddings(self):
        """Load pre-computed reference embeddings"""
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
            print("No reference embeddings file found, creating default references")
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
        
        print("Creating embeddings for default reference patterns...")
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
        if confidence_score >= 8.0:  # Only add high-confidence solutions
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
    """Get embedding for text with caching """
    # Check cache first
    cached_embedding = embedding_cache.get(text)
    if cached_embedding is not None:
        return cached_embedding
    
    try:
        response = openai_client.embeddings.create(
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
    """Perform embedding based validation of the reasoning and solution"""
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


def call_gemini_with_tools(user_prompt, system_prompt, temperature=0.1):
    """ Gemini API call for 1.5 Flash"""
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            gemini_tools = get_gemini_tools()
            
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = model.generate_content(
                full_prompt,
                tools=gemini_tools,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=4000,
                    candidate_count=1,
                )
            )
            
            cot_reasoning = ""
            function_calls = []
            
           
            if not response or not response.candidates:
                raise ValueError("No response candidates from Gemini 1.5 Flash")
                
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
            
            # Extract function call
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
                print(f"Gemini 1.5 Flash API retry {attempt + 1}/{API_RETRY_ATTEMPTS}: {e}")
                time.sleep(API_RETRY_DELAY)
                continue
            else:
                print(f"Gemini 1.5 Flash API failed after {API_RETRY_ATTEMPTS} attempts: {e}")
                return {
                    "cot_reasoning": f"API Error: {str(e)}",
                    "function_called": None,
                    "function_call": None,
                    "success": False,
                    "error": str(e)
                }


def generate_new_attempt(question_number, attempt_count):
    """Generate a new attempt for the question using Gemini """
    # Load original puzzle data
    json_file_path = os.path.join(INPUTS_FOLDER, f"questionText{question_number}.json")
    if not os.path.exists(json_file_path):
        return None, None
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_json_data = f.read()
    
    # Create a new prompt with slight variation to encourage different reasoning
    temperature = min(0.3 + (attempt_count * 0.1), 0.7)  # Increase temperature for later attempts
    
    # Same prompts as OpenAI version for consistency
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
        
        result = call_gemini_with_tools(user_prompt, system_prompt, temperature)
        
        if result["success"]:
            return result["cot_reasoning"], result["function_call"]
        else:
            return None, None
        
    except Exception as e:
        print(f"Error generating new attempt: {e}")
        return None, None


def get_json_files_from_inputs():
    """Get all questionText*.json files from inputs folder sorted by number"""
    if not os.path.exists(INPUTS_FOLDER):
        print(f"Error: {INPUTS_FOLDER} folder not found!")
        return []
    
    json_files = []
    for filename in os.listdir(INPUTS_FOLDER):
        if filename.startswith('questionText') and filename.endswith('.json'):
            filepath = os.path.join(INPUTS_FOLDER, filename)
            if os.path.getsize(filepath) > 0:  # Check if file is not empty
                json_files.append(filepath)
    
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'questionText(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    json_files.sort(key=extract_number)
    return json_files


def chunk_files(json_files, chunk_size):
    """Split files into chunks for processing"""
    chunks = []
    for i in range(0, len(json_files), chunk_size):
        chunk = json_files[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def process_single_file_with_embedding(json_file_path, max_trials=3):
    """Process a single file with embedding based validation"""
    filename = os.path.basename(json_file_path)
    
    try:
        # Read file
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_json_data = f.read()
        
        number_match = re.search(r'(\d+)', filename)
        question_number = number_match.group(1) if number_match else filename
        
        # Same prompts as OpenAI version for consistency
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

        # Make initial call to Gemini 
        result = call_gemini_with_tools(user_prompt, system_prompt)
        
        if not result["success"]:
            # Save failure case
            cot_filename = f"embedding_cot_reasoning_question{question_number}.json"
            cot_data = {
                "puzzle_file": json_file_path,
                "model_used": MODEL_NAME,
                "processing_type": "gemini15_with_embedding",
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
        
        initial_reasoning = result["cot_reasoning"]
        initial_function_call = result["function_call"]
        
        print(f"[Question {question_number}] Initial CoT length: {len(initial_reasoning)} chars")
        print(f"[Question {question_number}] Function call captured: {result['function_called']}")
        
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
            elif attempt < max_trials - 1:  # Not the last attempt
                # Generate new attempt
                print(f"[Question {question_number}] Generating new attempt...")
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
        
        # If we got a good solution, add it to references
        if final_similarity >= SIMILARITY_THRESHOLD:
            reference_manager.add_successful_solution(current_reasoning, current_function_call, final_confidence)
        
        # Save CoT file
        cot_filename = f"embedding_cot_reasoning_question{question_number}.json"
        cot_data = {
            "puzzle_file": json_file_path,
            "model_used": MODEL_NAME,
            "processing_type": "gemini15_with_embedding",
            "solving_session": {
                "raw_llm_response": current_reasoning,
                "cot_reasoning": current_reasoning,
                "reasoning_length": len(current_reasoning),
                "function_called": result["function_called"],
                "function_arguments": current_function_call,
                "success": bool(current_function_call),
                "extraction_method": "gemini15_enhanced",
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


def process_chunk_parallel(chunk_files, chunk_id):
    """Process a chunk of files in parallel"""
    print(f"\n{'='*50}")
    print(f"PROCESSING CHUNK {chunk_id} ({len(chunk_files)} files)")
    print(f"{'='*50}")
    
    results = []
    
    # Process files in parallel within the chunk
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as executor:
        future_to_file = {
            executor.submit(process_single_file_with_embedding, json_file_path): json_file_path 
            for json_file_path in chunk_files
        }
        
        for future in as_completed(future_to_file):
            json_file_path = future_to_file[future]
            filename = os.path.basename(json_file_path)
            try:
                result = future.result()
                results.append(result)
                print(f"✓ [Chunk {chunk_id}] {filename} completed")
            except Exception as e:
                print(f"✗ [Chunk {chunk_id}] {filename} error: {e}")
                number_match = re.search(r'(\d+)', filename)
                question_number = number_match.group(1) if number_match else filename
                results.append({
                    "question_number": question_number,
                    "status": "processing_error",
                    "error": str(e)
                })
    
    # Sort results by question number
    results.sort(key=lambda x: int(x["question_number"]) if str(x["question_number"]).isdigit() else 0)
    
    print(f"Chunk {chunk_id} completed: {len(results)} results")
    return results


def main():
    """Main function for chunked Gemini  processing with embedding validation"""
    print(f"Model: {MODEL_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"Max trials: {MAX_ATTEMPTS}")
    print(f"Chunk size: {CHUNK_SIZE} files per chunk")
    print(f"Max concurrent files per chunk: {MAX_CONCURRENT_FILES}")
    print(f"Delay between attempts: {DELAY_BETWEEN_ATTEMPTS}s")
    print(f"API retry attempts: {API_RETRY_ATTEMPTS}")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Missing GEMINI_API_KEY in .env file")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Missing OPENAI_API_KEY in .env file (needed for embeddings)")
        return
    
    # Get input files
    json_files = get_json_files_from_inputs()
    if not json_files:
        print("No questionText*.json files found in inputs folder!")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    # Split into chunks
    file_chunks = chunk_files(json_files, CHUNK_SIZE)
    print(f"Split into {len(file_chunks)} chunks of ~{CHUNK_SIZE} files each")
    
    # Process chunks sequentially 
    all_results = []
    start_time = time.time()
    
    for chunk_id, current_chunk in enumerate(file_chunks, 1):
        chunk_results = process_chunk_parallel(current_chunk, chunk_id)
        all_results.extend(chunk_results)
        
        print(f"Completed {chunk_id}/{len(file_chunks)} chunks")
        print(f"Total results so far: {len(all_results)}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate comprehensive statistics
    successful = len([r for r in all_results if r.get("status") == "accepted"])
    low_similarity = len([r for r in all_results if r.get("status") == "low_similarity"])
    max_trials_reached = len([r for r in all_results if r.get("status") == "max_trials_reached"])
    no_function = len([r for r in all_results if r.get("status") == "no_function_call"])
    errors = len([r for r in all_results if r.get("status") in ["processing_error"]])
    
    similarity_scores = [r.get("similarity_score", 0.0) for r in all_results if r.get("similarity_score", 0.0) > 0]
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    
    confidence_scores = [r.get("confidence_score", 0) for r in all_results if r.get("confidence_score", 0) > 0]
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    total_attempts = sum([r.get("total_attempts", 1) for r in all_results])
    avg_attempts = total_attempts / len(all_results) if all_results else 0
    
    # Save comprehensive summary
    summary_data = {
        "processing_session": {
            "total_files": len(json_files),
            "total_chunks": len(file_chunks),
            "chunk_size": CHUNK_SIZE,
            "max_concurrent_files": MAX_CONCURRENT_FILES,
            "successful": successful,
            "low_similarity": low_similarity,
            "max_trials_reached": max_trials_reached,
            "no_function_call": no_function,
            "errors": errors,
            "model_used": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "processing_type": "gemini15_with_embedding",
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
            "reference_embeddings_count": len(reference_manager.reference_embeddings),
            "api_stability": {
                "retry_attempts": API_RETRY_ATTEMPTS,
                "retry_delay": API_RETRY_DELAY,
                "model_version": "gemini-1.5-flash"
            }
        },
        "results": all_results
    }
    
    summary_filename = f"gemini15_embedding_processing_summary_{int(time.time())}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save final reference embeddings
    reference_manager.save_reference_embeddings()
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("GEMINI  EMBEDDING-BASED PROCESSING COMPLETE")
    print("="*60)
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
    
    print(f"\nGemini  optimizations:")
    print(f"  Enhanced function extraction: 4 methods")
    print(f"  Improved API stability: {API_RETRY_ATTEMPTS} retry attempts")
    print(f"  Optimized concurrency: {MAX_CONCURRENT_FILES} files per chunk")
    print(f"  Same embedding validation as OpenAI version")
    
    print(f"\nSummary saved to: {summary_filename}")


if __name__ == "__main__":
    main()
