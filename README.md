# An-Analysis-of-Architectural-Impact-on-LLM-based-Abstract-Visual-Reasoning
This repository provides a **comprehensive framework** for solving **Raven’s Progressive Matrices (RPMs)** using **Large Language Models (LLMs)** and **multi-agent reasoning architectures**.    
It was designed for systematic evaluation of **abstract visual reasoning** with different paradigms: direct reasoning, reflection, multi-agent decomposition, and embedding-based validation.    

This work accompanies the experiments described in the paper:    
**_"An Analysis of Architectural Impact on LLM-based Abstract Visual Reasoning: A Systematic Benchmark on RAVEN-FAIR"_**.    
A preprint of the paper will be linked here once available.  

---

## Core Idea  

The system tests whether modern LLMs (GPT, LLaMA, Gemini, Claude) can perform **structured reasoning** over visual puzzles.   
Each solver takes structured JSON puzzle descriptions and produces the missing answer panel through:  

1. **Chain-of-Thought reasoning (CoT)** – step-by-step analysis of structure, shapes, colors, sizes, rotations.    
2. **Tool function call** – invoking `generate_visual_panel(config_type, objects)` to render the predicted panel.    
3. **Evaluation** – comparing reasoning styles and generated panels across different architectures.  
 
---

## Repository Structure  

### 1. **Baseline – Single-Shot Solvers**  
Direct reasoning without feedback.    
- `LLAMA_Single.py` – LLaMA via Groq API    
- `GPT_Single.py` – OpenAI GPT    
- `GEMINI_Single.py` – Google Gemini (multi-threaded)    
- `CLAUDE_Single.py` – Anthropic Claude    

### 2. **Self-Reflection Solvers**  
Models re-evaluate their own outputs, detect errors, and retry if necessary.    
- `LLAMA_SelfReflection.py`    
- `GPT_SelfReflection.py`    
- `GEMINI_SelfReflection.py`    
- `CLAUDE_SelfReflection.py`    

### 3. **Multi-Agent Solvers**  
Problem is decomposed into **feature-specialist agents** working in parallel:    
- `config_agent` → layout type (singleton, grid, etc.)    
- `shape_agent` → geometric progression of shapes    
- `color_agent` → grayscale progression (0–9)    
- `size_agent` → scaling values (0.1–1.0)    
- `angle_agent` → rotations (-180 to 180°)    

Files:    
- `LLAMA_MultiModel.py`    
- `GPT_MultiModel.py`    
- `CLAUDE_MultiModel.py`    
- `GEMINI_MultiModel.py`  

### 4. **Embedding-Based Validation**  
Uses **embedding similarity** against a database of high-quality reference solutions.    
This allows systematic detection of **semantic drift**  in reasoning.    

Files:    
- `LLAMA_Embedding.py`    
- `GPT_Embedding.py`    
- `GEMINI_Embedding.py`    
- `CLAUDE_Embedding.py`    

### 5. **Shared Tools**  
- `tool.py` – contains the `generate_visual_panel` function used across all solvers.    

---

## ⚙How It Works  

1. **Input**    
   Place RPM puzzle descriptions (`questionText*.json`) in the `inputs/` directory.    

2. **Reasoning**    
   Each solver generates **step-by-step CoT reasoning**, focusing on one or more features.    

3. **Panel Generation**    
   The model calls `generate_visual_panel()` with precise parameters (`config_type`, list of objects).    

4. **Validation (optional)**    
   - Reflection solvers re-check answers.    
   - Multi-agent solvers aggregate feature-specific votes.    
   - Embedding solvers compare to a **reference embedding memory** to detect inconsistencies.    

5. **Output**    
   - Reasoning traces: `cot_reasoning_questionX.json`    
   - Generated panel images: `answer_panel_X.png`    
   - Embedding similarity logs (for embedding-based solvers).    

---

## Usage  

### Install dependencies  
```bash  
git clone <repo-url>  
cd <repo-name>  
pip install -r requirements.txt
```bash 

### 2. Configure API keys
Create a `.env` file in the project root:

```env  
OPENAI_API_KEY=xxxx  
GEMINI_API_KEY=xxxx  
ANTHROPIC_API_KEY=xxxx  
GROQ_API_KEY=xxxx  


### 3. Prepare input puzzles

All solvers expect puzzle definitions in **JSON format**.    
Place them inside the `inputs/` directory, for example:
inputs/questionText1.json  
inputs/questionText2.json

### 4. Run solvers

#### Single-Shot (baseline)
```bash  
python GPT_Single.py  
python LLAMA_Single.py  
python GEMINI_Single.py  
python CLAUDE_Single.py

#### Self-Reflection
```bash  
python GPT_SelfReflection.py
python LLAMA_SelfReflection.py
python GEMINI_SelfReflection.py
python CLAUDE_SelfReflection.py

#### Multi-Agent  
```bash  
python GPT_MultiModel.py  
python LLAMA_MultiModel.py  
python CLAUDE_MultiModel.py  
python GEMINI_MultiModel.py  

#### Embedding-based validation  
```bash    
python GPT_Embedding.py  
python LLAMA_Embedding.py  
python GEMINI_Embedding.py  
python CLAUDE_Embedding.py


