#!/usr/bin/env python3
"""
Submission evaluation script for residual stream interventions.
Handles ARC_easy, 4_answer_MCQA, arithmetic, and ravel_task submissions.

Usage:
    python evaluate_submission.py --submission_folder mock_submission/
"""

import os
import sys
import json
import argparse
import importlib.util
import torch
import gc

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CausalAbstraction.experiments.residual_stream_experiment import PatchResidualStream
from CausalAbstraction.experiments.filter_experiment import FilterExperiment


def parse_submission_folder(folder_name):
    """
    Parse submission folder name to extract task, model, and variable.
    
    Args:
        folder_name (str): Folder name in format {TASK}_{MODEL}_{VARIABLE}
        
    Returns:
        tuple: (task, model, variable) or (None, None, None) if invalid
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None, None, None
    
    # Handle multi-part task names
    if parts[0] == "4" and parts[1] == "answer" and parts[2] == "MCQA":
        task = "4_answer_MCQA"
        model = parts[3]
        variable = "_".join(parts[4:])
    elif parts[0] == "ARC" and parts[1] == "easy":
        task = "ARC_easy"
        model = parts[2]
        variable = "_".join(parts[3:])
    elif parts[0] == "ravel" and parts[1] == "task":
        task = "ravel_task"
        model = parts[2]
        variable = "_".join(parts[3:])
    elif parts[0] == "arithmetic":
        task = "arithmetic"
        model = parts[1]
        variable = "_".join(parts[2:])
    else:
        return None, None, None
    
    # Validate task is supported by this script
    supported_tasks = ["4_answer_MCQA", "ARC_easy", "arithmetic", "ravel_task"]
    if task not in supported_tasks:
        return None, None, None
        
    return task, model, variable


def import_custom_modules(submission_path):
    """
    Import custom featurizer.py and token_position.py from submission folder.
    
    Args:
        submission_path (str): Path to submission folder
        
    Returns:
        tuple: (featurizer_module, token_position_module) or (None, None) if import fails
    """
    featurizer_module = None
    token_position_module = None
    
    # Try to import featurizer.py
    featurizer_path = os.path.join(submission_path, "featurizer.py")
    if os.path.exists(featurizer_path):
        try:
            spec = importlib.util.spec_from_file_location("custom_featurizer", featurizer_path)
            featurizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(featurizer_module)
            print(f"Successfully imported custom featurizer from {featurizer_path}")
        except Exception as e:
            print(f"Error importing featurizer.py: {e}")
    
    # Try to import token_position.py
    token_position_path = os.path.join(submission_path, "token_position.py")
    if os.path.exists(token_position_path):
        try:
            spec = importlib.util.spec_from_file_location("custom_token_position", token_position_path)
            token_position_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(token_position_module)
            print(f"Successfully imported custom token_position from {token_position_path}")
        except Exception as e:
            print(f"Error importing token_position.py: {e}")
    
    return featurizer_module, token_position_module


def get_task_module_and_pipeline(task, model):
    """
    Get the appropriate task module and setup pipeline based on task and model.
    
    Args:
        task (str): Task name
        model (str): Model name
        
    Returns:
        tuple: (task_module, pipeline, causal_model, get_counterfactual_datasets_fn)
    """
    from CausalAbstraction.neural.pipeline import LMPipeline
    
    # Map model names to their full model paths
    model_mapping = {
        "GPT2LMHeadModel": "gpt2",
        "Qwen2ForCausalLM": "Qwen/Qwen2.5-0.5B",  
        "Gemma2ForCausalLM": "google/gemma-2-2b",
        "LlamaForCausalLM": "meta-llama/Meta-Llama-3.1-8B-Instruct"
    }
    
    model_path = model_mapping.get(model, model)
    
    if task == "4_answer_MCQA":
        from tasks.simple_MCQA.simple_MCQA import get_counterfactual_datasets, get_causal_model
        task_module = "simple_MCQA"
    elif task == "ARC_easy":
        from tasks.ARC.ARC import get_counterfactual_datasets, get_causal_model
        task_module = "ARC"
    elif task == "arithmetic":
        from tasks.two_digit_addition_task.arithmetic import get_counterfactual_datasets, get_causal_model
        task_module = "arithmetic"
    elif task == "ravel_task":
        from tasks.RAVEL.ravel import get_counterfactual_datasets, get_causal_model
        task_module = "ravel"
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Create pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(model_path, max_new_tokens=1, device=device, dtype=torch.float16)
    pipeline.tokenizer.padding_side = "left"
    
    # Get causal model
    causal_model = get_causal_model()
    
    return task_module, pipeline, causal_model, get_counterfactual_datasets


def get_token_positions(task, pipeline, causal_model, custom_token_position_module):
    """
    Get token positions for the task, using custom module if available.
    
    Args:
        task (str): Task name
        pipeline: LM pipeline
        causal_model: Causal model
        custom_token_position_module: Custom token position module (if any)
        
    Returns:
        list: List of TokenPosition objects
    """
    # Try custom token position function first
    if custom_token_position_module and hasattr(custom_token_position_module, 'get_token_positions'):
        try:
            return custom_token_position_module.get_token_positions(pipeline, causal_model)
        except Exception as e:
            print(f"Error using custom token positions: {e}")
            print("Falling back to default token positions...")
    
    # Fall back to task-specific defaults
    if task == "4_answer_MCQA":
        from tasks.simple_MCQA.simple_MCQA import get_token_positions
    elif task == "ARC_easy":
        from tasks.ARC.ARC import get_token_positions
    elif task == "arithmetic":
        from tasks.two_digit_addition_task.arithmetic import get_token_positions
    elif task == "ravel_task":
        from tasks.RAVEL.ravel import get_token_positions
    
    return get_token_positions(pipeline, causal_model)


def load_featurizers_from_submission(submission_folder_path, pipeline):
    """
    Load pre-trained featurizers from submission folder.
    
    Args:
        submission_folder_path (str): Path to specific submission folder
        pipeline: LM pipeline with model configuration
        
    Returns:
        dict: Dictionary mapping (layer, position_id) tuples to Featurizer objects
    """
    from CausalAbstraction.neural.featurizers import Featurizer
    
    featurizers = {}
    
    # List all files in the submission folder
    try:
        files = os.listdir(submission_folder_path)
    except FileNotFoundError:
        print(f"Submission folder not found: {submission_folder_path}")
        return featurizers
    
    # Find all featurizer files (those ending with '_featurizer')
    featurizer_files = [f for f in files if f.endswith('_featurizer')]
    
    print(f"Found {len(featurizer_files)} featurizer files")
    
    for featurizer_file in featurizer_files:
        # Extract the base name (without '_featurizer' suffix)
        base_name = featurizer_file[:-11]  # Remove '_featurizer'
        
        # Parse the model unit ID to extract layer and position
        # Expected format: ResidualStream(Layer:X,Token:Y)
        try:
            if "ResidualStream" in base_name and "Layer:" in base_name and "Token:" in base_name:
                # Extract layer number
                layer_start = base_name.find("Layer:") + 6
                layer_end = base_name.find(",", layer_start)
                layer = int(base_name[layer_start:layer_end])
                
                # Extract token position
                token_start = base_name.find("Token:") + 6
                token_end = base_name.find(")", token_start)
                position_id = base_name[token_start:token_end]
                
                # Check if all required files exist
                base_path = os.path.join(submission_folder_path, base_name)
                featurizer_path = base_path + "_featurizer"
                inverse_featurizer_path = base_path + "_inverse_featurizer"
                indices_path = base_path + "_indices"
                
                if not all(os.path.exists(p) for p in [featurizer_path, inverse_featurizer_path, indices_path]):
                    print(f"Missing files for {base_name}")
                    continue
                
                # Load the featurizer
                featurizer = Featurizer.load_modules(base_path)
                
                # Load and set indices if they exist
                try:
                    with open(indices_path, 'r') as f:
                        indices = json.load(f)
                    if indices is not None:
                        featurizer.set_feature_indices(indices)
                except Exception as e:
                    print(f"Warning: Could not load indices for {base_name}: {e}")
                
                # Store in the featurizers dictionary
                featurizers[(layer, position_id)] = featurizer
                print(f"Loaded featurizer for layer {layer}, position {position_id}")
                
        except Exception as e:
            print(f"Error parsing or loading featurizer {base_name}: {e}")
            continue
    
    print(f"Successfully loaded {len(featurizers)} featurizers")
    return featurizers


def evaluate_submission_task(task_folder_path, submission_base_path, private_data=True, public_data=False):
    """
    Evaluate a single submission task folder.
    
    Args:
        task_folder_path (str): Path to the specific task submission folder
        submission_base_path (str): Path to the base submission folder
        private_data (bool): Whether to evaluate on private test data
        public_data (bool): Whether to evaluate on public test data
        
    Returns:
        bool: True if evaluation successful, False otherwise
    """
    folder_name = os.path.basename(task_folder_path)
    task, model, variable = parse_submission_folder(folder_name)
    
    if not all([task, model, variable]):
        print(f"ERROR: Invalid folder name format: {folder_name}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {folder_name}")
    print(f"Task: {task}, Model: {model}, Variable: {variable}")
    print(f"{'='*60}")
    
    try:
        # Import custom modules from base submission folder
        custom_featurizer_module, token_position_module = import_custom_modules(submission_base_path)
        
        # Get task components
        _, pipeline, causal_model, get_counterfactual_datasets = get_task_module_and_pipeline(task, model)
        
        # Load datasets
        print("Loading datasets...")
        dataset_size = None  # Load all data
        counterfactual_datasets = get_counterfactual_datasets(hf=True, size=dataset_size, load_private_data=private_data)
        counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "test" in k}
        
        # Filter datasets based on flags
        if not private_data:
            counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "private" not in k}
        if not public_data:
            counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "test" not in k or "private" in k}
    
        
        print(f"Loaded {len(counterfactual_datasets)} datasets")
        
        # Get token positions
        token_positions = get_token_positions(task, pipeline, causal_model, token_position_module)
        print(f"Using {len(token_positions)} token positions")
        
        # Setup checker function
        def checker(output_text, expected):
            return expected in output_text
        

        batch_size_by_task = {
            "4_answer_MCQA": 128,
            "ARC_easy": 32,
            "arithmetic": 256,
            "ravel_task": 128 
        }

        batch_size = batch_size_by_task.get(task)
        # Filter experiments - only keep examples where model performs well
        print("Filtering datasets based on model performance...")
        filter_experiment = FilterExperiment(pipeline, causal_model, checker)
        filtered_datasets = filter_experiment.filter(counterfactual_datasets, verbose=True, batch_size=batch_size)
        
        # Load pre-trained featurizers from submission
        print("Loading pre-trained featurizers...")
        featurizers = load_featurizers_from_submission(task_folder_path, pipeline)
        
        if not featurizers:
            print("ERROR: No featurizers found in submission folder")
            return False
        
        # Extract layers that actually have featurizers
        layers_with_featurizers = sorted(list(set(layer for layer, _ in featurizers.keys())))
        print(f"Found featurizers for layers: {layers_with_featurizers}")
        
        # Create PatchResidualStream experiment with loaded featurizers
        config = {
            "method_name": "submission",
            "batch_size": batch_size,
            "evaluation_batch_size": batch_size 
        }
        
        # Only use layers that have featurizers in the submission
        layers = layers_with_featurizers
        
        experiment = PatchResidualStream(
            pipeline=pipeline,
            causal_model=causal_model,
            layers=layers,
            token_positions=token_positions,
            checker=checker,
            featurizers=featurizers,
            config=config
        )
        
        # Run evaluation on test data
        print("Running residual stream evaluation...")
        test_data = {k: v for k, v in filtered_datasets.items() if "test" in k}
        
        results = experiment.perform_interventions(
            test_data, 
            verbose=True, 
            target_variables_list=[[variable]], 
            save_dir=task_folder_path
        )
        
        print(f"Successfully evaluated {folder_name}")
        return True
        
    except Exception as e:
        print(f"ERROR evaluating {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Evaluate submissions for residual stream tasks")
    parser.add_argument("--submission_folder", required=True, 
                       help="Path to submission folder containing task subfolders")
    parser.add_argument("--private_data", action="store_true", default=True,
                       help="Evaluate on private test data (default: True)")
    parser.add_argument("--public_data", action="store_true", default=False,
                       help="Also evaluate on public test data (default: False)")
    parser.add_argument("--specific_task", type=str, default=None,
                       help="Evaluate only a specific task folder")
    
    args = parser.parse_args()
    
    submission_path = os.path.abspath(args.submission_folder)
    
    if not os.path.exists(submission_path):
        print(f"ERROR: Submission folder does not exist: {submission_path}")
        return 1
    
    print(f"Evaluating submissions in: {submission_path}")
    print(f"Private data: {args.private_data}")
    print(f"Public data: {args.public_data}")
    
    # Find all task folders
    task_folders = []
    for item in os.listdir(submission_path):
        item_path = os.path.join(submission_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Skip non-task folders (like __pycache__)
            if item in ['__pycache__', '.git']:
                continue
            task, model, variable = parse_submission_folder(item)
            if all([task, model, variable]):
                if args.specific_task is None or item == args.specific_task:
                    task_folders.append(item_path)
    
    if not task_folders:
        print("ERROR: No valid task folders found in submission directory")
        return 1
    
    print(f"Found {len(task_folders)} task folders to evaluate")
    
    # Evaluate each task folder
    successful = 0
    total = len(task_folders)
    
    for task_folder_path in task_folders:
        if evaluate_submission_task(task_folder_path, submission_path, args.private_data, args.public_data):
            successful += 1
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"Successfully evaluated: {successful}/{total} submissions")
    print(f"{'='*60}")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())