#!/usr/bin/env python3
"""
IOI submission evaluation script for attention head interventions.
Handles IOI task submissions with DAS-trained attention heads.

Usage:
    python ioi_evaluate_submission.py --submission_folder ioi_submission/
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

from CausalAbstraction.experiments.attention_head_experiment import PatchAttentionHeads, AttentionHead
from CausalAbstraction.experiments.filter_experiment import FilterExperiment


def parse_ioi_submission_folder(folder_name):
    """
    Parse IOI submission folder name to extract task, model, and variable.
    
    Args:
        folder_name (str): Folder name in format ioi_task_{MODEL}_{VARIABLE}
        
    Returns:
        tuple: (task, model, variable) or (None, None, None) if invalid
    """
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None, None, None
    
    # Expected format: ioi_task_{MODEL}_{VARIABLE}
    if parts[0] == "ioi" and parts[1] == "task":
        task = "ioi_task"
        model = parts[2]
        variable = "_".join(parts[3:])
    else:
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


def load_linear_params(submission_path, model_name):
    """
    Load linear parameters for IOI causal model from any JSON file in submission folder.
    
    Args:
        submission_path (str): Path to submission folder
        model_name (str): Model name to look up parameters for
        
    Returns:
        dict: linear_params_dict
    """
    # Look for any JSON file in the submission folder
    json_files = [f for f in os.listdir(submission_path) if f.endswith('.json')]
    
    linear_params_file = None
    all_coeffs = None
    
    # Try each JSON file until we find one with valid linear parameters
    for json_file in json_files:
        file_path = os.path.join(submission_path, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if this JSON contains model coefficients
            # It should have model names as keys with coefficient dictionaries
            if isinstance(data, dict) and any(isinstance(v, dict) and 'bias' in v and 'token_coeff' in v and 'position_coeff' in v for v in data.values()):
                all_coeffs = data
                linear_params_file = file_path
                print(f"Found linear parameters in: {json_file}")
                break
        except Exception as e:
            # Skip files that can't be parsed or don't have the right structure
            continue
    
    if all_coeffs is None:
        # Fall back to baselines folder
        linear_params_file = "baselines/ioi_linear_params.json"
        print(f"No valid linear parameters found in submission folder, falling back to: {linear_params_file}")
        try:
            with open(linear_params_file, 'r') as f:
                all_coeffs = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load linear parameters from any source: {e}")

    # Find the coefficients for this model.
    # Submissions key the file by short name ("gpt2", "qwen", ...), as written by
    # ioi_learn_linear_params.py, while model_name is the class name.
    short_names = {"GPT2LMHeadModel": "gpt2", "Qwen2ForCausalLM": "qwen",
                   "LlamaForCausalLM": "llama", "Gemma2ForCausalLM": "gemma"}
    if model_name in all_coeffs:
        coeffs = all_coeffs[model_name]
    elif short_names.get(model_name) in all_coeffs:
        coeffs = all_coeffs[short_names[model_name]]
    elif "gemma" in model_name.lower() and "gemma" in all_coeffs:
        coeffs = all_coeffs["gemma"]
    elif "default" in all_coeffs:
        coeffs = all_coeffs["default"]
    else:
        # Use the first available coefficients
        coeffs = next(iter(all_coeffs.values()))

    # Validate required keys
    required_keys = ['bias', 'token_coeff', 'position_coeff']
    for key in required_keys:
        if key not in coeffs:
            raise ValueError(f"Missing required key '{key}' in linear_coeffs for model {model_name}")

    linear_params = {
        "bias": float(coeffs['bias']),
        "token_coeff": float(coeffs['token_coeff']),
        "position_coeff": float(coeffs['position_coeff'])
    }
    
    print(f"Using linear parameters:")
    print(f"  bias: {linear_params['bias']}")
    print(f"  token_coeff: {linear_params['token_coeff']}")
    print(f"  position_coeff: {linear_params['position_coeff']}")
    
    return linear_params


def get_ioi_task_components(model_name, linear_params):
    """
    Get IOI task components including pipeline and causal model.
    
    Args:
        model_name (str): Model name
        linear_params (dict): Linear parameters for causal model
        
    Returns:
        tuple: (pipeline, causal_model, get_counterfactual_datasets, get_token_positions)
    """
    from tasks.IOI_task.ioi_task import get_causal_model, get_counterfactual_datasets, get_token_positions
    from baselines.ioi_baselines.ioi_utils import setup_pipeline
    
    # Map model class names to their identifiers
    model_mapping = {
        "Gemma2ForCausalLM": "gemma",
        "GPT2LMHeadModel": "gpt2",
        "Qwen2ForCausalLM": "qwen",
        "LlamaForCausalLM": "llama"
    }
    
    model_id = model_mapping.get(model_name, model_name.lower())
    
    # Setup pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, _ = setup_pipeline(model_id, device, eval_batch_size=None)
    
    # Get causal model with linear parameters
    causal_model = get_causal_model(linear_params)
    
    return pipeline, causal_model, get_counterfactual_datasets, get_token_positions


def get_token_positions_for_ioi(pipeline, causal_model, custom_token_position_module):
    """
    Get token positions for IOI task, using custom module if available.
    
    Args:
        pipeline: LM pipeline
        causal_model: Causal model
        custom_token_position_module: Custom token position module (if any)
        
    Returns:
        list: List of TokenPosition objects
    """
    # Try custom token position function first
    if custom_token_position_module and hasattr(custom_token_position_module, 'get_token_positions'):
        try:
            print("Using custom token positions")
            return custom_token_position_module.get_token_positions(pipeline, causal_model)
        except Exception as e:
            print(f"Error using custom token positions: {e}")
            print("Falling back to default token positions...")
    
    # Fall back to default IOI token positions
    print("Using default IOI token positions")
    from tasks.IOI_task.ioi_task import get_token_positions
    return get_token_positions(pipeline, causal_model)


def load_attention_head_featurizers(submission_folder_path, token_positions):
    """
    Load pre-trained attention head featurizers from submission folder.
    
    Args:
        submission_folder_path (str): Path to specific submission folder
        
    Returns:
        dict: Dictionary mapping (layer, head) tuples to Featurizer objects
    """
    from CausalAbstraction.neural.featurizers import Featurizer
    
    featurizers = {}
    
    # List all files in the submission folder
    try:
        files = os.listdir(submission_folder_path)
    except FileNotFoundError:
        print(f"Submission folder not found: {submission_folder_path}")
        return featurizers
    
    # Find all main featurizer files (ending with '_featurizer' but not '_inverse_featurizer')
    featurizer_files = [f for f in files if f.endswith('_featurizer') and not f.endswith('_inverse_featurizer')]
    
    print(f"Found {len(featurizer_files)} featurizer files")
    
    for featurizer_file in featurizer_files:
        # Extract the base name (without '_featurizer' suffix)
        base_name = featurizer_file[:-11]  # Remove '_featurizer'
        
        # Parse the model unit ID to extract layer and head
        # Expected format: AttentionHead(Layer*X,Head*Y,Token*all)
        # where * is some character
        try:
            if "AttentionHead" in base_name and "Layer" in base_name and "Head" in base_name:
                # Extract layer and head from the base name
                model_unit = AttentionHead.load_modules(base_name, submission_folder_path, token_positions)
                featurizer =model_unit.featurizer
                layer = model_unit.component.layer
                head = model_unit.head
                # Store in the featurizers dictionary with position_id "all"
                featurizers[(layer, head)] = featurizer
                print(f"Loaded featurizer for layer {layer}, head {head}")
                
        except Exception as e:
            print(f"Error parsing or loading featurizer {base_name}: {e}")
            continue
    
    print(f"Successfully loaded {len(featurizers)} attention head featurizers")
    return featurizers



def evaluate_ioi_submission_task(task_folder_path, submission_base_path, private_data=True, public_data=False, quick_test=False):
    """
    Evaluate a single IOI submission task folder.
    
    Args:
        task_folder_path (str): Path to the specific task submission folder
        submission_base_path (str): Path to the base submission folder
        private_data (bool): Whether to evaluate on private test data
        public_data (bool): Whether to evaluate on public test data
        
    Returns:
        bool: True if evaluation successful, False otherwise
    """
    folder_name = os.path.basename(task_folder_path)
    task, model, variable = parse_ioi_submission_folder(folder_name)
    
    if not all([task, model, variable]):
        # If this is a subfolder, use the parent folder name
        folder_name = os.path.basename(os.path.dirname(task_folder_path))
        task, model, variable = parse_ioi_submission_folder(folder_name)
        if not all([task, model, variable]):
            print(f"ERROR: Invalid IOI folder name format: {folder_name}")
            return False
    
    print(f"\n{'='*60}")
    print(f"Evaluating IOI submission: {folder_name}")
    print(f"Task: {task}, Model: {model}, Variable: {variable}")
    print(f"{'='*60}")
    
    try:
        # Import custom modules from base submission folder
        _, token_position_module = import_custom_modules(submission_base_path)
        
        # Load linear parameters from ioi_linear_params.json
        linear_params = load_linear_params(submission_base_path, model)
        
        # Get task components
        pipeline, causal_model, get_counterfactual_datasets, _ = get_ioi_task_components(model, linear_params)
        
        # Load datasets
        print("Loading IOI datasets...")
        size = 10 if quick_test else None  # Limit size for quick test
        counterfactual_datasets = get_counterfactual_datasets(hf=True, size=size, load_private_data=private_data)
        
        print(f"Loaded {len(counterfactual_datasets)} datasets")
        
        # Get token positions
        token_positions = get_token_positions_for_ioi(pipeline, causal_model, token_position_module)
        print(f"Using {len(token_positions)} token positions")
        
        # Setup checker function for IOI
        from baselines.ioi_baselines.ioi_utils import filter_checker
        
        # Filter experiments - only keep examples where model performs well
        print("Filtering datasets based on model performance...")
        filter_experiment = FilterExperiment(pipeline, causal_model, filter_checker)

        counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "test" in k and "same" not in k}
        # Since we already loaded the appropriate data based on flags, 
        # we just need to filter by public/private if both are loaded
        if private_data and not public_data:
            # Only keep private test data
            counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "private" in k}
        elif public_data and not private_data:
            # Only keep public test data (this is the default case)
            counterfactual_datasets = {k: v for k, v in counterfactual_datasets.items() if "private" not in k}
        # If both flags are True, keep all test data

        filtered_datasets = filter_experiment.filter(counterfactual_datasets, verbose=True, batch_size=1024)
        
        # Load pre-trained attention head featurizers from submission
        print("Loading pre-trained attention head featurizers...")
        featurizers = load_attention_head_featurizers(task_folder_path, token_positions)
        
        if not featurizers or len(featurizers) == 0:
            print("ERROR: No attention head featurizers found in submission folder")
            return False
        
        # Extract attention heads that actually have featurizers (convert from (layer, head, position) to (layer, head))
        attention_heads = featurizers.keys()
        print(f"Found featurizers for attention heads: {attention_heads}")
        
        # Setup checker function for evaluation
        from baselines.ioi_baselines.ioi_utils import checker
        def eval_checker(raw_output, example_setting):
            return checker(raw_output, example_setting, pipeline)
        
        # Create PatchAttentionHeads experiment with loaded featurizers
        config = {
            "method_name": "submission",
            "evaluation_batch_size": 1024,
            "output_scores": True,
            "check_raw": True
        }
        
        experiment = PatchAttentionHeads(
            pipeline=pipeline,
            causal_model=causal_model,
            layer_head_list=attention_heads,
            token_positions=token_positions,
            checker=eval_checker,
            featurizers=featurizers,
            config=config
        )
        
        # Run evaluation on test data
        print("Running attention head evaluation...")
        test_data = {k: v for k, v in filtered_datasets.items() if "test" in k}
        
        # Since we already loaded the appropriate data based on flags, 
        # we just need to filter by public/private if both are loaded
        if private_data and not public_data:
            # Only keep private test data
            test_data = {k: v for k, v in test_data.items() if "private" in k}
        elif public_data and not private_data:
            # Only keep public test data (this is the default case)
            test_data = {k: v for k, v in test_data.items() if "private" not in k}
        # If both flags are True, keep all test data
        
        experiment.perform_interventions(
            test_data, 
            verbose=True, 
            target_variables_list=[[variable]], 
            save_dir=task_folder_path
        )
        
        print(f"Successfully evaluated IOI submission {folder_name}")
        return True
        
    except Exception as e:
        print(f"ERROR evaluating IOI submission {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Evaluate IOI submissions for attention head tasks")
    parser.add_argument("--submission_folder", required=True, 
                       help="Path to IOI submission folder containing task subfolders")
    parser.add_argument("--private_data", action="store_true", default=True,
                       help="Evaluate on private test data (default: False)")
    parser.add_argument("--public_data", action="store_true", default=False,
                       help="Evaluate on public test data (default: True)")
    parser.add_argument("--specific_task", type=str, default=None,
                       help="Evaluate only a specific task folder")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run a quick test with limited data (for debugging purposes)")
    
    args = parser.parse_args()
    
    submission_path = os.path.abspath(args.submission_folder)
    
    if not os.path.exists(submission_path):
        print(f"ERROR: Submission folder does not exist: {submission_path}")
        return 1
    
    print(f"Evaluating IOI submissions in: {submission_path}")
    print(f"Private data: {args.private_data}")
    print(f"Public data: {args.public_data}")
    
    # Find all IOI task folders
    task_folders = []
    for item in os.listdir(submission_path):
        item_path = os.path.join(submission_path, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Skip non-task folders (like __pycache__)
            if item in ['__pycache__', '.git']:
                continue
            task, model, variable = parse_ioi_submission_folder(item)
            if all([task, model, variable]):
                if args.specific_task is None or item == args.specific_task:
                    task_folders.append(item_path)
    
    if not task_folders:
        print("ERROR: No valid IOI task folders found in submission directory")
        return 1
    
    print(f"Found {len(task_folders)} IOI task folders to evaluate")
    
    # Evaluate each task folder
    successful = 0
    total = len(task_folders)
    
    for task_folder_path in task_folders:
        if evaluate_ioi_submission_task(task_folder_path, submission_path, args.private_data, args.public_data, quick_test=args.quick_test):
            successful += 1
            continue
        for subfolder in os.listdir(task_folder_path):
            subfolder_path = os.path.join(task_folder_path, subfolder)
            if evaluate_ioi_submission_task(subfolder_path, submission_path, args.private_data, args.public_data, quick_test=args.quick_test):
                successful += 1
    
    print(f"\n{'='*60}")
    print(f"IOI EVALUATION COMPLETE")
    print(f"Successfully evaluated: {successful}/{total} submissions")
    print(f"{'='*60}")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())