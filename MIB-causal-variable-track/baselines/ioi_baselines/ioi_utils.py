import torch
import gc
import numpy as np
from experiments.pyvene_core import _prepare_intervenable_inputs

def clear_memory():
    """Clear memory between experiments to prevent OOM errors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_diff(logits, params, pipeline):
    """
    Compute the difference in logit scores between two tokens.
    
    Args:
        logits: Tensor containing logit scores for tokens
        params: Dictionary containing 'name_A', 'name_B', and 'name_C'
        pipeline: Pipeline object with tokenizer
        
    Returns:
        Tensor: logit_IO - logit_S
    """
    # Extract names from params
    name_A = params["name_A"]
    name_B = params["name_B"]
    name_C = params["name_C"]
    
    if not isinstance(name_A, list):
        name_A = [name_A]
    if not isinstance(name_B, list):
        name_B = [name_B]
    if not isinstance(name_C, list):
        name_C = [name_C]

    token_id_A = [pipeline.tokenizer.encode(A, add_special_tokens=False)[0] for A in name_A]
    token_id_B = [pipeline.tokenizer.encode(B, add_special_tokens=False)[0] for B in name_B]
    token_id_C = [pipeline.tokenizer.encode(C, add_special_tokens=False)[0] for C in name_C]

    token_id_IO, token_id_S = [], []
    for i in range(len(token_id_A)):
        if token_id_A[i] == token_id_C[i]:
            token_id_S.append(token_id_A[i])
            token_id_IO.append(token_id_B[i])
        elif token_id_B[i] == token_id_C[i]:
            token_id_S.append(token_id_B[i])
            token_id_IO.append(token_id_A[i])
    
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # Get the logit scores for both tokens
    if len(logits.shape) == 3:
        logits = logits.squeeze(1)
    if len(logits.shape) == 2:
        # Create batch indices
        batch_indices = torch.arange(logits.shape[0])
        
        # Extract specific logits using batch indices
        logit_S = logits[batch_indices, token_id_S]
        logit_IO = logits[batch_indices, token_id_IO]
    elif len(logits.shape) == 1:
        logit_S = logits[token_id_S[0]]
        logit_IO = logits[token_id_IO[0]]
    
    return logit_IO - logit_S

def checker(logits, params, pipeline):
    """
    Compute the squared error between the actual logit difference and the target logit difference.
    
    Args:
        logits: Tensor containing logit scores for tokens
        params: Dictionary containing 'name_A', 'name_B', 'name_C', and 'logit_diff'
        pipeline: Pipeline object with tokenizer
    
    Returns:
        Tensor: Squared error between the computed logit difference and the target logit difference
    """
    if isinstance(logits, list):
        logits = logits[0]

    target_diff = params["logit_diff"]
    actual_diff = log_diff(logits, params, pipeline)
    if isinstance(target_diff, torch.Tensor):
        target_diff = target_diff.to(actual_diff.device).to(actual_diff.dtype)
    
    squared_error = (actual_diff - target_diff) ** 2
    
    return squared_error

def filter_checker(output_text, expected):
    """
    Simple checker for filtering that just checks if the expected token appears in the output.
    Used only for dataset filtering, not for the actual experiments.
    
    Args:
        output_text (str): The model's output text
        expected (str): The expected output
        
    Returns:
        bool: True if expected token appears in output
    """
    return expected in output_text

def custom_loss(logits, params, pipeline):
    """
    Average loss function for training that handles both single examples and batches.
    
    Args:
        logits: Model logits
        params: Parameters (can be single dict or list of dicts for batch)
        pipeline: Pipeline object with tokenizer
        
    Returns:
        Tensor: Average loss
    """
    if isinstance(params, list):
        # params is a list of dicts, one for each example in the batch
        total_loss = 0
        for i, param_dict in enumerate(params):
            # Extract the i-th logits for this example
            example_logits = logits[i] if logits.dim() > 1 else logits
            loss = checker(example_logits, param_dict, pipeline)
            total_loss += loss
        return total_loss / len(params)
    else:
        # Single example case (original behavior)
        return checker(logits, params, pipeline).mean()

def ioi_loss_and_metric_fn(pipeline, intervenable_model, batch, model_units_list):
    """
    Calculate loss and evaluation metrics for IOI interventions.
    
    Uses the checker function as a metric (squared error) and custom_loss 
    as the loss function for training.
    
    Args:
        pipeline: Pipeline object
        intervenable_model: The intervenable model
        batch: Batch of data
        model_units_list: List of model units
        
    Returns:
        tuple: (loss, eval_metrics, logging_info)
    """
    # 1. Prepare intervenable inputs
    batched_base, batched_counterfactuals, inv_locations, feature_indices = _prepare_intervenable_inputs(
        pipeline, batch, model_units_list)
    
    # 2. Run the intervenable model to get logits
    _, counterfactual_logits = intervenable_model(
        batched_base, batched_counterfactuals, 
        unit_locations=inv_locations, 
        subspaces=feature_indices
    )
    
    # 3. Extract the logits (last token position since max_new_tokens=1)
    logits = counterfactual_logits.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
    
    # 4. Get the settings/parameters from the batch
    # These should contain name_A, name_B, name_C, and logit_diff
    settings = batch['setting']  # or batch['label'] depending on how it's structured
    
    # 5. Compute loss using custom_loss
    loss = custom_loss(logits, settings, pipeline)
    
    # 6. Compute metrics using checker (squared errors)
    squared_errors = []
    for i in range(len(logits)):
        error = checker(logits[i], settings[i], pipeline)
        squared_errors.append(error.item())
    
    eval_metrics = {
        "mse": np.mean(squared_errors),  # Mean squared error
        "rmse": np.sqrt(np.mean(squared_errors))  # Root mean squared error
    }
    
    # 7. Prepare logging info
    logging_info = {
        "batch_size": len(batch['input']),
        "avg_logit_diff": np.mean([s['logit_diff'] for s in settings])
    }
    
    return loss, eval_metrics, logging_info

def get_model_config(model_name):
    """
    Get model configuration based on model name.
    
    Args:
        model_name (str): One of "gpt2", "qwen", "llama", "gemma"
        
    Returns:
        dict: Configuration dictionary with model_path, batch_size, and special_config
    """
    model_configs = {
        "gpt2": {
            "model_path": "openai-community/gpt2",
            "batch_size": 1024,
            "special_config": True  # Needs special GPT2Config
        },
        "qwen": {
            "model_path": "Qwen/Qwen2.5-0.5B",
            "batch_size": 256,
            "special_config": False
        },
        "llama": {
            "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "batch_size": 256,
            "special_config": False
        },
        "gemma": {
            "model_path": "google/gemma-2-2b",
            "batch_size": 256,
            "special_config": False
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}. Choose from {list(model_configs.keys())}")
    
    return model_configs[model_name]

def setup_pipeline(model_name, device, eval_batch_size=None):
    """
    Set up the pipeline for a given model.
    
    Args:
        model_name (str): One of "gpt2", "qwen", "llama", "gemma"
        device (str): Device to use
        eval_batch_size (int, optional): Override default batch size
        
    Returns:
        tuple: (pipeline, batch_size)
    """
    from neural.pipeline import LMPipeline
    
    config = get_model_config(model_name)
    model_path = config["model_path"]
    batch_size = eval_batch_size if eval_batch_size else config["batch_size"]
    
    if config["special_config"]:
        # Special configuration for GPT2
        from transformers import GPT2Config
        gpt_config = GPT2Config.from_pretrained(model_path)
        pipeline = LMPipeline(model_path, max_new_tokens=1, device=device, dtype=torch.float32, 
                            max_length=32, logit_labels=True, position_ids=True, config=gpt_config)
    else:
        pipeline = LMPipeline(model_path, max_new_tokens=1, device=device, dtype=torch.float16,
                              max_length=32, logit_labels=True)
        # pyvene reads config.head_dim for Qwen2 attention heads, which recent
        # transformers versions no longer set on Qwen2Config.
        model_config = pipeline.model.config
        if not hasattr(model_config, "head_dim"):
            model_config.head_dim = model_config.hidden_size // model_config.num_attention_heads

    pipeline.tokenizer.padding_side = "left"
    
    return pipeline, batch_size