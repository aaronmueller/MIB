"""
Token position definitions for MCQA task submission.
This file provides token position functions that identify key tokens in MCQA prompts.
"""

import re
from CausalAbstraction.model_units.LM_units import TokenPosition


def get_last_token_index(prompt, pipeline):
    """
    Get the index of the last token in the prompt.
    
    Args:
        prompt (str): The input prompt
        pipeline: The tokenizer pipeline
    
    Returns:
        list[int]: List containing the index of the last token
    """
    input_ids = list(pipeline.load(prompt)["input_ids"][0])
    return [len(input_ids) - 1]


def get_correct_symbol_index(prompt, pipeline, task):
    """
    Find the index of the correct answer symbol in the prompt.
    
    Args:
        prompt (str): The prompt text
        pipeline: The tokenizer pipeline
        task: The task object containing causal model
        
    Returns:
        list[int]: List containing the index of the correct answer symbol token
    """
    # Run the model to get the answer position
    output = task.causal_model.run_forward(task.input_loader(prompt))
    pointer = output["answer_pointer"]
    correct_symbol = output[f"symbol{pointer}"]
    
    # Find all single uppercase letters in the prompt
    matches = list(re.finditer(r"\b[A-Z]\b", prompt))
    
    # Find the match corresponding to our correct symbol
    symbol_match = None
    for match in matches:
        if prompt[match.start():match.end()] == correct_symbol:
            symbol_match = match
            break
            
    if not symbol_match:
        raise ValueError(f"Could not find correct symbol {correct_symbol} in prompt: {prompt}")
    
    # Get the substring up to the symbol match end
    substring = prompt[:symbol_match.end()]
    tokenized_substring = list(pipeline.load(substring)["input_ids"][0])
    
    # The symbol token will be at the end of the substring
    return [len(tokenized_substring) - 1]


def get_token_positions(pipeline, task):
    """
    Get token positions for the MCQA task.
    
    This function identifies key token positions in MCQA prompts:
    - correct_symbol: The position of the correct answer symbol (A, B, C, or D)
    - last_token: The position of the last token in the prompt
    
    Args:
        pipeline: The language model pipeline with tokenizer
        task: The MCQA task object
        
    Returns:
        list[TokenPosition]: List of TokenPosition objects for intervention experiments
    """
    # Create TokenPosition objects
    token_positions = [
        TokenPosition(
            lambda x: get_correct_symbol_index(x, pipeline, task), 
            pipeline, 
            id="correct_symbol"
        ),
        TokenPosition(
            lambda x: get_last_token_index(x, pipeline), 
            pipeline, 
            id="last_token"
        )
    ]
    return token_positions