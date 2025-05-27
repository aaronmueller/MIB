
import inspect
import argparse
import importlib.util
import ast
import os
import torch
from CausalAbstraction.model_units.model_units import Featurizer, SubspaceFeaturizerModule, SubspaceInverseFeaturizerModule, SAEFeaturizerModule, SAEInverseFeaturizerModule, IdentityFeaturizerModule, IdentityInverseFeaturizerModule
from typing import Optional, Tuple, Union
from collections import defaultdict


TASKS = set(["ioi_task", "4_answer_MCQA", "ARC_easy", "arithmetic", "ravel_task"])
TASK_VARIABLES = {"ioi_task": ["output_token", "output_position"],
                  "4_answer_MCQA": ["answer_pointer", "answer"],
                  "arc": ["answer_pointer", "answer"],
                  "arithmetic": ["ones_carry"],
                  "ravel_task": ["Country", "Continent", "Language"]}
MODELS = set(["GPT2LMHeadModel", "Qwen2ForCausalLM", "Gemma2ForCausalLM", "LlamaForCausalLM"])
# create pairs of valid task/model combinations
VALID_TASK_MODELS = set([("ioi_task", "GPT2LMHeadModel"),
                         ("ioi_task", "Qwen2ForCausalLM"),
                         ("ioi_task", "Gemma2ForCausalLM"),
                         ("ioi_task", "LlamaForCausalLM"),
                         ("4_answer_MCQA", "Qwen2ForCausalLM"),
                         ("4_answer_MCQA", "Gemma2ForCausalLM"),
                         ("4_answer_MCQA", "LlamaForCausalLM"),
                         ("ARC_easy", "Gemma2ForCausalLM"),
                         ("ARC_easy", "LlamaForCausalLM"),
                         ("arithmetic", "Gemma2ForCausalLM"),
                         ("arithmetic", "LlamaForCausalLM"),
                         ("ravel_task", "Gemma2ForCausalLM"),
                         ("ravel_task", "LlamaForCausalLM")])

class FeaturizerValidator:
    def __init__(self, base_featurizer_class):
        self.base_featurizer_class = base_featurizer_class
        self.featurizer_class_name = None
        
        # torch.nn.Module
        self.module_value, self.module_attr = "torch", "Module"
        self.featurizer_module_class_name_1 = None
        self.featurizer_module_class_name_2 = None


    def find_featurizer_subclass(self, module_path: str) -> Tuple[bool, Union[str, None]]:
        """
        Finds the first class in the module that inherits from Featurizer.
        
        Args:
            module_path: Path to the uploaded Python file
                
        Returns:
            Tuple of (success, class_name, message)
        """
        # First try with AST for safety
        try:
            with open(module_path, 'r') as file:
                tree = ast.parse(file.read(), filename=module_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == self.base_featurizer_class.__name__:
                            return True, node.name, f"Found class '{node.name}' that inherits from {self.base_featurizer_class.__name__}"
            
            return False, None, f"No class inheriting from {self.base_featurizer_class.__name__} found"
        
        except Exception as e:
            return False, None, f"Error during static analysis: {str(e)}"


    def find_featurizer_module_classes(self, module_path: str) -> Tuple[bool, Union[str, None]]:
        try:
            with open(module_path, 'r') as file:
                tree = ast.parse(file.read(), filename=module_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if (isinstance(base, ast.Attribute) and base.attr == self.module_attr):
                            if self.featurizer_module_class_name_1 is None:
                                self.featurizer_module_class_name_1 = node.name
                            else:
                                self.featurizer_module_class_name_2 = node.name
                                return True, f"Found two featurizer modules: {self.featurizer_module_class_name_1}, {self.featurizer_module_class_name_2}"
            
            if self.featurizer_module_class_name_1:
                return True, f"Found one featurizer module: {self.featurizer_module_class_name_1}"
            return False, f"Found no featurizer modules."
        
        except Exception as e:
            return False, f"Error during static analysis: {e}"
    

    def validate_uploaded_module(self, module_path: str) -> Tuple[bool, str]:
        """
        Validates an uploaded module to ensure it properly extends the Featurizer class.
        
        Args:
            module_path: Path to the uploaded Python file
            class_name: Name of the class to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        # First, find the name of the featurizer class we're verifying
        found, class_name, message = self.find_featurizer_subclass(module_path)
        if not found:
            return False, message
        else:
            print("Verified featurizer subclass.")

        # Second, find the name of the featurizer and inverse featurizer modules
        modules_found, modules_message = self.find_featurizer_module_classes(module_path)
        if not modules_found:
            return False, modules_message
        else:
            print(f"Verified featurizer module(s): {modules_message}")

        # Then, perform static code analysis on the featurizer class for basic safety
        inheritance_check, ast_message = self._verify_inheritance_with_ast(module_path, class_name)
        if not inheritance_check:
            return False, ast_message
            
        # Then, try to load and validate the featurizer class
        return self._verify_inheritance_with_import(module_path, class_name)

        # TODO: try directly loading featurizer module and inverse featurizer module?
    

    def _verify_inheritance_with_ast(self, module_path: str, class_name: str) -> Tuple[bool, str]:
        """Verify inheritance using AST without executing code"""
        try:
            with open(module_path, 'r') as file:
                tree = ast.parse(file.read(), filename=module_path)
            
            # Look for class definitions that match the target class name
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Check if any base class name matches 'Featurizer'
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == self.base_featurizer_class.__name__:
                            return True, "Static analysis indicates proper inheritance"
                        
                    return False, f"Class '{class_name}' does not appear to inherit from {self.base_featurizer_class.__name__}"
            
            return False, f"Class '{class_name}' not found in the uploaded module"
            
        except Exception as e:
            return False, f"Error during static analysis: {str(e)}"
    

    def _verify_inheritance_with_import(self, module_path: str, class_name: str) -> Tuple[bool, str]:
        """Safely import the module and verify inheritance using Python's introspection"""
        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location("uploaded_module", module_path)
            if spec is None or spec.loader is None:
                return False, "Could not load the module specification"
                
            uploaded_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(uploaded_module)
            
            # Get the class from the module
            if not hasattr(uploaded_module, class_name):
                return False, f"Class '{class_name}' not found in the uploaded module"
                
            uploaded_class = getattr(uploaded_module, class_name)
            
            # Check if it's a proper subclass
            if not inspect.isclass(uploaded_class):
                return False, f"'{class_name}' is not a class"
                
            if not issubclass(uploaded_class, self.base_featurizer_class):
                return False, f"'{class_name}' does not inherit from {self.base_featurizer_class.__name__}"
                
            # Optional: Check method resolution order
            mro = inspect.getmro(uploaded_class)
            if self.base_featurizer_class not in mro:
                return False, f"{self.base_featurizer_class.__name__} not in the method resolution order"
                
            return True, f"Class '{class_name}' properly extends {self.base_featurizer_class.__name__}"
            
        except Exception as e:
            return False, f"Error during dynamic validation: {str(e)}"

def verify_directory(path: str, modules) -> Tuple[str, Union[str, None]]:
    count = 0
    for filename in os.listdir(path):
        count += 1
        
        valid_index = False
        valid_featurizer = False
        valid_inverse_featurizer = False

        prefix = "_".join(filename.split("_")[:-1])
        filepath = os.path.join(path, prefix)
        if os.path.exists(filepath+ "_indices"):
            with open(filepath+ "_indices", 'r') as file:
                # TODO: make sure this actually works
                s = file.readlines()[0]
                is_list = s == "null" or isinstance(ast.literal_eval(s), list)
                if is_list:
                    valid_index = True
        if os.path.exists(filepath+ "_featurizer"):
            with open(filepath + "_featurizer", 'r'):
                # Try to load the .pt file
                checkpoint = torch.load(filepath+ "_featurizer", map_location=torch.device('cpu'))
                
                # Check if this is a checkpoint with model_info
                if 'model_info' in checkpoint and 'state_dict' in checkpoint:
                    model_info = checkpoint['model_info']
                    # Check file endings or model_info to determine type
                    if ('featurizer_class' in model_info and \
                        model_info['featurizer_class'] in modules
                    ):
                        valid_featurizer = True
        if os.path.exists(filepath + "_inverse_featurizer"):
            with open(filepath + "_inverse_featurizer", 'r') as file:
                # Try to load the .pt file
                checkpoint = torch.load(filepath+ "_inverse_featurizer", map_location=torch.device('cpu'))
                
                # Check if this is a checkpoint with model_info
                if 'model_info' in checkpoint and 'state_dict' in checkpoint:
                    model_info = checkpoint['model_info']                  
                    
                    if ('inverse_featurizer_class' in model_info and 
                        model_info['inverse_featurizer_class'] in modules
                    ):
                        valid_inverse_featurizer = True
        if valid_index and valid_featurizer and valid_inverse_featurizer:
            return True, "Found valid featurizer modules and indices!"
        if valid_index and not valid_featurizer and not valid_inverse_featurizer:
            return True, "Found valid indices but no featurizer modules!"
    return False, "No valid featurizer/inverse featurizer/indices triplet files."



def validate_token_positions(filepath: str) -> Tuple[bool, str]:
    """
    Validate that a python file contains a get_token_positions function that returns 
    a list of TokenPosition objects.
    """
    try:
        with open(filepath, 'r') as file:
            tree = ast.parse(file.read(), filename=filepath)
    except Exception as e:
        return False, f"Error parsing file: {str(e)}"
        
    # Find the get_token_positions function
    function_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_token_positions":
            function_node = node
            break
    
    if function_node is None:
        return False, "No get_token_positions function found"
    
    # Analyze return statements in the function
    return_nodes = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return) and node.value is not None:
            return_nodes.append(node)
    
    if not return_nodes:
        return False, "get_token_positions function has no return statements"
    
    # Check each return statement
    for return_node in return_nodes:
        # Case 1: Direct list return like: return [TokenPosition(...), TokenPosition(...)]
        if isinstance(return_node.value, ast.List):
            if _check_list_contains_token_position(return_node.value):
                return True, "Found valid get_token_positions function returning list of TokenPosition objects"
            else:
                return False, "get_token_positions returns a list but it doesn't contain TokenPosition objects"
        
        # Case 2: Variable return like: return token_positions
        elif isinstance(return_node.value, ast.Name):
            var_name = return_node.value.id
            
            # Look for assignments to this variable in the function
            if _check_variable_assigned_token_position_list(function_node, var_name):
                return True, "Found valid get_token_positions function returning list of TokenPosition objects"
            else:
                return False, "get_token_positions returns a list but it doesn't contain TokenPosition objects"
        
        else:
            return False, "get_token_positions function does not return a list"
    
    return False, "Could not determine return type of get_token_positions function"


def _check_list_contains_token_position(list_node: ast.List) -> bool:
    """Check if an AST List node contains TokenPosition constructor calls."""
    for element in list_node.elts:
        if isinstance(element, ast.Call):
            # Check for TokenPosition constructor call
            if isinstance(element.func, ast.Name) and element.func.id == "TokenPosition":
                return True
    return False


def _check_variable_assigned_token_position_list(function_node: ast.FunctionDef, var_name: str) -> bool:
    """Check if a variable is assigned a list containing TokenPosition objects within the function."""
    for node in ast.walk(function_node):
        if isinstance(node, ast.Assign):
            # Check each assignment target
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    # Check if the assigned value is a list with TokenPosition objects
                    if isinstance(node.value, ast.List):
                        return _check_list_contains_token_position(node.value)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=str)
    args = parser.parse_args()

    errors, warnings = [], []
    
    file_exists = False
    token_verifies = False
    message = ""
    #Verify token positions
    for filename in os.listdir(args.submission_dir):
        if filename.endswith(".py"):
            file_exists = True
            filepath = os.path.join(args.submission_dir, filename)
            class_verified, message = validate_token_positions(filepath)
            if class_verified:
                token_verifies = True
                break

    if token_verifies:
        print("Token position function verification succeeded.")
    if not token_verifies and file_exists:
        errors.append(f"Token position function verification failed: {message}")


    #Verify featurizer class
    featurizer_validator = None
    file_exists = False
    class_verified = False
    message = ""
    for filename in os.listdir(args.submission_dir):
        if filename.endswith(".py"):
            file_exists = True
            filepath = os.path.join(args.submission_dir, filename)
            featurizer_validator = FeaturizerValidator(Featurizer)
            class_verified, message = featurizer_validator.validate_uploaded_module(filepath)
            if class_verified:
                break
    
    if class_verified:
        print("Featurizer script verification succeeded.")
    if not class_verified and file_exists:
        errors.append(f"Featurizer script verification failed: {message}")
    
    modules = [SubspaceFeaturizerModule.__name__,
              SubspaceInverseFeaturizerModule.__name__,
              SAEFeaturizerModule.__name__,
              SAEInverseFeaturizerModule.__name__,
              IdentityFeaturizerModule.__name__,
              IdentityInverseFeaturizerModule.__name__,
              ]
    if featurizer_validator is not None:
        modules.append(featurizer_validator.featurizer_module_class_name_1)
        modules.append(featurizer_validator.featurizer_module_class_name_2)
    

    found_triplets = set()
    
    # Verify saved featurizers and indices
    for dirname in os.listdir(args.submission_dir):
        dirpath = os.path.join(args.submission_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        # Parse directory name in format: TASK_MODEL_VARIABLE
        # Split by underscore and try to match known components
        parts = dirname.split('_')
        
        # Find task name (can be multi-part like "4_answer_MCQA")
        curr_task = None
        task_end_idx = 0
        for task in TASKS:
            task_parts = task.split('_')
            if len(parts) >= len(task_parts):
                if parts[:len(task_parts)] == task_parts:
                    curr_task = task
                    task_end_idx = len(task_parts)
                    break
        
        if curr_task is None:
            warnings.append(f"Skipped directory `{dirname}`: Could not identify valid task name")
            continue
        
        # Find model name in remaining parts
        remaining_parts = parts[task_end_idx:]
        curr_model = None
        model_end_idx = 0
        
        for model in MODELS:
            model_parts = model.split('_') if '_' in model else [model]
            if len(remaining_parts) >= len(model_parts):
                # Check if model parts match at any position in remaining parts
                for start_pos in range(len(remaining_parts) - len(model_parts) + 1):
                    if remaining_parts[start_pos:start_pos + len(model_parts)] == model_parts:
                        curr_model = model
                        model_end_idx = task_end_idx + start_pos + len(model_parts)
                        break
                if curr_model:
                    break
        
        if curr_model is None:
            warnings.append(f"Skipped directory `{dirname}`: Could not identify valid model name")
            continue
        
        # Find variable name in remaining parts
        variable_parts = parts[model_end_idx:]
        curr_variable = None
        
        if curr_task in TASK_VARIABLES:
            for variable in TASK_VARIABLES[curr_task]:
                variable_parts_expected = variable.split('_') if '_' in variable else [variable]
                if variable_parts == variable_parts_expected:
                    curr_variable = variable
                    break
        
        if curr_variable is None:
            warnings.append(f"Skipped directory `{dirname}`: Could not identify valid variable name for task {curr_task}")
            continue
        
        # Check if this task/model combination is valid
        if (curr_task, curr_model) not in VALID_TASK_MODELS:
            warnings.append(f"Skipped directory `{dirname}`: Task {curr_task} and model {curr_model} not in valid combinations")
            continue
        
        # Verify the directory contains valid featurizer files
        contains_valid_triplet, message = verify_directory(dirpath, modules)
        if contains_valid_triplet:
            found_triplets.add((curr_task, curr_model, curr_variable))
        else:
            warnings.append(f"Couldn't find a valid featurizer/inverse featurizer/indices triplet in {dirname}: {message}")
    
    # TODO: If we expect token position function(s) inside the task/model/counterfactual folder, verify it here
    
    out_str = ""
    if len(errors) > 0:
        out_str += "ERRORS:\n" + "\n".join(errors)
    else:
        out_str += "Valid submission."
    if len(warnings) > 0:
        out_str += "\n\n------\n\n"
        out_str += "WARNINGS:\n" + "\n".join(warnings)
    if len(errors) == 0 and len(warnings) == 0:
        out_str = f"Perfect submission! No errors or warnings. Found {len(found_triplets)} valid triplet(s)."

    print(out_str)

