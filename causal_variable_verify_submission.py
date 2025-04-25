
import inspect
import argparse
import importlib.util
import ast
import os
import torch
from model_units.model_units import Featurizer, SubspaceFeaturizerModule, SubspaceInverseFeaturizerModule, SAEFeaturizerModule, SAEInverseFeaturizerModule, IdentityFeaturizerModule, IdentityInverseFeaturizerModule
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
    # determine whether a python file contains a function called get_token_positions
    with open(filepath, 'r') as file:
        tree = ast.parse(file.read(), filename=filepath)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "get_token_positions":
                # Check if the function has a decorator
                if any(isinstance(decorator, ast.Name) and decorator.id == "token_position" for decorator in node.decorator_list):
                    return True, "Found valid token position function"
                else:
                    return False, "Token position function does not have the correct decorator"
    return False, "No token position function found"

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

        curr_task = None
        for task in TASKS:
            if dirname.startswith(task) or f"_{task}" in dirname:
                curr_task = task
                break
        if curr_task is None:
            warnings.append(f"Skipped directory `{dirname}`: Not a valid task name")
            continue
        for dirname2 in os.listdir(dirpath):
            dirpath2 = os.path.join(dirpath, dirname2)
            curr_model = None
            for model in MODELS:
                if dirname2.startswith(model) or f"_{model}" in dirname2:
                    curr_model = model
                    continue
            if curr_model is None:
                warnings.append(f"Skipped directory `{dirname2}`: Not a valid model name")
            if (curr_task, curr_model) not in VALID_TASK_MODELS:
                warnings.append(f"Skipped directory `{dirname2}`: Task {curr_task} and model {curr_model} not in valid combinations.")
                continue
            for dirname3 in os.listdir(dirpath2):
                dirpath3 = os.path.join(dirpath2, dirname3)
                curr_variable = None
                for variable in TASK_VARIABLES[curr_task]:
                    if dirname3 == variable:
                        curr_variable = variable
                        break    
                if curr_variable is None:
                    #warning that variable invalid for task
                    warnings.append(f"Skipped directory `{dirname}`: Not a valid variable name for task {curr_task}")
                    continue
                contains_valid_triplet, message = verify_directory(dirpath3, modules)
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

