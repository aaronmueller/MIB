import os
import json
from typing import Dict, Tuple, List, Any
import statistics


def analyze_json(json_data: Dict, private:bool) -> Tuple[float, float]:
    """
    Analyze a single JSON file:
    1. Average model units across datasets
    2. Select highest accuracy position for each layer
    3. Compute average accuracy across layers and highest accuracy across layers
    
    Args:
        json_data: Loaded JSON data
        
    Returns:
        Tuple of (average_accuracy_across_layers, highest_accuracy_across_layers)
    """
    datasets = json_data.get("dataset", {})
    
    # Step 1: Combine all datasets and compute average accuracy per model unit
    all_units = {}
    
    for dataset_name, dataset_data in datasets.items():
        if private and "private" not in dataset_name:
            continue
        elif not private and "private" in dataset_name:
            continue
        model_units = dataset_data.get("model_unit", {})
        
        for unit_id, unit_data in model_units.items():
            if unit_id not in all_units:
                all_units[unit_id] = {"accuracies": [], "metadata": unit_data.get("metadata", {})}
            
            all_units[unit_id]["accuracies"].append(unit_data.get("accuracy", 0))
    
    # Calculate average accuracy for each unit
    for unit_id, unit_info in all_units.items():
        unit_info["avg_accuracy"] = statistics.mean(unit_info["accuracies"]) if unit_info["accuracies"] else 0
    
    # Step 2: Group by layer and select highest accuracy for each layer
    layers = {}
    for unit_id, unit_info in all_units.items():
        layer = unit_info["metadata"].get("layer", None)
        if layer is not None:
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(unit_info["avg_accuracy"])
    
    # Get the highest accuracy for each layer
    highest_per_layer = {layer: max(accuracies) for layer, accuracies in layers.items() if accuracies}
    
    # Step 3: Compute stats across layers
    if highest_per_layer:
        average_accuracy = statistics.mean(highest_per_layer.values())
        highest_accuracy = max(highest_per_layer.values())
    else:
        average_accuracy = 0
        highest_accuracy = 0
    
    return average_accuracy, highest_accuracy


def process_folder(folder_path: str, private:bool) -> Dict[Tuple, Dict[str, float]]:
    """
    Process all JSON files in a folder and aggregate results
    
    Args:
        folder_path: Path to the folder containing JSON files
        
    Returns:
        Dictionary mapping (method_name, model_name, task_name, target_variables) 
        to stats (average and highest accuracy)
    """
    results = {}
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                
            # Extract key components
            method_name = json_data.get("method_name", "unknown")
            model_name = json_data.get("model_name", "unknown")
            task_name = json_data.get("task_name", "unknown")
            target_variables = json_data.get("target_variables", "unknown")
            
            # Create the key
            key = (method_name, model_name, task_name, target_variables)
            
            # Analyze the data
            avg_acc, high_acc = analyze_json(json_data, private)
            
            # Store the results
            results[key] = {
                "average_accuracy": avg_acc,
                "highest_accuracy": high_acc
            }
            results[key]["method_name"] = method_name
            results[key]["model_name"] = model_name
            results[key]["task_name"] = task_name
            results[key]["target_variables"] = target_variables
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return results

def print_latex_table(results, print_highs=False, merge_variables=True):
    """
    Print LaTeX code for a table with the provided results.
    
    Args:
        results: Dictionary with (method_name, model_name, task_name, target_variables) 
                keys and accuracy values
    """
    # Function to escape special LaTeX characters
    def escape_latex(text):
        # Replace special characters with their LaTeX escape sequences
        replacements = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
        }
        for char, replacement in replacements.items():
            if isinstance(text, str):
                text = text.replace(char, replacement)
        return text
    
    # Extract unique methods, models, and tasks
    methods = set()
    tasks = set()
    models = {}  # Changed to dictionary mapping from task_name to sets
    variables = {}
    
    # Track accuracies by (method, model, task)
    averages = {}
    accuracies = {}
    highs = {}
    # For averaging across target_variables
    accuracy_sums = {}
    count_by_key = {}
    
    # Process results to extract components and organize data
    for key, value in results.items():
        if isinstance(key, str):
            # Handle string keys (converted for JSON serialization)
            import ast
            key = ast.literal_eval(key)
        
        method_name, model_name, task_name, target_variables = key
        
        methods.add(method_name)
        tasks.add(task_name)
        
        # Add models as a dictionary mapping from task_name to sets
        if task_name not in models:
            models[task_name] = set()
        models[task_name].add(model_name)
        
        if task_name not in variables:
            variables[task_name] = set()
        variables[task_name].add(target_variables)
        
        # Group by (method, model, task) to average across target_variables
        group_key = (method_name, model_name, task_name)
        
        if group_key not in accuracy_sums:
            accuracy_sums[group_key] = 0
            count_by_key[group_key] = 0
            
        accuracy_sums[group_key] += value["average_accuracy"]
        count_by_key[group_key] += 1
        accuracies[(method_name, model_name, task_name, target_variables)] = value["average_accuracy"]

        if group_key not in highs:
            highs[group_key] = {target_variables: value["highest_accuracy"]}
        else:
            highs[group_key][target_variables] = value["highest_accuracy"]

    
    # Calculate averages across target_variables
    for group_key, total in accuracy_sums.items():
        count = count_by_key[group_key]
        if count > 0:
            averages[group_key] = total / count
    
    # Sort sets for consistent display
    methods = sorted(methods)
    tasks = sorted(tasks)
    models = {k:sorted(v) for k,v in models.items()}  # Sort models for each task
    variables = {k:sorted(v) for k,v in variables.items()}

    methods_display = {}
    for method in methods:
        if "full_vector" in method:
            methods_display[method] = "Full Vector"
        elif "+" in method:
            #remove text before the plus sign
            methods_display[method] = "+" + method.split("+")[1]
        else:
            methods_display[method] = method
    
    # Apply replacements for display
    models_display = {}
    for task in tasks:
        for model in models[task]:
            if "LlamaForCausalLM" in model:
                models_display[model] = "Llama-3.1"
            elif "Gemma2ForCausalLM" in model:
                models_display[model] = "Gemma-2"
            elif "Qwen" in model:
                models_display[model] = "Qwen-2.5"
            else:
                models_display[model] = "GPT-2"
            
    
    tasks_display = {}
    for task in tasks:
        if "4_answer_MCQA" in task:
            tasks_display[task] = "MCQA"
        elif "ARC_easy" in task:
            tasks_display[task] = "ARC (Easy)"
        elif "null" in task:
            tasks_display[task] = "Arithmetic (+)"
        elif "ioi" in task:
            tasks_display[task] = "IOI"
        else:
            tasks_display[task] = task
    
    variables_display = {}
    for task_name in variables:
        for variable in variables[task_name]:
            if "answer_pointer" in variable:
                variables_display[variable] = "\\order"
            elif "answer" in variable:
                variables_display[variable] = "\\answer"
            elif "ones_carry" in variable:
                variables_display[variable] = "\\carry"
            elif "output_position" in variable:
                variables_display[variable] = "\\position"
            elif "output_token" in variable:
                variables_display[variable] = "\\token"
            else:
                variables_display[variable] = variable

    # Begin LaTeX table
    latex_code = [
        "\\begin{table*}[htbp]",
        "  \\centering",
        "  \\begin{tabular}{l" + "".join(["c" * len(models[task])*len(variables[task]) for task in tasks]) + "}",
        "    \\toprule"
    ]
    
    if merge_variables:
        # First row - Task names with multicolumns
        first_row = ["    "]  # Empty first cell
        for task in tasks:
            task_display = escape_latex(tasks_display[task])
            first_row.append(f"\\multicolumn{{{len(models[task])}}}{{c}}{{{task_display}}}")
        latex_code.append(" & ".join(first_row) + " \\\\")
        
        # Add cmidrule for each task
        cmidrule_parts = []
        col_index = 2  # Starting from column 2 (after the Method column)
        for task in tasks:
            col_end = col_index + len(models[task]) - 1
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_index}-{col_end}}}")
            col_index += len(models[task])
        latex_code.append("    " + "".join(cmidrule_parts))
        
        # Second row - Model names
        second_row = ["    Method"]
        for task in tasks:
            for model in models[task]:
                model_display = escape_latex(models_display[model])
                second_row.append(model_display)
        latex_code.append(" & ".join(second_row) + " \\\\")
        latex_code.append("    \\midrule")
        
        # Data rows - Methods and accuracy values
        for method in methods:
            row = [f"    {escape_latex(methods_display[method])}"]
            for task in tasks:
                for model in models[task]:
                    key = (method, model, task)
                    if key in averages:
                        # Round to the nearest digit
                        value = int(round(100*averages[key], 0))
                        row.append(f"{value}")

                        for target_variable in variables[task_name]:
                            if target_variable in highs[key] and print_highs:
                                # Append the highest accuracy
                                row[-1] += f" ({variables_display[target_variable]}={round(100*highs[key][target_variable], 0)})"
                    else:
                        row.append("--")
            latex_code.append(" & ".join(row) + " \\\\")
    else:
        # First row - Task names with multicolumns
        first_row = ["    "]
        for task in tasks:
            task_display = escape_latex(tasks_display[task])
            first_row.append(f"\\multicolumn{{{len(models[task])*len(variables[task])}}}{{c}}{{{task_display}}}")
        latex_code.append(" & ".join(first_row) + " \\\\")
        
        # Add cmidrule for each task
        cmidrule_parts = []
        col_index = 2  # Starting from column 2 (after the Method column)
        for task in tasks:
            col_end = col_index + len(models[task])*len(variables[task]) - 1
            cmidrule_parts.append(f"\\cmidrule(lr){{{col_index}-{col_end}}}")
            col_index += len(models[task])*len(variables[task])
        latex_code.append("    " + "".join(cmidrule_parts))

        # Second row - Model names
        second_row = [""]
        for task in tasks:
            for model in models[task]:
                model_display = escape_latex(models_display[model])
                second_row.append(f"\\multicolumn{{{len(variables[task])}}}{{c}}{{{model_display}}}")
        latex_code.append(" & ".join(second_row) + " \\\\")
        
        # Add cmidrule for each model across tasks
        cmidrule_parts = []
        col_index = 2  # Starting from column 2 (after the Method column)
        for task in tasks:
            for model in models[task]:
                col_end = col_index + len(variables[task]) - 1
                cmidrule_parts.append(f"\\cmidrule(lr){{{col_index}-{col_end}}}")
                col_index += len(variables[task])
        latex_code.append("    " + "".join(cmidrule_parts))

        # Third row - Target variables
        third_row = ["    Method"]
        for task in tasks:
            for model in models[task]:
                for target_variable in variables[task]:
                    variable_display = escape_latex(variables_display[target_variable])
                    third_row.append(variable_display)
        latex_code.append(" & ".join(third_row) + " \\\\")
        latex_code.append("    \\midrule")

        # Data rows - Methods and accuracy values
        for method in methods:
            row = [f"    {escape_latex(methods_display[method])}"]
            for task in tasks:
                for model in models[task]:
                    for target_variable in variables[task]:
                        key = (method, model, task, target_variable)
                        if key in accuracies:
                            # Round to nearest hundredth
                            value = int(round(100*accuracies[key], 0))
                            row.append(f"{value}")

                            if print_highs and (method, model, task) in highs:
                                # Append the highest accuracy
                                row[-1] += f" (\\textbf{{{int(round(100*highs[(method,model, task)][target_variable], 0))}}})"
                        else:
                            row.append("--")
            latex_code.append(" & ".join(row) + " \\\\")
    
    # End LaTeX table
    latex_code.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\caption{Results of different methods across tasks and models.}",
        "  \\label{tab:results}",
        "\\end{table*}"
    ])
    
    # Print LaTeX code
    print("\nLaTeX Table Code:")
    print("\n".join(latex_code))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process JSON result files')
    parser.add_argument('--folder_path', help='Path to the folder containing JSON result files')
    parser.add_argument('--output', '-o', default='aggregated_results.json', 
                       help='Output file path for the aggregated results')
    parser.add_argument('--private', action='store_true', help='Only process private datasets')
    parser.add_argument('--print_highs', action='store_true',
                        help='Print highest accuracies in LaTeX table')
    parser.add_argument('--merge_variables', action='store_true',
                        help='Merge variables in LaTeX table')
    
    args = parser.parse_args()
    
    print(args.folder_path)
    results = process_folder(args.folder_path, args.private)
    
    # Call the new function to print LaTeX table
    print_latex_table(results, args.print_highs, args.merge_variables)
    
    # Convert tuple keys to strings for JSON serialization
    serializable_results = {
        str(key): value
        for key, value in results.items()
    }
    
    with open(args.output, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Processed {len(results)} files. Results saved to {args.output}")
    
    # Print some summary statistics
    if results:
        avg_accuracies = [r["average_accuracy"] for r in results.values() if "average_accuracy" in r]
        high_accuracies = [r["highest_accuracy"] for r in results.values() if "highest_accuracy" in r]
        
        print(f"Overall average accuracy: {statistics.mean(avg_accuracies):.4f}")
        print(f"Overall highest accuracy: {max(high_accuracies):.4f}")

if __name__ == "__main__":
    main()
