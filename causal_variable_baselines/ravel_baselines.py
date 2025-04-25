import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CausalAbstraction.tasks.RAVEL.ravel import get_token_positions, get_task
from CausalAbstraction.experiments.aggregate_experiments import residual_stream_baselines
from CausalAbstraction.pipeline import LMPipeline
import torch
import re
import gc
import random
from copy import deepcopy
from datasets import Dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments with optional flags.")
    parser.add_argument("--skip_gemma", action="store_true", help="Skip running experiments for Gemma model.")
    parser.add_argument("--skip_llama", action="store_true", help="Skip running experiments for Llama model.")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--methods", nargs="+", default=["full_vector", "DAS", "DBM+PCA", "DBM", "DBM+SAE"], help="List of methods to run.")
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    task = get_task(hf=True, size=10)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"

    def checker(output_text, expected):
        if expected is None:
            return True
        if output_text is None:
            return False

        output_clean = re.sub(r'[^\w\s]+', '', output_text.lower()).strip()
        expected_list = [e.strip().lower() for e in expected.split(',')]

        if any(part in output_clean for part in expected_list):
            return True
        
        # Edge cases
        if re.search(r'united states|united kingdom|czech republic', expected, re.IGNORECASE):
            raw_expected = expected.strip().lower().replace('the ', '')
            raw_output = output_text.strip().lower().replace('the ', '')
            if raw_output.startswith(raw_expected) or raw_output.startswith('england') or raw_output == "us":
                return True
        if re.search(r'south korea', expected, re.IGNORECASE):
            if output_clean.startswith('korea') or output_clean.startswith('south korea'):
                return True
        if re.search(r'persian|farsi', expected, re.IGNORECASE):
            if output_clean.startswith('persian') or output_clean.startswith('farsi'):
                return True
        if re.search(r'oceania', expected, re.IGNORECASE):
            if output_clean.startswith('australia'):
                return True
        if re.search(r'north america', expected, re.IGNORECASE):
            if 'north america' in output_clean or output_clean == 'na' or output_clean.startswith('america'):
                return True
        if re.search(r'mandarin|chinese', expected, re.IGNORECASE):
            if 'chinese' in output_clean or 'mandarin' in output_clean:
                return True

        return False

    def get_filtered_indices(dataset, variables_list, target_variable):
        """
        Return a list of row indices to keep, where we keep total_size * ratio rows for each attribute.
        """
        attr_to_indices = {attr: [] for attr in variables_list}
        for i, ex in enumerate(dataset["input"]):
            attr = ex["queried_attribute"]
            if attr in attr_to_indices:
                attr_to_indices[attr].append(i)
        half = len(attr_to_indices[target_variable])
        for attr in variables_list:
            random.shuffle(attr_to_indices[attr])

        final_indices = []
        for attr in variables_list:
            if attr == target_variable: 
                final_indices.extend(attr_to_indices[attr][:half])
            else:
                final_indices.extend(attr_to_indices[attr][:int(half//2)])
                
        random.shuffle(final_indices)
        return final_indices

    def filter_task_dataset(task, dataset_name, variables_list, target_variable):
        """
        Filter one dataset in task by overwriting counterfactual_datasets and raw_counterfactual_datasets.
        """
        # Get the indices to filter
        dataset = task.counterfactual_datasets[dataset_name]
        raw_dataset = task.raw_counterfactual_datasets[dataset_name]
        final_indices = get_filtered_indices(dataset, variables_list=variables_list, target_variable=target_variable)
        # Apply the filter and update task object
        filtered_dataset = dataset.select(final_indices)
        filtered_raw_dataset = raw_dataset.select(final_indices)
        task.counterfactual_datasets[dataset_name] = filtered_dataset
        task.raw_counterfactual_datasets[dataset_name] = filtered_raw_dataset

    for model_name in ["google/gemma-2-2b", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        if (args.skip_gemma and model_name == "google/gemma-2-2b") or \
           (args.skip_llama and model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"):
            continue
        
        pipeline = LMPipeline(model_name, max_new_tokens=2, device=device, dtype=torch.float16)
        pipeline.tokenizer.padding_side = "left"
        if "gemma" in model_name:
            batch_size = 512
            model_id = "gemma-2-2b"
        else:
            batch_size = 256
            model_id = "llama-3.1-8b"

        print("DEVICE:", pipeline.model.device)

        print("INPUT:", task.raw_all_data["input"][0])
        print("LABEL:", task.raw_all_data["label"][0])
        print("PREDICTION:", pipeline.dump(pipeline.generate(task.raw_all_data["input"][0])))

        task.filter(pipeline, checker, verbose=True, batch_size=batch_size)

        token_positions = get_token_positions(pipeline, task, model_name=model_id)
        input = task.sample_raw_input()
        print(input)
        for token_position in token_positions:
            print(token_position.highlight_selected_token(input))

        gc.collect()
        torch.cuda.empty_cache()

        start = 0
        end = pipeline.get_num_layers()
        if "gemma" in model_name:
            config = {"batch_size": 128, "evaluation_batch_size":batch_size, "training_epoch": 1, "n_features": 288, "regularization_coefficient": 0.0, "output_scores": False}
        else:
            config = {"batch_size": 32, "evaluation_batch_size":batch_size, "training_epoch": 1, "n_features": 512, "regularization_coefficient": 0.0, "output_scores": False}
        verbose = True
        results_dir = "results_ravel"

        target_variables = ["Continent", "Country", "Language"]
        names = ["attribute", "wikipedia"]

        train_data = [name + "_train" for name in names]
        validation_data = [name + "_val" for name in names]
        test_data = [name + "_test" for name in names]
        test_data += [name + "_testprivate" for name in names]

        for variable in target_variables:
            if variable != "Country":
                continue
            temp_task = deepcopy(task)
            for name in train_data + test_data:
                print(len(temp_task.counterfactual_datasets[name]))
                filter_task_dataset(task=temp_task, dataset_name=name, variables_list=target_variables, target_variable=variable)
                print(len(temp_task.counterfactual_datasets[name]))
            

            residual_stream_baselines(
                pipeline=pipeline,
                task=temp_task,
                token_positions=token_positions,
                train_data=train_data,
                test_data=test_data,
                config=config,
                target_variables=[variable],
                checker=checker,
                start=start,
                end=end,
                verbose=verbose,
                results_dir=results_dir,
                methods=args.methods
            )