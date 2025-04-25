import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CausalAbstraction.tasks.ARC.ARC import get_token_positions, get_task
import gc, torch
from CausalAbstraction.pipeline import LMPipeline
from CausalAbstraction.experiments.aggregate_experiments import residual_stream_baselines

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments with optional flags.")
    parser.add_argument("--skip_gemma", action="store_true", help="Skip running experiments for Gemma model.")
    parser.add_argument("--skip_llama", action="store_true", help="Skip running experiments for Llama model.")
    parser.add_argument("--skip_answer_pointer", action="store_true", help="Skip experiments for answer_pointer variable.")
    parser.add_argument("--use_gpu1", action="store_true", help="Use GPU1 instead of GPU0 if available.")
    parser.add_argument("--skip_answer", action="store_true", help="Skip experiments for answer variable.")
    parser.add_argument("--methods", nargs="+", default=["full_vector", "DAS", "DBM+PCA", "DBM", "DBM+SAE"], help="List of methods to run.")
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    task = get_task(hf=True, size=None)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.use_gpu1 and torch.cuda.is_available():
        device = "cuda:1"

    def checker(output_text, expected):
        return expected in output_text

    for model_name in ["google/gemma-2-2b", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        if (args.skip_gemma and model_name == "google/gemma-2-2b") or (args.skip_llama and model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"):
            continue

        pipeline = LMPipeline(model_name, max_new_tokens=1, device=device, dtype=torch.float16)
        pipeline.tokenizer.padding_side = "left"
        batch_size = 64

        task.filter(pipeline, checker, verbose=True, batch_size=batch_size)

        token_positions = get_token_positions(pipeline, task)
        input = task.sample_raw_input()
        print(input)
        for token_position in token_positions:
            print(token_position.highlight_selected_token(input))

        gc.collect()
        torch.cuda.empty_cache()

        start = 0
        end = pipeline.get_num_layers()

        if model_name == "google/gemma-2-2b":
            config = {"batch_size": 48, "evaluation_batch_size":batch_size, "training_epoch": 2, "n_features": 16, "regularization_coefficient": 0.0}
        elif model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            config = {"batch_size": 16, "evaluation_batch_size":batch_size, "training_epoch": 1, "n_features": 16, "regularization_coefficient": 0.0}
        names = ["answerPosition", "randomLetter", "answerPosition_randomLetter"]

        train_data = [name + "_train" for name in names]
        validation_data = [name + "_validation" for name in names]
        test_data = [name + "_test" for name in names]
        # test_data += [name + "_testprivate" for name in names]
        verbose = False
        results_dir = "results_ARC_easy"


        if not args.skip_answer_pointer:
            residual_stream_baselines(
                pipeline=pipeline,
                task=task,
                token_positions=token_positions,
                train_data=train_data,
                test_data=test_data,
                config=config,
                target_variables=["answer_pointer"],
                checker=checker,
                start=start,
                end=end,
                verbose=verbose,
                results_dir=results_dir,
                methods=args.methods
            )

        if not args.skip_answer:
            config["n_features"] = pipeline.model.config.hidden_size // 2
            residual_stream_baselines(
                pipeline=pipeline,
                task=task,
                token_positions=token_positions,
                train_data=train_data,
                test_data=test_data,
                config=config,
                target_variables=["answer"],
                checker=checker,
                start=start,
                end=end,
                verbose=verbose,
                results_dir=results_dir,
                methods=args.methods
            )