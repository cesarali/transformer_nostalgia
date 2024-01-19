import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer


import click
from nostalgia.utils.helper import export_list_of_lists_to_csv
from nostalgia.data.dataloaders import DataLoaderFactory
from nostalgia.data.dataloaders.reasoning import load_tokenizer
from nostalgia.models.models import LLMCausal, ModelFactory

from nostalgia.utils.evaluation import EvaluationFactory
from nostalgia.utils.helper import load_yaml

# Define a list of valid model types
VALID_MODEL_TYPES = ["meta-llama/Llama-2-7b-chat-hf"]

@click.command()
@click.option("--config", "-c", default="config.yaml", type=click.Path(exists=True, dir_okay=False), help="Path to config file.")
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
@click.option("-m", "--max_samples", "max_samples", default=20, type=int, help="Maximal number of samples from dataset")
@click.option("-s", "--state", "state_list", type=int, multiple=True, help="State to investigate.")
@click.option("-g", "--generate", "generate", flag_value=True, default=False, help="Generates new activations if true.")
@click.option("-d", "--delta", "time_delta", default=1, help="Represents the difference of timesteps to investigate.")
@click.option("-l", "--layer", "layers_to_save", type=str, multiple=True, help="Layers to investigate.")
def main(config: Path, log_level, max_samples, state_list, generate, time_delta, layers_to_save):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    state_list = list(set(state_list))
    state_list.sort()

    if time_delta < 1:
        raise ValueError("Please pass a time delta above 0!")

    if max_samples < 0:
        raise ValueError("Please pass a non-negative count of maximum samples to compute")
    
    if len(layers_to_save) == 0:
        raise ValueError("Please pass one or multiple layers to investigate via: -l \"LAYER_NAME\"")

    config_dict = load_yaml(config)

    # Your inference code here using the provided config parameters and prompt
    print(f"Model Type: {config_dict['model']['name']}")
    print(f"Dataset Name: {config_dict['dataset']['name']}")
    print(f"Dataset Location: {config_dict['dataset']['root_dir']}")
    print(f"Prompt Location: {config_dict['dataset']['prompt_path']}")
    print(f"Output Location: {config_dict['output_path']}")
    print(f"States to investigate: {state_list}")
    print(f"Generate new activations: {generate}")
    print(f"Time delta: {time_delta}")

    tokenizer_conf = config_dict["tokenizer"]
    dataset_conf = config_dict["dataset"]
    model_conf = config_dict["model"]
    generation_conf = config_dict["generation"]

    tokenizer = load_tokenizer(**tokenizer_conf)
    dataloader = DataLoaderFactory.create(**dataset_conf, tokenizer=tokenizer)
    num_added_tokens = len(tokenizer.added_tokens_decoder)
    device = config_dict["device_map"]
    if isinstance(tokenizer, GPT2Tokenizer):
        num_added_tokens -= 1
    model = ModelFactory.create(
        **model_conf, pad_token_id=tokenizer.pad_token_id, num_added_tokens=num_added_tokens, device_map=device
    )

    max_new_tokens = generation_conf["max_new_tokens"]
    answer_pattern = re.compile(generation_conf["answer_pattern"])
    targets = []
    predictions = []
    output_path = Path(config_dict["output_path"])

    model.set_layers_to_save(layers_to_save)
    model.set_generate_new_activations(generate)
    saved_ids = []
    id_counter = 0

    # 1.) generate activations

    dataset = getattr(dataloader, dataset_conf["split"] + "_it")

    if max_samples > len(dataset) - 1:
        max_samples = len(dataset) - 1

    if generate:
        for x in tqdm(dataset):
            model.set_ids_to_save(x["id"])
            generate_txt = model.generate(
                x["input_ids"].to("cuda" if device == "auto" else device),
                x["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=generation_conf["do_sample"],
                tokenizer=tokenizer,
            )

            for _id, p, t in zip(x["id"], generate_txt, x["answerKey"]):
                match = answer_pattern.search(p)
                if match:
                    extracted_value = match.group(1)
                else:
                    extracted_value = "X"
                targets.append({"id": _id, "answerKey": t})
                predictions.append([_id, extracted_value.upper(), p])
            
            id_counter += len(x["id"])
            saved_ids.extend(x["id"])
            if id_counter >= max_samples:
                break

        output_path.mkdir(parents=True, exist_ok=True)
        export_list_of_lists_to_csv(predictions, output_path / "predictions.csv")
    

    if not generate:
        if not os.path.isdir("activations/"):
            print("Warning: There are no activations saved at ./activations/. Please use -g option for generation.")
            return
        saved_ids = [ item for item in os.listdir("activations/") if os.path.isdir(os.path.join("activations/", item)) ]
        if len(saved_ids)==0:
            print("Warning: There are no activations saved at ./activations/. Please use -g option for generation.")
            return


    # 2.) load activations and compute norms    

    norms = []
    for id in saved_ids:
        norm_sequence = []
        last_it = 0
        for it in range(1,max_new_tokens):
            delta_layerwise = []
            for layer in layers_to_save:
                state1_act, state2_act = model.load_state_from_disk(id=id,layer_name=layer,input_iteration=last_it,output_iteration=it)
                delta = (state2_act - state1_act).squeeze()
                delta_layerwise.append(delta)

            delta_layerwise = torch.stack(delta_layerwise)
            norm_sequence.append(torch.norm(delta_layerwise, dim=-1))

            last_it = it

        norm_sequence = torch.stack(norm_sequence)
        norms.append(norm_sequence)
        
    norms = torch.stack(norms)

    # 3.) fitting and plotting

    import matplotlib.pyplot as plt
    plt.figure(1)

    if len(state_list) == 0:        
        for layer_idx, layer in enumerate(layers_to_save):
            plt.subplot(1,len(layers_to_save), layer_idx+1)
            plt.title(layer)
            t =norms[:,:,layer_idx]
            
            plt.hist( t.flatten(), bins=100, density=True)


    else:
        # plot user specified activations

        axes_list = []
        xmax = 0

        for state_idx,state in enumerate(state_list):
            for layer_idx, layer in enumerate(layers_to_save):
                ax = plt.subplot(len(state_list),len(layers_to_save), state_idx*len(layers_to_save)+layer_idx+1)
                axes_list.append(ax)

                # plot title only over the first row
                if state_idx == 0:
                    plt.title(layer)

                if layer_idx == 0:
                    state1 = state + time_delta
                    state2 = state 
                    plt.ylabel(f"t{state1}-t{state2}")

                t =norms[:,state,layer_idx]

                p = plt.hist( t.flatten(), bins=100, density=True)

                p_xmax = p[1].max()
                if xmax < p_xmax:
                    xmax = p_xmax

        for ax in axes_list:
            ax.set_xlim(-0.1, xmax)

    plt.show()

if __name__ == "__main__":
    main()
