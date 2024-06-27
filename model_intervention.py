import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer, utils
import copy
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import re

torch.set_grad_enabled(False)

def _extract_prefix_before_layer_number(name):
    """
    Extracts the prefix before the layer number in a parameter name.

    Args:
        name (str): The name of the parameter.

    Returns:
        str: The prefix before the layer number.
    """
    parts = name.split('.')
    prefix_parts = []
    for part in parts:
        if re.match(r'^\d+$', part):  # Check if the part is a number
            break
        prefix_parts.append(part)
    return '.'.join(prefix_parts)

def _extract_layer_prefixes(model):
    """
    Extracts the unique layer prefixes from a model's parameters.

    Args:
        model (torch.nn.Module): The model from which to extract layer prefixes.

    Returns:
        str: The unique prefix for the layers.
    """
    for name, param in model.named_parameters():
        prefix = _extract_prefix_before_layer_number(name)
        if 'weight' not in prefix and 'bias' not in prefix:
            return prefix

class ModelExperiment:
    """
    Class to perform experiments on a transformer model such as layer swapping and ablation.

    Attributes:
        model_name (str): The name of the model to use.
        device (str): The device to use for computation.
        model (AutoModelForCausalLM): The original transformer model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        original_state_dict (OrderedDict): The original state dictionary of the model.
        hooked_model (HookedTransformer): The hooked version of the model for intervention experiments.
    """
    def __init__(self, model_name="openai-community/gpt2", device='cuda:1'):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        
        self.hooked_model = HookedTransformer.from_pretrained(self.model_name.split("/")[-1], device=self.device, tokenizer=self.tokenizer, hf_model=self.model, center_unembed=True, center_writing_weights=True, fold_ln=True)
        self.hooked_model.eval()

    def reset_to_original_model(self):
        """
        Resets the model to its original state.
        """
        self.model.load_state_dict(self.original_state_dict)
        self.hooked_model = HookedTransformer.from_pretrained(self.model_name.split("/")[-1], device=self.device, tokenizer=self.tokenizer, hf_model=self.model, center_unembed=True, center_writing_weights=True, fold_ln=True)
        self.hooked_model.eval()

    def swap_model_layers(self, layer_a, layer_b):
        """
        Swaps two layers in the model.

        Args:
            layer_a (int): The index of the first layer.
            layer_b (int): The index of the second layer.
        """
        def swap_state_dict_layers(model_state_dict, layer_a, layer_b):
            prefix = _extract_layer_prefixes(self.model)
            layer_a_pattern = f"{prefix}.{layer_a}."
            layer_b_pattern = f"{prefix}.{layer_b}."
            result_dict = copy.deepcopy(model_state_dict)
            temp_storage = {}

            for key in model_state_dict.keys():
                if layer_a_pattern in key:
                    new_key = key.replace(layer_a_pattern, layer_b_pattern)
                    temp_storage[new_key] = model_state_dict[key]
                elif layer_b_pattern in key:
                    new_key = key.replace(layer_b_pattern, layer_a_pattern)
                    temp_storage[new_key] = model_state_dict[key]

            for temp_key, value in temp_storage.items():
                result_dict[temp_key] = value

            return result_dict

        new_dict = swap_state_dict_layers(self.original_state_dict, layer_a, layer_b)
        self.model.load_state_dict(new_dict)
        self.hooked_model = HookedTransformer.from_pretrained(self.model_name.split("/")[-1], device=self.device, tokenizer=self.tokenizer, hf_model=self.model, center_unembed=True, center_writing_weights=True, fold_ln=True)
        self.hooked_model.eval()

    def ablate_model_layer(self, text, layer):
        """
        Ablates (sets to zero) the outputs of a specific layer.

        Args:
            text (str): The input text to the model.
            layer (int): The index of the layer to ablate.

        Returns:
            tuple: The logits and loss after ablation.
        """
        def zero_out_layer_hook(value, hook):
            value[:, :, :] = 0.
            return value

        tokens = self.hooked_model.to_tokens(text, prepend_bos=True)
        logits = self.hooked_model.run_with_hooks(
            tokens,
            return_type="logits",
            fwd_hooks=[
                (utils.get_act_name("attn_out", layer), zero_out_layer_hook),
                (utils.get_act_name("mlp_out", layer), zero_out_layer_hook)
            ]
        )
        loss = self.hooked_model.loss_fn(logits, tokens, per_token=True)
        self.hooked_model.reset_hooks()
        return logits, loss

    def compute_logits_and_loss(self, input):
        """
        Computes the logits and loss for a given input.

        Args:
            input (str): The input text to the model.

        Returns:
            tuple: The logits and loss for the input text.
        """
        logits = self.hooked_model(input)
        loss = self.hooked_model.loss_fn(logits, self.hooked_model.to_tokens(input, prepend_bos=True), per_token=True)
        return logits, loss

class ModelMetrics:
    """
    Class to compute various metrics for model evaluation.
    """
    @staticmethod
    def compute_kl_divergence(logit_p: torch.Tensor, logit_q: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Computes the Kullback-Leibler (KL) divergence between two sets of logits.

        Args:
            logit_p (torch.Tensor): The first set of logits.
            logit_q (torch.Tensor): The second set of logits.
            dim (int): The dimension over which to compute the KL divergence.

        Returns:
            torch.Tensor: The KL divergence.
        """
        log_p = logit_p.log_softmax(dim)
        log_q = logit_q.log_softmax(dim)
        return torch.sum(log_p.exp() * (log_p - log_q), dim)

    @staticmethod
    def compute_base2_entropy(logits):
        """
        Computes the base-2 entropy of the logits.

        Args:
            logits (torch.Tensor): The logits from the model.

        Returns:
            torch.Tensor: The entropy.
        """
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-9)
        log_probs = torch.log(probs) / torch.log(torch.tensor(2.0))
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

def run_layer_intervention_experiment(model, intervention_type='swap', num_samples=1388):
    """
    Runs a layer intervention experiment by swapping or ablating layers in the model.

    Args:
        model (ModelExperiment): The model to run the experiment on.
        intervention_type (str): The type of intervention ('swap' or 'ablate').
        num_samples (int): The number of samples to use in the experiment.

    Returns:
        pd.DataFrame: A dataframe containing the results of the experiment.
    """
    torch.cuda.empty_cache()

    dataset = load_dataset("EleutherAI/the_pile_deduplicated", split='train', streaming=True)
    
    kl_divs = []
    losses = []
    layers = []
    token_normal = []
    token_intervened = []
    token = []
    entropy_normal = []
    entropy_intervened = []
    loss_normals = []
    loss_interveneds = []

    metrics = ModelMetrics()
    normal_model = model
    intervened_model = ModelExperiment(model_name=model.model_name, device=model.device)

    n_layers = normal_model.hooked_model.cfg.n_layers
    layer_range = range(n_layers - 1) if intervention_type == 'swap' else range(n_layers)

    for i in tqdm(layer_range):
        if intervention_type == 'swap':
            intervened_model.swap_model_layers(i, i+1)
        
        for sample in dataset.take(num_samples):
            logits_normal, loss_normal = normal_model.compute_logits_and_loss(sample["text"])
            
            if intervention_type == 'swap':
                logits_intervened, loss_intervened = intervened_model.compute_logits_and_loss(sample["text"])
            else:  # ablation
                logits_intervened, loss_intervened = normal_model.ablate_model_layer(sample["text"], i)
            
            token.extend([normal_model.hooked_model.to_string(tkn) for tkn in normal_model.hooked_model.to_tokens(sample["text"], prepend_bos=True)[0, 1:]])
            
            logits_normal = logits_normal[0, 1:].cpu()
            logits_intervened = logits_intervened[0, 1:].cpu()
            loss_normal = loss_normal.cpu()
            loss_intervened = loss_intervened.cpu()

            loss_normals.extend(loss_normal.squeeze(0).tolist())
            loss_interveneds.extend(loss_intervened.squeeze(0).tolist())
            
            kl_div = metrics.compute_kl_divergence(logits_normal, logits_intervened, dim=-1)
            kl_divs.extend(kl_div.tolist())
            losses.extend((loss_normal - loss_intervened).squeeze(0).tolist())
            layers.extend([i] * loss_normal.shape[1])

            entropy_normal.extend(metrics.compute_base2_entropy(logits_normal).tolist())
            entropy_intervened.extend(metrics.compute_base2_entropy(logits_intervened).tolist())

            token_normal.extend([normal_model.hooked_model.to_string(tkn) for tkn in logits_normal.argmax(-1).tolist()])
            token_intervened.extend([intervened_model.hooked_model.to_string(tkn) for tkn in logits_intervened.argmax(-1).tolist()])

        if intervention_type == 'swap':
            intervened_model.reset_to_original_model()

    results = pd.DataFrame({
        'Token': token,
        'KL Divergence': kl_divs,
        'Layer Intervened':[s for s in layers],
        'Loss Difference': losses,
        'Token Normal': token_normal,
        'Token Intervened': token_intervened,
        'Loss Normal': loss_normals,
        'Loss Intervened': loss_interveneds,
        'Entropy Normal': entropy_normal,
        'Entropy Intervened': entropy_intervened,
    })

    return results

if __name__ == "__main__":
    #For Phi Models do: (See Github Issue)
    #   hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
    #   tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token = True, use_fast=False, trust_remote_code=True)

    #Put / before name, or use HuggingFace model names {family}/{model_name}
    #Make sure transformer_lens supports the model https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
    
    #Consider fixing the number of tokens that the model processes or remove the end of sentence token for analysis


    model = ModelExperiment(model_name="EleutherAI/pythia-410m-deduped", device='cuda:1')
    
    print("Running swap experiment...")
    swap_results = run_layer_intervention_experiment(model, intervention_type='swap', num_samples=10)
    swap_results.to_csv("swap_experiment_results.csv", index=False)
    print(swap_results.head())
    #Save 
    #swap_results.to_csv("swap_experiment_results.csv", index

    print("\nRunning ablation experiment...")
    ablation_results = run_layer_intervention_experiment(model, intervention_type='ablate', num_samples=10)
    ablation_results.to_csv("ablation_experiment_results.csv", index=False)
    print(ablation_results.head())
    #Save
    #ablation_results.to_csv("ablation_experiment_results.csv", index=False)
