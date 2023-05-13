import copy
import torch 
import itertools
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class Intervention:
    def __init__(self, model, inputs, meds, num_batch, med_type, layers):
        self.inputs = inputs
        self.mediators = meds
        self.num_batch = num_batch
        self.med_type_to_interv = med_type
        self.layers_to_interv = layers
        self.cf_hidden = self.set_cf_hidden(model)
        self.mode = None
        self.in_type = None
            
    def set_mode(self, mode):
        self.mode = mode
        
    def set_in_type(self, in_type):
        self.in_type = in_type
        
    def set_cf_hidden(self, model):
        cf_hidden = {}
        for in_type in ["control", "intervention"]:
            cf_in = self.inputs["intervention"] if in_type == "control" else self.inputs["control"]
            output = model(**cf_in, output_hidden_states=True, output_attentions=True)
            if output.__class__.__name__.startswith("Seq2Seq"):
                cf_hidden[in_type] = output.encoder_hidden_states[1:] + output.decoder_hidden_states[1:]
            else:
                cf_hidden[in_type] = output.hidden_states
        return cf_hidden
    
    def get_input(self):
        return self.inputs[self.in_type]

    def get_mode(self):
        return self.mode
        
    def get_cf_hidden(self):
        return self.cf_hidden[self.in_type]

    def get_med_type_to_interv(self):
        return self.med_type_to_interv
    
    def get_layers_to_interv(self):
        return self.layers_to_interv
                
    def get_med_batches(self, layer):
        meds_to_interv = self.mediators[self.med_type_to_interv][layer]
        if self.mode == "single":
            batch_size = len(meds_to_interv) // self.num_batch
            med_batches = [meds_to_interv[i:i + min(batch_size, len(meds_to_interv) - i)] 
                            for i in range(0, len(meds_to_interv), batch_size)]
            
        elif self.mode == "pair":
            pairs = [i + j if i != j else i for i, j in itertools.product(meds_to_interv, repeat=2)]
            batch_size = len(pairs) // self.num_batch
            med_batches = [pairs[i:i + min(batch_size, len(pairs) - i)] 
                            for i in range(0, len(pairs), batch_size)]
        else:
            raise ValueError("mode must be either 'single' or 'pair'")
        
        return med_batches

            
class CMA: 
    def __init__(self, checkpoint, outcome, device):
        self.device = device

        self.config = AutoConfig.from_pretrained(checkpoint)
        self.label2id = {k.lower(): int(v) for k, v in self.config.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(self.device)
        
        self.layers = self.get_layers()
        self.num_neurons = self.config.hidden_size
        
        self.outcome = torch.tensor([self.label2id[outcome]], device=self.device)
        
        self.intervention = None
                                                                    
    def get_layers(self):
        model_name = self.model.__class__.__name__
        base_model = getattr(self.model, self.model.base_model_prefix)
        if model_name.startswith("Bart"):
            layers = base_model.encoder.layers + base_model.decoder.layers
        elif model_name.startswith("GPT"):
            layers = base_model.h
        elif model_name.startswith("DistilBert"):
            layers = base_model.transformer.layer
        elif model_name.startswith("Albert"):
            layers = []
            for i in range(self.config.num_hidden_layers):
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))
                layer_group = base_model.encoder.albert_layer_groups[group_idx]
                layers.append(layer_group)
            layers = torch.nn.ModuleList(layers)
        elif any(model_name.startswith(x) for x in ["Bert", "Roberta", "Deberta", "MobileBert"]):
            layers = torch.nn.ModuleList([base_model.embeddings] + [*base_model.encoder.layer])
        return layers
    
    def get_cls_token(self, input_ids):
        model_name = self.model.__class__.__name__
        if model_name.startswith("Bart"):
            token = input_ids.eq(self.config.eos_token_id)[0]
        elif model_name.startswith("GPT2"):
            token = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        elif any(model_name.startswith(x) for x in ["Bert", "Roberta", "Deberta", "DistilBert", "MobileBert"]):
            token = 0
        return token
    
    def intervene(self, input, layer, med_batch):
        def replace_output(module, input, output, new_output, med_batch):
            is_tuple = True
            if not isinstance(output, tuple):
                is_tuple = False
                output = (output,)
            for i, meds in enumerate(med_batch):
                if self.intervention.get_med_type_to_interv() == "neuron":
                    output[0][i, :, meds] = new_output[i, :, meds]
                elif self.intervention.get_med_type_to_interv() == "token":
                    output[0][i, meds, :] = new_output[i, meds, :]
            if not is_tuple:
                return output[0]
            return output

        batch_size = len(med_batch)
        in_batch = {k: v.repeat(batch_size, 1) for k, v in input.items()}
        
        hooks = []
        with torch.no_grad():
            new_output = self.intervention.get_cf_hidden()[layer].repeat(batch_size, 1, 1)
            hooks.append(self.layers[layer].register_forward_hook(
                lambda m, i, o: replace_output(m, i, o, new_output, med_batch)))
            logits = self.model(**in_batch).logits
            for hook in hooks:
                hook.remove()
        return logits

    def get_logits(self, input):
        if self.intervention.get_mode() is None:
            logits = self.model(**input).logits
        else:
            logits = []
            for layer in self.intervention.get_layers_to_interv():
                med_batches = self.intervention.get_med_batches(layer)
                logits.append(torch.cat([self.intervene(input, layer, med_batch) for med_batch in med_batches], dim=0))
            logits = torch.stack(logits, dim=0)
        return logits
        
    def get_response(self, in_type, mode=None):
        self.intervention.set_mode(mode)
        self.intervention.set_in_type(in_type)
        input = self.intervention.get_input()
        logits = self.get_logits(input)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        outcome_prob = probs.index_select(-1, self.outcome).sum(dim=-1)
        response = torch.log(outcome_prob / (1 - outcome_prob))
        # response = torch.log(outcome_prob)
        if mode == "pair":
            size = int(response.size(dim=1) ** (1/2))
            response = response.reshape((-1, size, size))
        return response
    
    def init_intervention(self, inputs, meds=None, num_batch=None, med_type=None, layers=None):
        self.intervention = Intervention(self.model, inputs, meds, num_batch, med_type, layers)

    def get_effect(self, effect_type):
        if effect_type == "TE":
            effect = self.get_response("intervention") - self.get_response("control")
        elif effect_type == "PIE":
            effect = self.get_response("control", "single") - self.get_response("control")
        elif effect_type == "TIE":
            effect = self.get_response("intervention") - self.get_response("intervention", "single")
        elif effect_type == "MI":
            t1 = self.get_response("control", "pair")
            t2 = self.get_response("control", "single")
            t3 = t2.clone()
            t4 = self.get_response("control")
            for layer in self.intervention.get_layers_to_interv():
                t1[layer, :, :] = t1[layer, :, :] - t2[layer, :]
            t1 = t1.swapaxes(1, 2)
            for layer in self.intervention.get_layers_to_interv():
                t1[layer, :, :] = t1[layer, :, :] - t3[layer, :]
            effect = t1.swapaxes(1, 2) + t4
        elif effect_type == "PIE_pair":
            effect = self.get_response("control", "pair") - self.get_response("control")
        elif effect_type == "TIE_pair":
            effect = self.get_response("intervention") - self.get_response("intervention", "pair")
        elif effect_type == "MI_T":
            t1 = self.get_response("intervention", "pair")
            t2 = self.get_response("intervention", "single")
            t3 = t2.clone()
            t4 = self.get_response("intervention")
            for layer in self.intervention.get_layers_to_interv():
                t1[layer, :, :] = t1[layer, :, :] - t2[layer, :]
            t1 = t1.swapaxes(1, 2)
            for layer in self.intervention.get_layers_to_interv():
                t1[layer, :, :] = t1[layer, :, :] - t3[layer, :]
            effect = t1.swapaxes(1, 2) + t4

        return effect 

    def get_sample_effects(self, sample, effect_type, meds_list=None, *args, **kwargs):
        effects = []
        for idx, inputs in tqdm(enumerate(sample), total=len(sample)):
             # Update the meds with the current instance's meds_token
            if meds_list is None:
                current_meds = None
            else:
                current_meds = copy.deepcopy(meds_list)
                current_meds["token"] = meds_list["token"][idx]

            self.init_intervention(inputs, meds=current_meds, *args, **kwargs)
            
            effect = self.get_effect(effect_type)
            effect = effect.detach().cpu().numpy()
            effects.append(effect)
        return effects