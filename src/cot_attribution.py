import os
import time
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain, combinations


class CoTAttribution:
    """
    A class to calculate the attribution of each sentence in the Chain of Thought (CoT) prompt for a given input and target.
    """

    def __init__(self, openai_org, openai_api_key):
        openai.organization = openai_org
        openai.api_key = openai_api_key
        self.request_counter = 0
        self.last_request_timestamp = time.time()

    def _get_response(self, input, CoT):
        """
        Get response from the OpenAI API.
        
        Args:
            input (str): The user input.
            CoT (str): The Chain of Thought (CoT) prompt.
        
        Returns:
            str: The response from the ChatGPT model.
        """

        # Limit the number of requests to 60 per minute
        if self.request_counter >= 60:
            print('Request counter exceeded. Waiting for 60 seconds.')
            time_since_last_request = time.time() - self.last_request_timestamp
            if time_since_last_request < 60:
                time.sleep(60 - time_since_last_request)

            self.request_counter = 0
            print('Reset request counter.')
            self.last_request_timestamp = time.time()

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": 'Take the last letters of the words in "Bill Gates" and concatenate them.'},
                {"role": "assistant", "content": CoT + ' The answer is ls.'},
                {"role": "user", "content": input},
            ],
            temperature=0,
        )

        self.request_counter += 1
        return response['choices'][0]['message']["content"]

    def _powerset(self, iterable):
        """
        Get the powerset of an iterable.
        
        Args:
            iterable (iterable): The input iterable.
        
        Returns:
            itertools.chain: The powerset of the input iterable.
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def _calculate_local_attribution(self, input, target, CoT, features):
        """
        Calculate the local attribution of sentences in the Chain of Thought (CoT) prompt for a given input and target.

        Args:
            input (str): The user input.
            target (str): The target response.
            CoT (list): The Chain of Thought prompt.
            features (tuple): The features (i.e. sentences in the Cot prompt) to be attributed to.
        
        Returns:
            int: The attribution score.
        """
        features_idx = self.features2idx[features]
        if self.local_attributions[-1][features_idx] is not None:
            return self.local_attributions[-1][features_idx]
        else:
            masked_CoT = [""] * len(CoT)
            for i in range(len(CoT)):
                if i in features:
                    masked_CoT[i] = CoT[i]
            response = self._get_response(input, ' '.join(masked_CoT))
            prediction = 1 if target in response else 0
            if len(features) == 1:
                attribution = prediction
            else:
                attribution = prediction - sum([self._calculate_local_attribution(input, target, CoT, fs) 
                                                for fs in self._powerset(features) if len(fs) < len(features)])
            self.predictions[-1][features_idx] = prediction
            self.local_attributions[-1][features_idx] = attribution
            return attribution
        
    def _calculate_shapely_value(self, attribution_scores):
        """
        Calculate the Shapely value of sentences in the Chain of Thought (CoT) prompt.
        """
        shapely_values = np.zeros(len(attribution_scores))
        for i in range(len(attribution_scores)):
            features = self.idx2features[i]
            if len(features) == 1:
                for j in range(len(attribution_scores)):
                    if features[0] in self.idx2features[j]:
                        shapely_values[i] += 1 / len(self.idx2features[j]) * attribution_scores[j]
        return shapely_values
    
    def _calculate_shapely_interaction_index(self, attribution_scores):
        """
        Calculate the Shapely interaction index of sentences in the Chain of Thought (CoT) prompt.
        """
        shapley_interaction_indices = np.zeros(len(attribution_scores))
        for i in range(len(attribution_scores)):
            features = self.idx2features[i]
            for j in range(len(attribution_scores)):
                if all([f in self.idx2features[j] for f in features]):
                    shapley_interaction_indices[i] += 1 / (len(self.idx2features[j]) - len(features) + 1) * attribution_scores[j]
        return shapley_interaction_indices
    
    def _calculate_total_indirect_effect(self, attribution_scores):
        """
        Calculate the total indirect effect of sentences in the Chain of Thought (CoT) prompt.
        """
        total_indirect_effects = np.zeros(len(attribution_scores))
        for i in range(len(attribution_scores)):
            features = self.idx2features[i]
            if len(features) == 1:
                for j in range(len(attribution_scores)):
                    if features[0] in self.idx2features[j]:
                        total_indirect_effects[i] += attribution_scores[j]
        return total_indirect_effects
    
    def _calculate_archipelago(self, attribution_scores):
        """
        Calculate the archipelago of sentences in the Chain of Thought (CoT) prompt.
        """
        archipelago = np.zeros(len(attribution_scores))
        for i in range(len(attribution_scores)):
            features = self.idx2features[i]
            for j in range(len(attribution_scores)):
                if all([f in features for f in self.idx2features[j]]):
                    archipelago[i] += attribution_scores[j]
        return archipelago


    def _calculate_global_attribution(self, features):
        """
        Calculate the global attribution of sentences in the Chain of Thought (CoT) prompt

        Args:
            features (tuple): The features (i.e. sentences in the Cot prompt) to be attributed to.

        Returns:
            int: The attribution score.
        """
        features_idx = self.features2idx[features]
        if self.global_attributions[features_idx] is not None:
            return self.global_attributions[features_idx]
        else:
            prediction = self.mean_predictions[features_idx]
            if len(features) == 1:
                attribution = prediction
            else:
                attribution = prediction - sum([self._calculate_global_attribution(fs) 
                                                for fs in self._powerset(features) if len(fs) < len(features)])
            self.global_attributions[features_idx] = attribution
            return attribution
        

    def run(self, CoT, input_file, target_file, output_file):
        """
        Run the attribution calculation.

        Args:
        CoT (list): The Chain of Thought prompt.
        input_file (str): The path to the input file.
        target_file (str): The path to the target file.
        output_file (str): The path to the output file.
        """
        self.features2idx = {features: i for i, features in enumerate(self._powerset(range(len(CoT))))}
        self.idx2features = {i: features for features, i in self.features2idx.items()}
        with open(input_file, 'r') as input_f, open(target_file, 'r') as target_f, open(output_file, 'w') as output_f:
            self.predictions = []
            self.local_attributions = []
            inputs = input_f.readlines()
            targets = target_f.readlines()
            output_f.write('Chain of Thought: ' + ' '.join(CoT) + '\n\n')

            # Calculate local attributions
            output_f.write('Local Attributions:\n')
            for input, target in tqdm(zip(inputs, targets), total=len(inputs)):
                input = input.strip()
                target = target.strip()
                baseline = self._get_response(input, '')
                if target in baseline:
                    output_f.write(input + '\n')
                    output_f.write('Standard prompt is sufficient.\n\n')
                    continue
                try:
                    self.predictions.append([None] * len(self.features2idx))
                    self.local_attributions.append([None] * len(self.features2idx))
                    self._calculate_local_attribution(input, target, CoT, tuple(range(len(CoT))))
                except Exception as e:
                    self.predictions.pop()
                    self.local_attributions.pop()
                    output_f.write(input + '\n')
                    output_f.write('Error: ' + str(e) + '\n\n')
                    continue
                shapely_values = self._calculate_shapely_value(self.local_attributions[-1])
                shapely_interaction_indices = self._calculate_shapely_interaction_index(self.local_attributions[-1])
                total_indirect_effects = self._calculate_total_indirect_effect(self.local_attributions[-1])
                archipelago = self._calculate_archipelago(self.local_attributions[-1])
                output_f.write(input + '\n')
                df = pd.DataFrame({'Features': [self.idx2features[i] for i in range(len(self.local_attributions[-1]))],
                                   'Attribution': self.local_attributions[-1],
                                   'Shapely Value': shapely_values,
                                   'SII': shapely_interaction_indices,
                                   'TIE': total_indirect_effects,
                                   'Archipelago': archipelago})
                output_f.write(df.to_string(index=False) + '\n\n')

            # Calculate global attributions
            self.mean_predictions = np.mean(self.predictions, axis=0)
            self.global_attributions = [None] * len(self.features2idx)
            self._calculate_global_attribution(tuple(range(len(CoT))))
            shapely_values = self._calculate_shapely_value(self.global_attributions)
            shapely_interaction_indices = self._calculate_shapely_interaction_index(self.global_attributions)
            total_indirect_effects = self._calculate_total_indirect_effect(self.global_attributions)
            archipelago = self._calculate_archipelago(self.global_attributions)
            output_f.write('Global attributions:\n')
            output_f.write('Number of samples: ' + str(len(self.predictions)) + '\n')
            df = pd.DataFrame({'Features': [self.idx2features[i] for i in range(len(self.global_attributions))],
                               'Attribution': self.global_attributions,
                               'Shapely Value': shapely_values,
                               'SII': shapely_interaction_indices,
                               'TIE': total_indirect_effects,
                               'Archipelago': archipelago})
            output_f.write(df.to_string(index=False) + '\n')
            

if __name__ == '__main__':
    openai_org = input('OpenAI organization: ')
    openai_api_key = input('OpenAI API key: ')
    CoT_attribution = CoTAttribution(openai_org, openai_api_key)
    CoT_sent = ['The last letter of "Bill" is "l".', 'The last letter of "Gates" is "s".', 'Concatenating them is "ls".']
    CoT_phrase = ['The last letter', 'of "Bill"', 'is "l".']
    CoT_attribution.run(CoT_sent, 'input/inputs_100.txt', 'input/targets_100.txt', 'output/output_100_sent.txt')
    CoT_attribution.run(CoT_phrase, 'input/inputs_100.txt', 'input/targets_100.txt', 'output/output_100_phrase.txt')
