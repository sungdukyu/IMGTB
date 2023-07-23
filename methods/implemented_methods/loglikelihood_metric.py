from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
import torch

class LoglikelihoodMetric(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.DEVICE)
            labels = tokenized.input_ids
            return -self.model(**tokenized, labels=labels).loss.item()
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1317