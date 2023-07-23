from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from methods.utils import timeit, get_clf_results
import torch
import numpy as np


class GLTRMetric(MetricBasedExperiment):
    def __init__(self, data, model, tokenizer, DEVICE, **kwargs): # Add new arguments, if needed, e.g. base model, DEVICE
        super().__init__(data, self.__class__.__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.DEVICE = DEVICE
    
    def criterion_fn(self, text: str):
        with torch.no_grad():
            tokenized = self.tokenizer(text, return_tensors="pt").to(self.DEVICE)
            logits = self.model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                    == labels.unsqueeze(-1)).nonzero()

            assert matches.shape[
                1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"
            ranks = ranks.float()
            res = np.array([0.0, 0.0, 0.0, 0.0])
            for i in range(len(ranks)):
                if ranks[i] < 10:
                    res[0] += 1
                elif ranks[i] < 100:
                    res[1] += 1
                elif ranks[i] < 1000:
                    res[2] += 1
                else:
                    res[3] += 1
            if res.sum() > 0:
                res = res / res.sum()

            return res
    
    @timeit
    def run(self):
        torch.manual_seed(0)
        np.random.seed(0)

        train_text = self.data['train']['text']
        train_label = self.data['train']['label']
        train_criterion = [self.criterion_fn(train_text[idx])
                        for idx in range(len(train_text))]
        x_train = np.array(train_criterion)
        y_train = train_label

        test_text = self.data['test']['text']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx])
                        for idx in range(len(test_text))]
        x_test = np.array(test_criterion)
        y_test = test_label

        train_res, test_res = get_clf_results(x_train, y_train, x_test, y_test)

        acc_train, precision_train, recall_train, f1_train, auc_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

        return {
            'name': f'{self.name}_threshold',
            'predictions': {'train': train_criterion, 'test': test_criterion},
            'general': {
                'acc_train': acc_train,
                'precision_train': precision_train,
                'recall_train': recall_train,
                'f1_train': f1_train,
                'auc_train': auc_train,
                'acc_test': acc_test,
                'precision_test': precision_test,
                'recall_test': recall_test,
                'f1_test': f1_test,
                'auc_test': auc_test,
            }
        }
