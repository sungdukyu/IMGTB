from methods.abstract_methods.metric_based_experiment import MetricBasedExperiment
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AzureOpenAI
import httpx
import os
import torch
import time
import gc
from tqdm import tqdm
from methods.utils import timeit, move_model_to_device


class AnchorEmbedding(MetricBasedExperiment):
    def __init__(self, data, config):
        super().__init__(data, self.__class__.__name__, config) # Set your own name or leave it set to the class name
        self.config['embedding_model_name'] = 'text-embedding-3-small' # OpenAI

        # set up a client
        if self.is_openai_model():
            # (Make sure to configure api key and endpoint correctly)
            # do not put openai_key and openai_endpoint. this can cause 2 problems.
            # 1. security issue. At the end of run, all config will dump to a json file.
            # 2. openai resource conflict. openai api is used in different methods, but they are not necessarily using the same key and endpoint.
            _openai_key = os.getenv("AZURE_OPENAI_API_KEY_2")
            _openai_endpoint = os.getenv("AZURE_OPENAI_API_BASE_2")
            self.client = AzureOpenAI(api_key = _openai_key,
                                      azure_endpoint = _openai_endpoint,
                                      api_version="2024-06-01",
                                      http_client = httpx.Client(verify=False),
                                      )

    def check_data_compatibility(self):
        # make sure it has 'text_anchor' in the key of dictionary data
        pass

    def is_openai_model(self):
        return self.config['embedding_model_name'] in ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
        
    def generate_embedding(self, input_text:str):
        if self.is_openai_model():
            response = self.client.embeddings.create(model = self.config['embedding_model_name'],
                                                     input = input_text,
                                                     )
            return np.array(response.data[0].embedding)
        
    def criterion_fn(self, text: str, text_anchor: str):
        """
        Computes a similarity score between the embeddings of the input text and the anchor text.
    
        Args:
            text (str)
            text_anchor (str)
            
        Returns a numpy array of numeric criteria (numeric scores)
        """

        embedding_test = self.generate_embedding(text)
        embedding_anchor = self.generate_embedding(text_anchor)
        sim_score = cosine_similarity([embedding_test], [embedding_anchor])
        sim_score = sim_score.reshape(1) # (1,1) -> (1), required for logistic regression

        return np.clip(sim_score, 0., 1.)
    
    # only differente to the original 'run' method: "train_text_anchor" and "test_text_anchor"
    @timeit
    def run(self):
        start_time = time.time()
        
        print(f"<<< Running {self.name} experiment >>>")
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Using cache dir {self.cache_dir}")
        
        if "cuda" in self.DEVICE and not torch.cuda.is_available():
            print(f'Setting default device to cpu. Cuda is not available.')
            self.DEVICE = "cpu"

        print(f"Loading BASE model {self.base_model_name}\n")
        self.base_model, self.base_tokenizer = self.load_base_model_and_tokenizer(
            self.base_model_name, self.cache_dir)
        move_model_to_device(self.base_model, self.DEVICE)
            
        torch.manual_seed(0)
        np.random.seed(0)

        # get train data
        train_text = self.data['train']['text']
        train_text_anchor = self.data['train']['text_anchor']
        train_label = self.data['train']['label']
        train_criterion = [self.criterion_fn(train_text[idx], train_text_anchor[idx])
                        for idx in tqdm(range(len(train_text)), desc="Computing metrics on train partition")]
        x_train = np.array(train_criterion) # predicted
        y_train = train_label               # truth

        test_text = self.data['test']['text']
        test_text_anchor = self.data['test']['text_anchor']
        test_label = self.data['test']['label']
        test_criterion = [self.criterion_fn(test_text[idx], test_text_anchor[idx])
                        for idx in tqdm(range(len(test_text)), desc="Computing metrics on test partition")]
        x_test = np.array(test_criterion)
        y_test = test_label
        train_pred, test_pred, train_pred_prob, test_pred_prob, train_res, test_res = self.get_clf_results(x_train, y_train, x_test, y_test)
                
        acc_train, precision_train, recall_train, f1_train, auc_train, specificity_train = train_res
        acc_test, precision_test, recall_test, f1_test, auc_test, specificity_test = test_res

        print(f"{self.name} acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}, specificity_train: {specificity_train}")
        print(f"{self.name} acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}, specificity_test: {specificity_test}")
        
        end_time = time.time()
        
         # Clean up
        del self.base_model
        gc.collect()
        torch.cuda.empty_cache()

        
        return {
            'name': f'{self.name}_threshold',
            'type': 'metric-based',
            'input_data': self.data,
            'predictions': {'train': train_pred, 'test': test_pred},
            'machine_prob': {'train': train_pred_prob, 'test': test_pred_prob},
            'criterion': {'train': [elem.tolist() for elem in train_criterion], 'test': [elem.tolist() for elem in test_criterion]},
            'running_time_seconds': end_time - start_time,
            'metrics_results': {
                'train': {
                    'acc': acc_train,
                    'precision': precision_train,
                    'recall': recall_train,
                    'f1': f1_train,
                    'specificity': specificity_train
                },
                'test': {
                    'acc': acc_test,
                    'precision': precision_test,
                    'recall': recall_test,
                    'f1': f1_test,
                    'specificity': specificity_test
                }
            },
            "config": self.config
        }