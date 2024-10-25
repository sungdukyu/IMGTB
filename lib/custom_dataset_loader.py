from typing import Dict, List, Any
import pandas as pd

# the custom dataset loader function name format: 'process_<dataset-name>'
#
# Usage: python benchmark.py --dataset custom auto <dataset_name>
# e.g.,  python benchmark.py --dataset custom auto train_iclr2022_test_iclr2022_mini

def read_csv(f:str, random_state:int=42, mini:bool=False):
        data = pd.read_csv(f)
        data = data[['label', 'text', 'text_anchor' if 'text_anchor' in data.columns else ['label', 'text']]]
        if mini:
            data = data.sample(frac=.005, random_state=random_state) # sampling only 0.005, so 'mini'
        return data.to_dict(orient='list')

# (for debugging)
def process_custom_mini(data, config) -> Dict[str, Dict[str, List[Any]]]: # data and config arguments are dummies and left for compatibility.
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.with_anchor.csv'
    f_test  = '~/my_datasets/test.500.iclr2022_gpt-4o.with_anchor.csv'
    return {"train": read_csv(f_train, mini=True), "test": read_csv(f_test, mini=True)}

# (train data is fixed aith 2022)
def process_train_iclr2022_gpt_4o_test_iclr2021_gpt4o(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2021_gpt-4o.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2022_gpt4o(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2022_gpt-4o.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2023_gpt4o(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2023_gpt-4o.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2024_gpt4o(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2024_gpt-4o.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2021_llama3(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2021_llama-3.1-70b.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2022_llama3(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2022_llama-3.1-70b.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2023_llama3(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2023_llama-3.1-70b.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

def process_train_iclr2022_gpt_4o_test_iclr2024_llama3(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.csv'
    f_test  = '~/my_datasets/test.500.iclr2024_llama-3.1-70b.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}

# (with anchor)
def process_train_iclr2022_gpt_4o_test_iclr2022_gpt4o_with_anchor(data, config) -> Dict[str, Dict[str, List[Any]]]:
    f_train = '~/my_datasets/train.100.iclr2022.gpt-4o.with_anchor.csv'
    f_test  = '~/my_datasets/test.500.iclr2022_gpt-4o.with_anchor.csv'
    return {"train": read_csv(f_train), "test": read_csv(f_test)}