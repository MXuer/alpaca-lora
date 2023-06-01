import os
import sys
import yaml
import json

from datasets import load_dataset

class DatasetPreporcess:
    def __init__(self, path) -> None:
        if path.endswith(".json") or path.endswith(".jsonl") and os.path.exists(path):
            self.data = load_dataset("json", data_files=path)
        else:
            self.data = load_dataset(path)
        if 'train' in self.data.keys():
            self.data = self.data['train']

        self.remove_col_list = [col for col in self.data.features if col not in ["conversations"]]

    def map2template(self, data_point):
        raise NotImplementedError
    
    def preprocess(self, select_num=None):
        if select_num is not None:
            self.data = self.data.select(range(select_num)).map(self.map2template)
        else:
            self.data = self.data.map(self.map2template)
        self.data = self.data.remove_columns(self.remove_col_list)
        return self.data

    def filter(self, data_point):
        """
        only keep three key: input, instruction, output
        """
        if self.normalize_dict["input"] != "":
            input_ = data_point[self.normalize_dict['input']]
        else:
            input_ = ""
        instruction = data_point[self.normalize_dict['instruction']]
        output = data_point[self.normalize_dict['output']]
        return {"input": input_, "instruction": instruction, "output": output}

    @property
    def show(self):
        example = json.dumps(self.data[0], indent=2, ensure_ascii=False)
        print(example)

class AlpacaPreprocess(DatasetPreporcess):
    def __init__(self, path="yahma/alpaca-cleaned"):
        super().__init__(path)

    def map2template(self, data_point):
        input_ = data_point['input']
        instruction = data_point['instruction']
        output = data_point['output']
        if input_:
            instruction = f"{instruction}\n{input_}" 
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point


class DollyPreprocess(DatasetPreporcess):
    def __init__(self, path="databricks/databricks-dolly-15k") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        input_ = data_point['context']
        instruction = data_point['instruction']
        output = data_point['response']
        if input_:
            instruction = f"{instruction}\n{input_}" 
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point


class BelleGroup3_5MPreprocess(DatasetPreporcess):
    def __init__(self, path="BelleGroup/train_3.5M_CN") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        conversations = []
        for index in range(0, len(data_point['conversations']), 2):
            conversations.append(
                {
                    "user": data_point['conversations'][index]['value'],
                    "doer": data_point['conversations'][index+1]['value']
                }
            )
        data_point = {
            "conversations": conversations
        }
        return data_point
    
    
class FireflyPreprocess(DatasetPreporcess):
    def __init__(self, path="YeungNLP/firefly-train-1.1M") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        instruction = data_point['input']
        output = data_point['target']
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point

class BelleGroup2MPreprocess(DatasetPreporcess):
    def __init__(self, path="BelleGroup/train_2M_CN") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        input_ = data_point['input']
        instruction = data_point['instruction']
        output = data_point['output']
        if input_:
            instruction = f"{instruction}\n{input_}" 
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point
    
class FinanceAlpacaPreprocess(DatasetPreporcess):
    def __init__(self, path="gbharti/finance-alpaca") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        input_ = data_point['input']
        instruction = data_point['instruction']
        output = data_point['output']
        if input_:
            instruction = f"{instruction}\n{input_}" 
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point
    

class OpenAsistantOasst1Preprocess(DatasetPreporcess):
    def __init__(self, path="OpenAssistant/oasst1") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        return data_point


class ChineseVicunaPreprocess(DatasetPreporcess):
    def __init__(self, path="Chinese-Vicuna/instruct_chat_50k.jsonl") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        conversations = []
        inputs = data_point['input']
        outputs = data_point['output']

        for index, input_ in enumerate(inputs):
            output = outputs[index]
            conversations.append(
                {
                    "user": input_,
                    "doer": output
                }
            )
        data_point = {
            "conversations": conversations
        }
        return data_point
    

class DataOceanPreprocess(DatasetPreporcess):
    def __init__(self, path="dataocean.json") -> None:
        super().__init__(path)

    def map2template(self, data_point):
        instruction = data_point['instruction']
        output = data_point['output']
        data_point = {
            "conversations":[
                {
                "user": instruction,
                "doer": output
                }
            ]
        }
        return data_point

all_dataset_class = [
    AlpacaPreprocess, 
    DollyPreprocess, 
    FinanceAlpacaPreprocess, 
    BelleGroup2MPreprocess, 
    BelleGroup3_5MPreprocess, 
    DataOceanPreprocess, 
    ChineseVicunaPreprocess
]

if __name__=="__main__":
    dp = DataOceanPreprocess()
    dp.show
    dp_data =  dp.preprocess(2)
    print(dp_data[1])
