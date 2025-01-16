from importlib.metadata import version
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import re

class SimpleTokenizer:
    def __init__ (self,path):
        self.path=path
        self.vocab=None
        self.generate_vocab()
        self.str_to_int = self.vocab
        self.int_to_str = {i:s for s,i in self.vocab.items()}
        
    def generate_vocab(self):
        raw_text=self.generate_rawtext()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_words = sorted(set(preprocessed))
        self.vocab = {token:integer for integer,token in enumerate(all_words)}

    def generate_rawtext(self):
        with open(self.path,'r',encoding='utf-8') as file:
            raw_text=file.read()
            return raw_text
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text
    
costum_tokenizer = SimpleTokenizer("C:\\Users\\Balu\\OneDrive\\Desktop\\GPT-2_Project\\data\\the-verdict.txt")


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=False,
                         num_workers=0):

    # Initialize the tokenizer
    #tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer=costum_tokenizer

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
    

with open("C:\\Users\\Balu\\OneDrive\\Desktop\\GPT-2_Project\\data\\the-verdict.txt",'r',encoding='utf-8') as file:
    raw_text=file.read()
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)