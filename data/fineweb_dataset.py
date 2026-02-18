import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset 
from transformers import AutoTokenizer

class FineWebDataset(IterableDataset):
    def __init__(self,tokenizer=None,seq_len = 1024,split = "train",):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer  = AutoTokenizer.from_pretrained("gpt2") if tokenizer is None else tokenizer
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=split,
            streaming=True
        )
    def __iter__(self):
        buffer = []

        for doc in self.dataset:
            #tokenize + append eos
            tokens = self.tokenizer.encode(doc["text"]) + [self.tokenizer.eos_token_id]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len+1:
                chunk = buffer[:self.seq_len+1]
                buffer = buffer[self.seq_len+1:]
                x = torch.tensor(chunk[:-1],dtype=torch.long)
                y = torch.tensor(chunk[1:],dtype=torch.long)

                yield x,y