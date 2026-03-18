import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset 
import tiktoken

class FineWebDataset(IterableDataset):
    def __init__(self,tokenizer=None,seq_len = 1024,split = "train", data_path=None):
        super().__init__()
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding("gpt2") if tokenizer is None else tokenizer
        if data_path:
            # Load from local Arrow files
            self.dataset = load_dataset(
                "arrow",
                data_files=f"{data_path}/*.arrow",
                split="train",
                streaming=True
            )
        else:
            # Fallback to HF streaming
            self.dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split=split,
                streaming=True
            )
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iterator = self.dataset
        else:
            # Multi-process data loading
            # Split the dataset into N parts (where N = num_workers)
            # Each worker gets a unique part based on its worker_id
            iterator = self.dataset.shard(
                num_shards= worker_info.num_workers,
                index=worker_info.id            )
        buffer = []

        for doc in iterator:
            tokens = self.tokenizer.encode_ordinary(doc["text"]) + [self.tokenizer.eot_token]
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len +1:
                chunk = buffer[:self.seq_len+1]
                buffer = buffer[self.seq_len+1:]
                x = torch.tensor(chunk[:-1],dtype=torch.long)
                y = torch.tensor(chunk[1:],dtype=torch.long)
                yield x,y