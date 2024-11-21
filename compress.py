import zlib
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import torch
import pickle


class ICompressor:
    # Define the compression and decompression methods
    def compress(self, text: str) -> Tuple[float, bytes]:
        pass

    def decompress(self, compressed_bytes: bytes) -> str:
        pass


class ZLibCompressor(ICompressor):
    def compress(self, text: str) -> Tuple[float, bytes]:
        compressed_data = zlib.compress(text.encode('utf-8'))
        compression_ratio = len(compressed_data) / len(text.encode('utf-8')) if text else 1.0
        return compression_ratio, compressed_data

    def decompress(self, compressed_bytes: bytes) -> str:
        decompressed_data = zlib.decompress(compressed_bytes).decode('utf-8')
        return decompressed_data


N_TOKENS = 20


def get_ranks_from_text(input_text, model, tokenizer):
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        add_special_tokens=False,
        padding=False,
        truncation=True,
        return_token_type_ids=False,
    )
    input_ids = inputs['input_ids'].to(model.device)
    ranks = [input_ids[0][0].cpu().tolist()]
    for i in tqdm(range(0, input_ids.shape[1] - 1)):
        start_pos = max(0, i - N_TOKENS + 1)
        cur_input_ids = input_ids[:, start_pos:i + 1]
        with torch.no_grad():
            outputs = model(cur_input_ids)
        logits = outputs.logits
        next_token_logits = logits[0, i - start_pos, :]
        sorted_indices = torch.argsort(next_token_logits, descending=True)
        actual_token_id = input_ids[0, i + 1]
        rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
    return ranks


def batched_get_ranks(input_text, model, tokenizer):
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        add_special_tokens=False,
        padding=False,
        truncation=True,
        return_token_type_ids=False,
    )
    input_ids = inputs['input_ids'].to(model.device)
    BS = 8

    ranks = [input_ids[0][0].cpu().tolist()]
    print(input_ids.shape)
    for i in tqdm(range(0, input_ids.shape[1] - 1, BS)):
        cur_input_ids = []
        for j in range(i, min(input_ids.shape[1] - 1, i + BS)):
            start_pos = max(0, j - N_TOKENS + 1)
            cur_input_ids.append(input_ids[:, start_pos:j + 1].flatten())
        max_length = max(len(x) for x in cur_input_ids)
        attention_mask = torch.ones(len(cur_input_ids), max_length, device=model.device)
        for j, x in enumerate(cur_input_ids):
            attention_mask[j, len(x):] = 0
        cur_input_ids = [torch.concat([x, torch.zeros(max_length - len(x), device=model.device, dtype=torch.long)]) for x in cur_input_ids]
        cur_input_ids = torch.stack(cur_input_ids)
        with torch.no_grad():
            outputs = model(cur_input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        for j in range(i, min(input_ids.shape[1] - 1, i + BS)):
            start_pos = max(0, j - N_TOKENS + 1)
            next_token_logits = logits[j - i, j - start_pos, :]
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            actual_token_id = input_ids[0, j + 1]
            rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)

    return ranks

def compress_ranks(ranks):
    return zlib.compress(pickle.dumps(ranks))


def get_text_from_ranks(ranks, model, tokenizer):
    gen_input_ids = torch.tensor([[ranks[0]]]).cuda()
    for i, rank in tqdm(enumerate(ranks[1:])):
        start_pos = max(0, i - N_TOKENS + 1)
        with torch.no_grad():
            outputs = model(gen_input_ids[:, start_pos:i + 1])
        logits = outputs.logits
        next_token_logits = logits[0, i - start_pos, :]
        sorted_indices = torch.argsort(next_token_logits, descending=True)
        token_num = sorted_indices[ranks[i + 1]].cpu().tolist()
        gen_input_ids = torch.cat((gen_input_ids, torch.tensor([[token_num]]).cuda()), dim=1)
    return tokenizer.batch_decode(gen_input_ids)[0]


def decompress_ranks(data):
    return pickle.loads(zlib.decompress(data))


class LLAmaCompressor(ICompressor):
    def __init__(self, model_name: str = 'tartuNLP/Llama-2-7b-Ukrainian'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype="float16"
        )
        self.model.eval()

    def compress(self, text: str) -> Tuple[float, bytes]:
        ranks = get_ranks_from_text(text, self.model, self.tokenizer)
        data = compress_ranks(ranks)
        return len(data) / len(text), data
