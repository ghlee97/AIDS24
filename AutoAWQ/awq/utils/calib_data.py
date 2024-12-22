import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    #testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    #testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    text_list = []
    for i in range(len(trainloader)):
        text_list += [{"text": tokenizer.decode(trainloader[i][0][0])}]

    from datasets import Dataset
    dataset = Dataset.from_list(text_list)

    return dataset  #, testenc




def get_c4(nsamples, seed, seqlen, tokenizer):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )


    #tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen: ####################
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    text_list = []
    for i in range(len(trainloader)):
        text_list += [{"text": tokenizer.decode(trainloader[i][0][0])}]

    from datasets import Dataset
    dataset = Dataset.from_list(text_list)
        

    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(valdata) - 1)
    #         tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)

    #return trainloader, valenc 
    return dataset  #, testenc



def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            dataset = dataset.shuffle(seed=42)
        elif data == "wikitext2":
            #dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            dataset = get_wikitext2(nsamples=n_samples, seed=42, seqlen=max_seq_len, tokenizer=tokenizer)
        elif data == "c4":
            dataset = get_c4(nsamples=n_samples, seed=42, seqlen=max_seq_len, tokenizer=tokenizer)
        else:
            dataset = load_dataset(data, split=split)
            dataset = dataset.shuffle(seed=42)
        #dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    #import pdb; pdb.set_trace()
    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]
