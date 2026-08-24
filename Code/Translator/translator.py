import torch
import torch.nn as nn
import os
from io import open
import numpy as np
import math
import jieba3
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

nlp_en = spacy.load("en_core_web_lg")
nlp_zh = spacy.load("zh_core_web_lg")

seq_len = 64
emb_dim = 300
batch_size = 64
cjk_ranges = [
        ( 0x4E00,  0x62FF),
        ( 0x6300,  0x77FF),
        ( 0x7800,  0x8CFF),
        ( 0x8D00,  0x9FCC),
        ( 0x3400,  0x4DB5),
        (0x20000, 0x215FF),
        (0x21600, 0x230FF),
        (0x23100, 0x245FF),
        (0x24600, 0x260FF),
        (0x26100, 0x275FF),
        (0x27600, 0x290FF),
        (0x29100, 0x2A6DF),
        (0x2A700, 0x2B734),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F)
    ]

sp_dict = {'<SOS>': torch.cat((torch.Tensor([1]), torch.zeros(emb_dim-1))).to(device),
                    '<EOS>': torch.cat((torch.zeros(1), torch.Tensor([1]),torch.zeros(emb_dim-2))).to(device),
                    '<PAD>': torch.zeros(emb_dim).to(device)}

Data:list
zh_vocab: list
en_vocab: list
en_dict: dict
zh_dict: dict
encode_dict: dict
train_dataset: Dataset
dataloader: DataLoader
zh_vocab_size: int

def is_zh(char):
    char = ord(char)
    for bottom, top in cjk_ranges:
        if bottom <= char <= top:
            return True
    return False

def test_zh(line):  # test if line contains at least one Chinese character
    for char in line:
        if is_zh(char):
            return True
    return False

def prepare_data(path, seq_len):
    with open(path, 'r', encoding='utf8') as f:
        data = f.readlines()
    data = [line.strip().split('\t') for line in data]
    data = [line for line in data if len(line) == 2 and test_zh(line[1]) and (not test_zh(line[0]))]
    en_tokens = [[word.text.lower() for word in nlp_en.tokenizer(line[0])] for line in data]
    zh_tokens = [[word for word in jieba3.jieba3(use_hmm=False).cut_text(line[1])] for line in data]
    out = [line for line in zip(en_tokens, zh_tokens) \
           if len(line[0]) < seq_len and len(line[1]) < seq_len - 2]
    return out

def zh_vocab(data):
    vocab = sorted(set(word for line in data for word in line[1]))
    vocab = ["<PAD>", "<SOS>", "<EOS>"] + vocab
    return vocab

def en_vocab(data):
    vocab = sorted(set(word for line in data for word in line[0]))
    return vocab

def emb_dict(vocab, lang):
    if lang == "en":
        dict = {word: nlp_en(word).vector for word in tqdm(vocab)}
        return dict
    if lang == "zh":
        dict = {word: nlp_zh(word).vector for word in tqdm(vocab)}
        return dict
    return None

def padding(line, seq_len):
    line = line + ['<PAD>'] * (seq_len - len(line))
    return line

def embed_word(word, lang, zh_dict, en_dict):
    if word in ["<SOS>", "<EOS>", "<PAD>"]:
        return sp_dict[word]
    if lang == "zh":
        return torch.Tensor(zh_dict[word]).to(device)
    if lang == "en":
        return torch.Tensor(en_dict[word]).to(device)
    return None

class TranslationDataset(Dataset):
    def __init__(self, data, zh_dict, en_dict, encode_dict):
        super(Dataset).__init__()
        self.data = data
        self.zh_dict = zh_dict
        self.en_dict = en_dict
        self.encode_dict = encode_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedded_input = torch.stack([embed_word(word, "en", self.zh_dict, self.en_dict) for word in padding(self.data[idx][0], seq_len)]).to(device)
        embedded_target = torch.stack([embed_word(word, "zh", self.zh_dict, self.en_dict) for word in padding(["<SOS>"] + self.data[idx][1], seq_len)]).to(device)
        output = torch.LongTensor([self.encode_dict[word] for word in padding(self.data[idx][1] + ["<EOS>"], seq_len)]).to(device)
        return embedded_input, embedded_target, output

def custom_collate(batch):
    input = torch.stack([line[0] for line in batch]).to(device)
    target = torch.stack([line[1] for line in batch]).to(device)
    out = torch.stack([line[2] for line in batch]).to(device)
    return [input, target, out]

class Transformer(nn.Module):
    def __init__(self, trg_vocab_size, emb_dim):
        super().__init__()
        self.positional_encoder = PositionalEncoder()
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=4, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(emb_dim, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, target, trg_mask, src_padding_mask, trg_padding_mask):
        src_input = self.positional_encoder(input) # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(target)
        output = self.transformer(src_input, trg_input, tgt_mask=trg_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=trg_padding_mask) # (B, L, d_model) => # (B, L, trg_vocab_size)
        output = self.softmax(self.fc(output))
        return output

class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Make initial positional encoding matrix with 0
        pe_matrix= torch.zeros(seq_len, emb_dim) # (L, d_model)
        # Calculating position encoding values
        for pos in range(seq_len):
            for i in range(emb_dim):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / emb_dim)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / emb_dim)))
        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        self.positional_encoding = pe_matrix.to(device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(emb_dim) # (B, L, d_model)
        x = x + self.positional_encoding # (B, L, d_model)
        return x

def look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).to(torch.bool).to(device)
    return mask

def padding_mask(lines):
    return torch.BoolTensor([[torch.equal(word, sp_dict["<PAD>"]) for word in line] for line in lines]).to(device)

def setup(path, seq_len, train:bool, load:bool):
    global Data, zh_vocab, en_vocab, en_dict, zh_dict, encode_dict, train_dataset, dataloader, zh_vocab_size
    if train:
        Data = prepare_data(path, seq_len)
    if load:
        zh_vocab = torch.load(os.getcwd() + "/zh_vocab.pt", weights_only=False)
        en_dict = torch.load(os.getcwd() + "/en_dict.pt", weights_only=False)
        zh_dict = torch.load(os.getcwd() + "/zh_dict.pt", weights_only=False)
        encode_dict = torch.load(os.getcwd() + "/encode_dict.pt", weights_only=False)
    else:
        zh_vocab = zh_vocab(Data)
        torch.save(zh_vocab, os.getcwd() + "/zh_vocab.pt")
        en_vocab = en_vocab(Data)
        en_dict = emb_dict(en_vocab, "en")
        torch.save(en_dict, os.getcwd() + "/en_dict.pt")
        zh_dict = emb_dict(zh_vocab, "zh")
        torch.save(zh_dict, os.getcwd() + "/zh_dict.pt")
        encode_dict = {word: zh_vocab.index(word) for word in tqdm(zh_vocab)}
        torch.save(encode_dict, os.getcwd() + "/encode_dict.pt")
    if train:
        train_dataset = TranslationDataset(Data, zh_dict, en_dict, encode_dict)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True)
    zh_vocab_size = len(zh_vocab)
    return None

def train(epoches, dataloader, model, loss, optimizer):
    for epoch in range(epoches):
        print('Epoch:', epoch+1)
        train_losses = []
        for data in tqdm(dataloader):
            input = data[0]
            target = data[1]
            output = data[2]
            out = model(input, target, look_ahead_mask(seq_len), padding_mask(input), padding_mask(target))
            optimizer.zero_grad()
            l = loss(out.view(-1, zh_vocab_size), output.view(-1))
            l.backward()
            optimizer.step()
            train_losses.append(l.item())
        mean_train_loss = np.mean(train_losses)
        print('Mean Loss:', mean_train_loss)
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training Done!")

def inference_greedy(model, line):
    infer_data = [[[word.text.lower() for word in nlp_en.tokenizer(line)], []]]
    infer_dataset = TranslationDataset(infer_data, zh_dict, en_dict, encode_dict)
    result = []
    for idx in range(seq_len):
        output = model(infer_dataset[0][0].unsqueeze(0), infer_dataset[0][1].unsqueeze(0),
                look_ahead_mask(seq_len), padding_mask(infer_dataset[0][0].unsqueeze(0)),
                padding_mask(infer_dataset[0][1].unsqueeze(0)))
        output = torch.argmax(output, dim=-1)
        if output[0][idx] == 2:
            break
        else:
            result.append(zh_vocab[output[0][idx]])
            infer_data[0][1].append(zh_vocab[output[0][idx]])
            infer_dataset = TranslationDataset(infer_data, zh_dict, en_dict, encode_dict)
    return '"' + line + '"' + "\n" + '"' + ''.join(result) + '"'

class Node:
    def __init__(self, words, prob):
        self.words = words
        self.prob = prob

    def update(self, word, prob):
        prob = (self.prob * pow(len(self.words), 0.75) + prob) / pow((len(self.words) + 1), 0.75)
        out = Node(self.words + [word], prob)
        return out

    def __eq__(self, other):
        return self.prob == other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __gt__(self, other):
        return self.prob > other.prob


def inference_beam(model, line, beam_size):
    infer_data = [[[word.text.lower() for word in nlp_en.tokenizer(line)], []]]
    root = Node([], 0)
    candidates = [root]
    for idx in range(seq_len):
        next_candidates = []
        for node in candidates:
            if len(node.words) > 0 and node.words[-1] == "<EOS>":
                next_candidates.append(node)
                continue
            else:
                infer_data[0][1] = node.words
                infer_dataset = TranslationDataset(infer_data, zh_dict, en_dict, encode_dict)
                output = model(infer_dataset[0][0].unsqueeze(0), infer_dataset[0][1].unsqueeze(0),
                            look_ahead_mask(seq_len), padding_mask(infer_dataset[0][0].unsqueeze(0)),
                            padding_mask(infer_dataset[0][1].unsqueeze(0)))
                output = torch.topk(output[0][idx], beam_size)
                for prob, word_idx in zip(output[0], output[1]):
                    next_candidates.append(node.update(zh_vocab[word_idx], prob))
        candidates = sorted(next_candidates)[-beam_size:]
        if all([node.words[-1] == "<EOS>" for node in candidates]):
            break
    out = max(candidates).words
    if out[-1] == "<EOS>":
        out = out[:-1]
    return '"' + line + '"' + "\n" + '"' + ''.join(out) + '"'

def start_train(eps):
    setup(os.getcwd() + "/news-commentary-v14.en-zh.tsv", seq_len, train=True, load=True)
    model = Transformer(zh_vocab_size, emb_dim).to(device)
    loss = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.train()
    train(eps, dataloader, model, loss, optimizer)
    return None

def start_infer(line, beam_size):
    setup(os.getcwd() + "/news-commentary-v14.en-zh.tsv", seq_len, train=False, load=True)
    model = Transformer(zh_vocab_size, emb_dim).to(device)
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.eval()
    return inference_beam(model, line, beam_size)



# start_train(10)
print(start_infer("As you can see, we have height and width at each block.", 10))
