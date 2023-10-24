import torch
from torchtext.legacy import data, datasets
import torch.nn as nn
import spacy

nlp = spacy.load('en_core_web_sm')

# Define the TEXT field
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_) # Build the vocabulary for the TEXT field
LABEL.build_vocab(train_data) 

class RNN(nn.Module):
    def __init__(self, word_limit, dimension_embedding, dimension_hidden, dimension_output, num_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(word_limit, dimension_embedding, padding_idx=pad_idx)
        self.rnn = nn.GRU(dimension_embedding, dimension_hidden, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(dimension_hidden * 2, dimension_output)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, len_txt):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

class SentimentAnalysisModel:
    def __init__(self, model_path, device, word_limit, dimension_embedding, dimension_hidden, dimension_output, num_layers, 
                 bidirectional, dropout, pad_idx):
        self.model = RNN(
            word_limit, dimension_embedding, dimension_hidden, dimension_output, num_layers, bidirectional, dropout, pad_idx
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict_sentiment(self, text, len_txt):
        with torch.no_grad():
            predictions = self.model(text, len_txt).squeeze(1)
        return predictions

dimension_input = len(TEXT.vocab)
dimension_embedding = 100  
dimension_hddn = 128  
dimension_out = 1 
layers = 2  
bidirectional = True  
dropout = 0.5  
idx_pad = TEXT.vocab.stoi[TEXT.pad_token] 

def get_model():
    model_path = 'tut2-model.pt'  # Update with the actual path to your model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return SentimentAnalysisModel(model_path, device, dimension_input, dimension_embedding, dimension_hddn, dimension_out, layers, bidirectional, dropout, idx_pad)

def preprocess(review):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenized = [tok.text for tok in nlp.tokenizer(review)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    return tensor

def length(review):
    tokenized = [tok.text for tok in nlp.tokenizer(review)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    length_tensor = torch.LongTensor(length)
    return length_tensor