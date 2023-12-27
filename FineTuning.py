import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import requests
from tqdm import tqdm

######################################################## Parameter ###############################################################################
sentence_max_length = 128
batch_size = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Pooler = True

######################################################## Parameter ###############################################################################

######################################################## Class & Function #########################################################################
def max_len(str_list):
    m_len = 0
    index = 0
    len_list = []
    for i, con in enumerate(str_list):
        if len(con) > m_len:
            m_len = len(con)
            index = i
        len_list.append(len(con))
    len_list.sort()
    print(len_list)
    # print(len_list[int(len(len_list)*0.9)])
    exit()
    return m_len, index


def data_process(name, tag):
    if tag == 'train':
        df = pd.read_csv(name)
        df = df.dropna(axis=0)
        train_sample = int(len(df) * 0.8)
        df_train = df[:train_sample]
        con_que = df_train.iloc[:, [0, 1]].values.tolist()

        train_data = []  # [[C1, C2,...,Cn, Q],...]
        temp = []

        current_question = con_que[0][1]

        for con, que in con_que:
            if current_question == que and type(con) != float:
                temp.append(con)
            else:
                train_data.append([list(temp), current_question])
                temp.clear()
                temp.append(con)
                current_question = que
        del con_que

        # df_valid = df[train_sample:train_sample+val_sample]
        df_valid = df[train_sample:]
        con_que = df_valid.iloc[:, [0, 1]].values.tolist()

        valid_data = []  # [[C1, C2,...,Cn, Q],...]
        temp = []

        current_question = con_que[0][1]

        for con, que in con_que:
            if current_question == que and type(con) != float:
                temp.append(con)
            else:
                valid_data.append([list(temp), current_question])
                temp.clear()
                temp.append(con)
                current_question = que

        return train_data, valid_data

    elif tag == 'test':
        df_test = pd.read_csv(name)
        df_test = df_test.dropna(axis=0)
        con_sent_que = df_test.iloc[:, [0, 1, 2]].values.tolist()
        test_con_sent = []  # [[C1, C2,...,Cn, Q],...]
        test_question = []
        temp = []

        current_question = con_sent_que[0][2]
        current_context = con_sent_que[0][0]
        num = 0
        for con, sent, que in con_sent_que:
            if current_question == que and type(sent) != float:
                temp.append(sent)
            else:
                num += 1
                test_con_sent.append([current_context, list(temp), num])
                test_question.append([current_question, num])
                temp.clear()
                temp.append(sent)
                current_question = que
                current_context = con

        return test_con_sent, test_question


def Reader(contexts, query):
    messages = [{"role": "contex", "content": context} for context in contexts]
    messages.append({"role": "user", "content": query})

    input_q = json.dumps({
        "model": "t5-small-fid", "mode": "",
        "messages": messages})

    response = requests.post(f'http://211.39.140.48:9090/predictions/temp', data=input_q)
    print(response.json().get('message'))


class SiameseNetwork(nn.Module):
    def __init__(self, Pooler):
        super(SiameseNetwork, self).__init__()
        self.kobert = kobert_model
        self.cosine_similarity = F.cosine_similarity
        self.Pooler = Pooler

    def forward(self, x1, x2):
        if Pooler == True:
            context_out = self.kobert(inputs_embeds=x1, token_type_ids=None).pooler_output
            question_out = self.kobert(inputs_embeds=x2, token_type_ids=None).pooler_output

        elif Pooler == False:
            context_out = self.kobert(inputs_embeds=x1, token_type_ids=None).last_hidden_state.view(-1,
                                                                                                    sentence_max_length * 768)
            question_out = self.kobert(inputs_embeds=x2, token_type_ids=None).last_hidden_state.view(-1,
                                                                                                     sentence_max_length * 768)

        return context_out, question_out


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 10
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')
        # torch.save(model.state_dict(), self.path)
        if Pooler == True:
            name = '/data/workspace/psh/model/Pooler/' + str(sentence_max_length) + '_short_KoBERT_Siamese.pt'
        elif Pooler == False:
            name = '/data/workspace/psh/model/NoPooler/' + str(sentence_max_length) + '_short_KoBERT_Siamese.pt'
        torch.save(model, name)
        self.val_loss_min = val_loss


######################################################## Class & Function #########################################################################

######################################################## Fine-tuning ##############################################################################
'''
input key의 구성
    input_ids : token의 id 리스트
    token_type_ids : 문장이 어디에 속해 있는지 표시 (Sentence Embedding)
    attention_mask : attention이 수행되야할 token=1 padding과 같이 필요 없으면 0
output key의 구성
    last_hidden_state : 마지막 layer의 hidden_state로 BERT의 최종 임베딩 / [1,token 개수,768] 각 token에 대한 768차원의 임베딩 벡터들의 집합
'''

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
if Pooler == True:
    kobert_model = BertModel.from_pretrained('skt/kobert-base-v1', add_pooling_layer=True,
                                             output_hidden_states=False).to(device)
elif Pooler == False:
    kobert_model = BertModel.from_pretrained('skt/kobert-base-v1', add_pooling_layer=False,
                                             output_hidden_states=False).to(device)

train_data_name = '/data/workspace/psh/data/Preprocess/' + str(sentence_max_length) + '/' + str(
    sentence_max_length) + '_Preprocess_data.csv'
training_data, validation_data = data_process(train_data_name, 'train')


# Convert sentences to BERT embeddings with max pooling
def get_kobert_embedding(sentences, tag):
    if tag == 'C':  # Context 내 문장 집합인 경우
        sentence_embeddings = torch.zeros(1, sentence_max_length, 768)
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(
                device)
            outputs = kobert_model(**inputs)
            sentence_embeddings += outputs.last_hidden_state.detach().cpu()
        mean_embedding = sentence_embeddings / len(sentences)

        return mean_embedding.squeeze()
    elif tag == 'Q':
        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(
            device)
        outputs = kobert_model(**inputs)

        return outputs.last_hidden_state.squeeze().detach().cpu()


# Prepare training pairs and labels
# train_pairs = [(get_kobert_embedding(pair[0],'C'), get_kobert_embedding(pair[1],'Q')) for pair in tqdm(training_data)]
# torch.save(train_pairs,'/data/workspace/psh/Embedding/'+str(sentence_max_length)+'_Train_Embedding_saved.pt')
train_embedd_name = '/data/workspace/psh/Embedding/Train/' + str(sentence_max_length) + '_Train_Embedding_saved.pt'
train_pairs = torch.load(train_embedd_name)
train_pairs = train_pairs[:800]
train_dataset = TensorDataset(torch.stack([pair[0] for pair in train_pairs]),
                              torch.stack([pair[1] for pair in train_pairs]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_pairs = [(get_kobert_embedding(pair[0],'C'), get_kobert_embedding(pair[1],'Q')) for pair in tqdm(validation_data)]
# torch.save(val_pairs,'/data/workspace/psh/Embedding/'+str(sentence_max_length)+'_Valid_Embedding_saved.pt')
valid_embedd_name = '/data/workspace/psh/Embedding/Train/' + str(sentence_max_length) + '_Valid_Embedding_saved.pt'
val_pairs = torch.load(valid_embedd_name)
val_pairs = val_pairs[:100]
val_dataset = TensorDataset(torch.stack([pair[0] for pair in val_pairs]), torch.stack([pair[1] for pair in val_pairs]))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print("Train, Validation embedding vector loaded")

Siamese_model = SiameseNetwork(Pooler).to(device)
optimizer = torch.optim.AdamW(Siamese_model.parameters(), lr=1e-5)
early_stopping = EarlyStopping(patience=5, verbose=True)
cosine_similarity = nn.CosineSimilarity()
num_epochs = 10

for epoch in range(num_epochs):
    Siamese_model.train()
    train_total_loss = 0
    val_total_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description("Train")
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            context_out, question_out = Siamese_model(x, y)

            loss = torch.mean(1 - cosine_similarity(context_out, question_out))
            train_total_loss += loss.item()

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

    Siamese_model.eval()
    with tqdm(val_loader, unit="batch") as vepoch:
        with torch.no_grad():
            for batch in vepoch:
                vepoch.set_description("Valid")
                x, y = batch
                x, y = x.to(device), y.to(device)

                context_out, question_out = Siamese_model(x, y)

                loss = torch.mean(1 - cosine_similarity(context_out, question_out))
                val_total_loss += loss.item()

                vepoch.set_postfix(loss=loss.item())

    train_average_loss = train_total_loss / len(train_loader)
    val_average_loss = val_total_loss / len(val_loader)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train average Loss: {train_average_loss:.8f}, Validation average Loss: {val_average_loss:.8f}")
    early_stopping(val_average_loss, Siamese_model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

if Pooler == True:
    name = '/data/workspace/psh/model/Pooler/' + str(sentence_max_length) + '_short_KoBERT_Siamese_Final.pt'
elif Pooler == False:
    name = '/data/workspace/psh/model/NoPooler/' + str(sentence_max_length) + '_short_KoBERT_Siamese_Final.pt'

torch.save(Siamese_model, name)

######################################################## Fine-tuning ##############################################################################


