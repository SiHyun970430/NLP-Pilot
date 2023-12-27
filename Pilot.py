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
from torch.nn.functional import normalize
from torch.nn.functional import cosine_similarity

'''
df = pd.read_parquet('/data/workspace/psh/data/klue/validation/0000.parquet')
print(df.columns)
df_qa = df[['context','question','answers']]
df_qa.to_csv('Validation_data.csv',index=False)
'''
######################################################## Parameter ###############################################################################
sentence_max_length = 64
train_sample = 32
valid_sample = 32
batch_size = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Pooler = False
######################################################## Parameter ###############################################################################

######################################################## Class & Function #########################################################################
def max_len(str_list):
    m_len = 0
    index = 0
    len_list = []
    for i,con in enumerate(str_list):
        if len(con)>m_len:
            m_len = len(con)
            index=i
        len_list.append(len(con))
    len_list.sort()
    print(len_list)
    #print(len_list[int(len(len_list)*0.9)])
    exit()
    return m_len,index
def data_process(name,tag):
    if tag == 'train':
        df = pd.read_csv(name)
        df = df.dropna(axis=0)
        train_sample = int(len(df)*0.8)
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

        #df_valid = df[train_sample:train_sample+val_sample]
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
        con_sent_que = df_test.iloc[:, [0,1,2,3]].values.tolist() # [context, sentence, question, answer]
        test_con_sent = []  # [[C1, C2,...,Cn, Q],...]
        test_que_ans = []
        temp = []

        current_question = con_sent_que[0][2]
        current_answer = con_sent_que[0][3]
        current_context= con_sent_que[0][0]
        num = 0
        for con,sent, que,ans in con_sent_que:
            if current_question == que and type(sent) != float:
                temp.append(sent)
            else:
                num+=1
                test_con_sent.append([current_context,list(temp),num])
                test_que_ans.append([current_question,num,current_answer])
                temp.clear()
                temp.append(sent)
                current_question = que
                current_answer = ans
                current_context = con

        return test_con_sent, test_que_ans


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
            context_out = self.kobert(inputs_embeds=x1,token_type_ids=None).last_hidden_state.view(-1,sentence_max_length*768)
            question_out = self.kobert(inputs_embeds=x2,token_type_ids=None).last_hidden_state.view(-1,sentence_max_length*768)


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
        #torch.save(model.state_dict(), self.path)
        if Pooler == True:
            name = '/data/workspace/psh/model/Pooler/' + str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'
        elif Pooler==False:
            name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'
        torch.save(model, name)
        self.val_loss_min = val_loss

def pearson_similarity(x, y):
    # 평균을 0으로 만들기 위해 중심화
    x = x - torch.mean(x)
    y = y - torch.mean(y)

    # 피어슨 유사도 계산
    numerator = torch.sum(x * y)
    denominator = torch.sqrt(torch.sum(x**2) * torch.sum(y**2))

    # 0으로 나누기를 방지하기 위해 작은 값을 더해줌
    epsilon = 1e-8
    similarity = numerator / (denominator + epsilon)

    return similarity
######################################################## Class & Function #########################################################################

######################################################## Preprocess ##############################################################################

'''
# with open('KorQuAD_v1.0_dev.json', 'r') as f:
#     test_data = json.load(f)
# test_data = [item for topic in test_data['data'] for item in topic['paragraphs'] ]
# test_data = test_data[:100]
# context_question = []
# for data in test_data:
#     context = data.get('context')
#     question = data.get('qas')[0].get('question')
#     answer = data['qas'][0]['answers'][0]['text']
#
#     context_question.append([context, question,answer])

df = pd.read_csv('/data/workspace/psh/data/Extracted/Test_data.csv')
context_question = df[['context','question','answers']].values.tolist()
cleared_sentences = []
for context, question,answers in context_question:
    answers = list(filter(None, list(answers.split('array([').pop().split(']')[0].split("'"))))
    answer = answers[0]
    sentence_list = context.split('.')
    sentence_list.pop()
    for sentence in sentence_list:
        if '\n' in sentence:                # \n 들어있는 문장 슬라이싱
            sliced = sentence.split('\n')
            sliced = list(filter(None, sliced))         # '' 공백인 원소 제거
            for element in sliced:
                if element != ' ':                      # ' ' 띄어쓰기만 있는 원소 제거
                    if len(element) > sentence_max_length:
                        div = round(len(element) / (int(len(element) / sentence_max_length) + 1)) + 1
                        splitted = [element[i:i + div] for i in range(0, len(element), div)]
                        for split in splitted:
                            #cleared_sentences.append([split,question])

                            cleared_sentences.append([context,split, question,answer]) # Test의 경우 context를 포함해야해서 앞에 context 포함

                else:
                        #cleared_sentences.append([element,question])
                        cleared_sentences.append([context,element, question,answer])   # Test의 경우 context를 포함해야해서 앞에 context 포함

        else:
            if len(sentence) > sentence_max_length:
                div = round(len(sentence) / (int(len(sentence) / sentence_max_length) + 1)) + 1
                splitted = [sentence[i:i + div] for i in range(0, len(sentence),div)]
                for split in splitted:
                    #cleared_sentences.append([split,question])
                    cleared_sentences.append([context, split, question,answer])    # Test의 경우 context를 포함해야해서 앞에 context 포함

            else:
                #cleared_sentences.append([sentence,question])
                cleared_sentences.append([context,sentence, question,answer]) # Test의 경우 context를 포함해야해서 앞에 context 포함


df = pd.DataFrame(cleared_sentences)
#name = '/data/workspace/psh/data/Preprocess/'+str(sentence_max_length)+'_Test_data.csv'
name = '/data/workspace/psh/data/Preprocess/'+str(sentence_max_length)+'/'+str(sentence_max_length)+'_Test_data.csv'
df.to_csv(name,index=False)
'''
######################################################## Preprocess ##############################################################################


######################################################## Fine-tuning ##############################################################################

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
if Pooler == True:
    kobert_model = BertModel.from_pretrained('skt/kobert-base-v1', add_pooling_layer=True, output_hidden_states=False).to(device)
elif Pooler == False:
    kobert_model = BertModel.from_pretrained('skt/kobert-base-v1', add_pooling_layer=False, output_hidden_states=False).to(device)

train_data_name = '/data/workspace/psh/data/Preprocess/'+str(sentence_max_length)+'/'+str(sentence_max_length)+'_Train_data.csv'
training_data, validation_data = data_process(train_data_name,'train')

# Convert sentences to BERT embeddings with max pooling
def get_kobert_embedding(sentences,tag):
    if tag == 'C': #Context 내 문장 집합인 경우
        sentence_embeddings = torch.zeros(1,sentence_max_length,768)
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(device)
            outputs = kobert_model(**inputs)
            sentence_embeddings += outputs.last_hidden_state.detach().cpu()
        mean_embedding = sentence_embeddings/len(sentences)

        return mean_embedding.squeeze()
    elif tag == 'Q':
        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(device)
        outputs = kobert_model(**inputs)

        return outputs.last_hidden_state.squeeze().detach().cpu()

# Prepare training pairs and labels
#train_pairs = [(get_kobert_embedding(pair[0],'C'), get_kobert_embedding(pair[1],'Q')) for pair in tqdm(training_data)]
#torch.save(train_pairs,'/data/workspace/psh/Embedding/Train/'+str(sentence_max_length)+'_Train_Embedding_saved.pt')
train_embedd_name = '/data/workspace/psh/Embedding/Train/'+str(sentence_max_length)+'_Train_Embedding_saved.pt'
train_pairs = torch.load(train_embedd_name)
train_pairs = train_pairs[:train_sample]
train_dataset = TensorDataset(torch.stack([pair[0] for pair in train_pairs]), torch.stack([pair[1] for pair in train_pairs]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#val_pairs = [(get_kobert_embedding(pair[0],'C'), get_kobert_embedding(pair[1],'Q')) for pair in tqdm(validation_data)]
#torch.save(val_pairs,'/data/workspace/psh/Embedding/Train/'+str(sentence_max_length)+'_Valid_Embedding_saved.pt')
valid_embedd_name = '/data/workspace/psh/Embedding/Train/'+str(sentence_max_length)+'_Valid_Embedding_saved.pt'
val_pairs = torch.load(valid_embedd_name)
val_pairs = val_pairs[:valid_sample]
val_dataset = TensorDataset(torch.stack([pair[0] for pair in val_pairs]), torch.stack([pair[1] for pair in val_pairs]))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print("Train, Validation embedding vector loaded")

Siamese_model = SiameseNetwork(Pooler).to(device)
optimizer = torch.optim.AdamW(Siamese_model.parameters(), lr=1e-5)
early_stopping = EarlyStopping(patience=5, verbose=True)
cosine_similarity = nn.CosineSimilarity()
num_epochs = 50

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
            context_out, question_out = Siamese_model(x,y)
            #loss = torch.mean(torch.cdist(context_out,question_out,p=2))
            loss = torch.mean(1-cosine_similarity(context_out, question_out))
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

                #loss = torch.mean(torch.cdist(context_out,question_out,p=2))
                loss = torch.mean(1-cosine_similarity(context_out, question_out))
                val_total_loss += loss.item()

                vepoch.set_postfix(loss=loss.item())

    train_average_loss = train_total_loss / len(train_loader)
    val_average_loss = val_total_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train average Loss: {train_average_loss:.8f}, Validation average Loss: {val_average_loss:.8f}")
    early_stopping(val_average_loss, Siamese_model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# if Pooler == True:
#     name = '/data/workspace/psh/model/Pooler/' + str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese_Final_EUC.pt'
# elif Pooler==False:
#     name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese_Final_EUC.pt'
# torch.save(Siamese_model,name)

######################################################## Fine-tuning ##############################################################################


########################################################### Test ##################################################################################

if Pooler == True:
    model_name = '/data/workspace/psh/model/Pooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'
elif Pooler == False:
    model_name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'

#model_name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_best.pt'
KoBERT_Retriver = torch.load(model_name)
kobert_model = KoBERT_Retriver.kobert  # 재학습된 kobert 모델 사용
kobert_model = kobert_model.to(device)


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

test_data_name = '/data/workspace/psh/data/Preprocess/' + str(sentence_max_length) + '/' + str(sentence_max_length) + '_Test_data.csv'
test_context_sentence, test_question = data_process(test_data_name, 'test')     # test_context_sentence = [context, sentence list, context_index]     test_question = [question, question_index, answer]
test_context_sentence,test_question = test_context_sentence[:100], test_question[:100]

Same_context = [[] for i in range(len(test_context_sentence))]
for context,_,id in test_context_sentence:
    Same_context[id-1].append(id)
    for compare_context,_,compare_id in test_context_sentence:
        if context == compare_context and compare_id != id:
            Same_context[id-1].append(compare_id)

def get_kobert_embedding(sentences, tag):
    if tag == 'C':  # Context 내 문장 집합인 경우
        sentence_embeddings = torch.zeros(1, sentence_max_length, 768)
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(
                device)
            outputs = kobert_model(**inputs)
            if Pooler == True:
                sentence_embeddings += outputs.pooler_output.detach().cpu()
            elif Pooler == False:
                sentence_embeddings += outputs.last_hidden_state.detach().cpu()
        mean_embedding = sentence_embeddings / len(sentences)

        return mean_embedding
    elif tag == 'Q':
        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', max_length=sentence_max_length).to(
            device)
        outputs = kobert_model(**inputs)
        if Pooler == True:
            return outputs.pooler_output.detach().cpu()
        elif Pooler == False:
            return outputs.last_hidden_state.detach().cpu()


with torch.no_grad():
    test_context_embeddings = [[context, get_kobert_embedding(sentence, 'C'), num] for context, sentence, num in tqdm(test_context_sentence)]  # [context, context_embedding_vector, context_num]
    test_question_embeddings = [[question, get_kobert_embedding(question, 'Q'), num, answer] for question, num, answer in tqdm(test_question)]  # [Sentence_embedding_vector, question_num]

    Score = 0
    Correct = [0, 0, 0,0,0]

    for question, question_embedding, question_num,answer in test_question_embeddings:
        cos_sim = []
        for context, context_embedding, context_num in test_context_embeddings:
            similarity = torch.mean(cosine_similarity(context_embedding, question_embedding))
            cos_sim.append([similarity.item(), context, context_num])
        cos_sim.sort(reverse=True)
        print('\n')
        print('=' * 150)
        print('\n[ Q.', question_num, 'Question:', question, ']')
        print("Same context with :", Same_context[question_num-1])
        print('\nCandidate 1:', cos_sim[0][1][:50], '....\nCosine_similarity:', cos_sim[0][0], '\tContext_num:',
              cos_sim[0][2])
        print('\nCandidate 2:', cos_sim[1][1][:50], '...\nCosine_similarity:', cos_sim[1][0], '\tContext_num:',
              cos_sim[1][2])
        print('\nCandidate 3:', cos_sim[2][1][:50], '...\nCosine_similarity:', cos_sim[2][0], '\tContext_num:',
              cos_sim[2][2])
        print('\nCandidate 4:', cos_sim[3][1][:50], '...\nCosine_similarity:', cos_sim[3][0], '\tContext_num:',
              cos_sim[3][2])
        print('\nCandidate 5:', cos_sim[4][1][:50], '...\nCosine_similarity:', cos_sim[4][0], '\tContext_num:',
              cos_sim[4][2])
        print('\n')

        context_nums = [cos_sim[0][2], cos_sim[1][2], cos_sim[2][2], cos_sim[3][2], cos_sim[4][2]]
        context = [cos_sim[0][1], cos_sim[1][1], cos_sim[2][1], cos_sim[3][1], cos_sim[4][1]]

        question_num = question_num
        for i,context_num in enumerate(context_nums):
            Same = Same_context[question_num-1]
            if context_num == question_num:
                if i == 0:
                    Correct[0] += 1
                    print("Answer in Candidate",i+1)
                    break
                elif i == 1:
                    Correct[1] += 1
                    print("Answer in Candidate", i + 1)
                    break
                elif i ==2:
                    Correct[2] += 1
                    print("Answer in Candidate", i + 1)
                    break
                elif i == 3:
                    Correct[3] += 1
                    print("Answer in Candidate", i + 1)
                    break
                elif i == 4:
                    Correct[4] += 1
                    print("Answer in Candidate", i + 1)
                    break
            elif context_num != question_num:
                if context_num in Same:
                    if i == 0:
                        Correct[0] += 1
                        print("Answer in Candidate", i + 1)
                        break
                    elif i == 1:
                        Correct[1] += 1
                        print("Answer in Candidate", i + 1)
                        break
                    elif i == 2:
                        Correct[2] += 1
                        print("Answer in Candidate", i + 1)
                        break
                    elif i == 3:
                        Correct[3] += 1
                        print("Answer in Candidate", i + 1)
                        break
                    elif i == 4:
                        Correct[4] += 1
                        print("Answer in Candidate", i + 1)
                        break

        #Reader(context, question, answer)
        print('=' * 150)

    print("Top-3 Score : ",Correct[0]*1+Correct[1]*2/3+Correct[2]*1/3, "Correct :",Correct[:3], 'Total :', sum(Correct[:3]))
    print("Top-5 Score : ", Correct[0] * 1 + Correct[1] * 4 / 5 + Correct[2] * 3 / 5 + Correct[3] * 2/5 + Correct[4]*1/5, "Correct :", Correct,
          'Total :', sum(Correct))

########################################################### Test ##################################################################################

