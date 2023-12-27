import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import json
import requests
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from transformers import BertModel
######################################################## Parameter ###############################################################################
sentence_max_length = 64
train_sample = 32
valid_sample = 32
batch_size = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
Pooler = False
######################################################## Parameter ###############################################################################

######################################################## Class & Function #########################################################################
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

def Reader(contexts, query,answer):

    messages = [{"role": "context", "content": context} for context in contexts]
    messages.append({"role": "user", "content": query})

    input_q = json.dumps({
        "model": "t5-small-fid", "mode": "",
        "messages": messages})

    response = requests.post(f'http://211.39.140.48:9090/predictions/temp', data=input_q)
    print("Response :",response.json()['choices'][0]['message']['content'])
    print("Answer :",answer)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self,Pooler).__init__()
        self.kobert = kobert_model
        self.cosine_similarity = F.cosine_similarity

    def forward(self, x1, x2):
        context_out = self.kobert(inputs_embeds=x1,token_type_ids=None).last_hidden_state.view(-1,sentence_max_length*768)
        question_out = self.kobert(inputs_embeds=x2,token_type_ids=None).last_hidden_state.view(-1,sentence_max_length*768)
        #similarity = self.cosine_similarity(context_out, question_out, dim=1)

        return context_out, question_out


######################################################## Class & Function #########################################################################

########################################################### Test ##################################################################################

if Pooler == True:
    model_name = '/data/workspace/psh/model/Pooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'
elif Pooler == False:
    model_name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_KoBERT_Siamese.pt'

#model_name = '/data/workspace/psh/model/NoPooler/'+str(sentence_max_length) + '_'+str(train_sample)+'_best.pt'
KoBERT_Retriver = torch.load(model_name)
kobert_model = KoBERT_Retriver.kobert  # 재학습된 kobert 모델 사용
kobert_model = kobert_model.to(device)

#kobert_model = BertModel.from_pretrained('skt/kobert-base-v1', add_pooling_layer=False, output_hidden_states=False).to(device)

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

#criterion = torch.nn.MSELoss()
with torch.no_grad():
    test_context_embeddings = [[context, get_kobert_embedding(sentence, 'C'), num] for context, sentence, num in tqdm(test_context_sentence)]  # [context, context_embedding_vector, context_num]
    #torch.save(test_context_embeddings,'/data/workspace/psh/Embedding/Pooler/' + str(sentence_max_length) + '_Test_context_Embedding_saved.pt')
    # if Pooler == True:
    #     test_context_name = '/data/workspace/psh/Embedding/Pooler/' + str(sentence_max_length) + '_Test_context_Embedding_saved.pt'
    # elif Pooler == False:
    #     test_context_name = '/data/workspace/psh/Embedding/NoPooler/' + str(sentence_max_length) + '_Test_context_Embedding_saved.pt'
    # test_context_embeddings = torch.load(test_context_name)
    #test_context_embeddings = test_context_embeddings[:100]
    test_question_embeddings = [[question, get_kobert_embedding(question, 'Q'), num, answer] for question, num, answer in tqdm(test_question)]  # [Sentence_embedding_vector, question_num]
    # #torch.save(test_question_embeddings,'/data/workspace/psh/Embedding/Pooler/' + str(sentence_max_length) + '_Test_question_Embedding_saved.pt')
    # if Pooler == True:
    #     test_question_name = '/data/workspace/psh/Embedding/Pooler/' + str(sentence_max_length) + '_Test_question_Embedding_saved.pt'
    # elif Pooler == False:
    #     test_question_name = '/data/workspace/psh/Embedding/NoPooler/' + str(sentence_max_length) + '_Test_question_Embedding_saved.pt'
    # test_question_embeddings = torch.load(test_question_name)
    #test_question_embeddings = test_question_embeddings[:100]

    Score = 0
    Correct = [0, 0, 0,0,0]

    for question, question_embedding, question_num,answer in test_question_embeddings:
        cos_sim = []
        for context, context_embedding, context_num in test_context_embeddings:
            similarity = torch.mean(cosine_similarity(context_embedding, question_embedding))
            #similarity = torch.mean(torch.cdist(context_embedding,question_embedding,p=2))         # Eucledian
            #similarity = criterion(context_embedding,question_embedding)                           # MSELoss
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

