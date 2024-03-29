import pandas
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np
from pandas import DataFrame

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
model = SentenceTransformer('sentence-t5-xxl', cache_folder='/')

df_train = pd.read_csv('./data/consolidated_xbrl_train.csv')
df_test = pd.read_csv('./data/consolidated_xbrl_test.csv')

df_merged = pd.concat([df_train, df_test])

all_unique_tag_docs = list(set(df_merged['Tag_Doc'].values.tolist()))

all_unique_tag_docs.append('others')

tag_doc_tag_words_mapping = {}

for i, row in df_merged.iterrows():
    if row['Tag_Doc'].lower() not in tag_doc_tag_words_mapping:
        tag_doc_tag_words_mapping[row['Tag_Doc'].lower()] = row.GT_Tag_Words

tag_doc_tag_words_mapping['others'] = 'others'

f1 = open('./lora_prediction.txt', 'r')

count = 0
count_correct = 0

all_tag_embedding = model.encode(all_unique_tag_docs)
true_list = []
pred_list = []
sentences = list(df_test['Sentence'])
gt_tag_words = list(df_test['Tag_Doc'])
numerals = list(df_test['Numeral'])

row_index = 0
d = {}
for line in f1:
    true_tag = line.strip().split('Pred:')[0].split('True:')[1]
    pred_tag = line.strip().split('Pred:')[1]

    pred_tag_embeddings = model.encode([pred_tag])

    cosine_similarities_pred_all = util.dot_score(pred_tag_embeddings, all_tag_embedding)

    values, indices = torch.topk(cosine_similarities_pred_all, 1)

    top_3_tags_list = []
    for index in indices.tolist()[0]:
        top_3_tags_list.append(all_unique_tag_docs[index])

    true_tag = true_tag.strip()

    if true_tag in top_3_tags_list:
        count_correct = count_correct + 1

        flag = True
    else:

        flag = False

    true_list.append(tag_doc_tag_words_mapping[true_tag.lower()])
    pred_list.append(tag_doc_tag_words_mapping[top_3_tags_list[0].strip().lower()])

    sentence = sentences[count]
    gt_tag = gt_tag_words[count]
    numeral = numerals[count]
    peft_pred_tag = pred_tag
    final_pred_tag = top_3_tags_list[0].strip().lower()

    final_tag_words = tag_doc_tag_words_mapping[top_3_tags_list[0].strip().lower()]
    gt_tag_words_text = tag_doc_tag_words_mapping[true_tag.lower()]

    count = count + 1

print('Accuracy,', count_correct, count, count_correct / count)

print('Macro Performance', precision_recall_fscore_support(true_list, pred_list, average='macro'))
print('Micro Performance', precision_recall_fscore_support(true_list, pred_list, average='micro'))
