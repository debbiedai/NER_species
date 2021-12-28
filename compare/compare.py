import os
import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
import inflect
from numpy.core.numeric import allclose
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold
import csv

nltk.download('punkt')
def xml2plaintext(xml_file):
    with open(xml_file, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, 'xml')
    passages = bs_data.find_all('passage')
    offset = 0
    all_text = ""
    all_annotations = []
    for passage in passages:
        text = passage.find('text').text
        all_text += text
        anno = passage.find_all('annotation')
        offset += int(passage.find('offset').text)
        for an in anno:
            an_type = an.find('infon', {'key':'type'})
            if an_type.text == 'Species':
                location = an.find('location')
                o = int(location['offset'])
                l = int(location['length'])
                t = an.find('text').text
                for word in t.split('/'):
                    all_annotations.append((o, l, word, 'Species'))
        all_text += " "
    return all_text, all_annotations
    
def ch2token(all_token, ch_annotations):
    print(len(all_token))
    inflect_eg = inflect.engine()
    index = 0
    token_index = []

    while len(ch_annotations):
        annotation = ch_annotations.pop(0)
        _, _, t, _ = annotation
        anno_token = word_tokenize(t)
        size = len(anno_token)
        while True:
            isLabel = True
            if index>len(all_token)-1:
                break
            for i in range(size):
                artical_t = inflect_eg.singular_noun(all_token[index + i])                
                if not artical_t:
                    artical_t = all_token[index + i]
                
                anno_t = inflect_eg.singular_noun(anno_token[i])
                if not anno_t:
                    anno_t = anno_token[i]
                if anno_t not in artical_t:
                    isLabel = False
                    break
                
            if isLabel:
                token_index.append((index, size, " ".join(anno_token)))
                if '-' in artical_t:
                    if '-' in anno_t:
                        artical_t = artical_t.replace(anno_t, '')
                    else:
                        artical_t = artical_t.replace(anno_t+'-', '')
                else:
                    artical_t = artical_t.replace(anno_t, '')
                if not artical_t:
                    index += size
                break            
            index += 1    

    return token_index


def remove_space(file, save_name, save_path):
    count = 0
    with open(str(file), 'r') as f:
        text = f.readlines()
    remove_space = []
    for i in text:
        x = i.split('\n')[0]
        if len(x) != 0:
            remove_space.append(x)
    save_data = {'Biobert':remove_space}
    ds = pd.DataFrame(save_data, columns=['Biobert'])
    ds.to_csv(os.path.join(save_path, str(save_name) + '_biobert.tsv'), index=False, header=['Biobert'])

def preprocess(fold_csv_path, save_path):
    for j in range(10):
        doc_list, text_list, tag_list, sent_list = [], [], [], []
        doc_num = 1
        sent_num = 1
        path = fold_csv_path + '/fold_name_' + str(j) + '.csv'
        fold_data = pd.read_csv(path)
        for file in fold_data['file_name']:
            print(file)
            text, annotations = xml2plaintext(file)
            text = text.replace('/', ' / ')

            token_text = word_tokenize(text)
            token_index = ch2token(token_text, annotations[:])

            #####
            for token in token_text:
                text_list.append(token)
            real_id = []
            inside_id = []
            for ans in token_index:
                offset, length, label = ans
                real_id.append(offset)
                while length != 1:
                    offset += 1
                    real_id.append(offset)
                    inside_id.append(offset)
                    length -= 1
            for i in range(len(token_text)):
                doc_list.append(file)
                if i in real_id and i not in inside_id:
                    tag_list.append('B')
                elif i in inside_id:
                    tag_list.append('I')
                else:
                    tag_list.append('O')
            doc_num += 1
            ##### from .xml to txt          

            # if not os.path.isdir('./plaintext_species'):
            #     os.mkdir('./plaintext_species')
            # with open('./plaintext_species/' + file.replace('.xml', '.txt'), 'w') as f:
            #     f.write(text)
            #     f.close()
            # with open('./plaintext_species/' + file.replace('.xml', '_chr_annotations.txt'), 'w') as f:
            #     for anno in annotations:
            #         offset, length, label, _ = anno
            #         f.write(str(offset) + ' ' + str(length) + ' ' + str(label) + '\n')
            #     f.close()

            # with open('./plaintext_species/' + file.replace('.xml', '_token_annotations.txt'), 'w') as f:
            #     for ans in token_index:
            #         offset, length, label = ans
            #         f.write(str(offset) + ' ' + str(length) + ' ' + str(label) + '\n')
            #     f.close()

        ### sent_id
        for i in range(len(text_list)-1):
            sent_list.append(sent_num)
            if text_list[i] == '.' and text_list[i+1][0] == text_list[i+1][0].upper():
                sent_num += 1
        sent_list.append(sent_num)
        ###
        word_list = np.array(text_list)
        tags_list = np.array(tag_list)
        name = ['Doc_ID', 'Sent_ID', 'Word', 'tag']
        f = np.stack((doc_list, sent_list, word_list, tags_list), 1)
        log = pd.DataFrame(data = f)
        log.to_csv(save_path + '/fold_' + str(j) + '_SR4GN'+ '.tsv', header=name, index=False)


def compare(fold_name):
    # fold_name = 'fold_0'
    df_b = pd.read_csv('./biobert/'+ str(fold_name) + '_biobert.tsv')
    df_s = pd.read_csv('./SR4GN/'+ str(fold_name) + '_SR4GN.tsv')
    sr4gn_tag, biobert_tag, compare_ = [], [], []

    sr4gn = df_s['tag']
    for j in sr4gn:
        sr4gn_tag.append(j)
    biobert = df_b['Biobert']
    for i in biobert:
        biobert_tag.append(i[-1])
    # print(len(sr4gn_tag))
    # print(len(biobert_tag))
    for s,b in zip(sr4gn_tag, biobert_tag):
        if s == b:
            compare_.append(0)
        else:
            compare_.append(1)
    save_data = {'Doc_ID':df_s['Doc_ID'], 'Sent_ID':df_s['Sent_ID'], 'Word':df_s['Word'], 'SR4GN':sr4gn_tag, 'Biobert':biobert_tag,'compare':compare_}
    ds = pd.DataFrame(save_data, columns=['Doc_ID', 'Sent_ID', 'Word','SR4GN', 'Biobert', 'compare'])
    ds.to_csv(fold_name + '_compare.tsv', index=False, sep='\t')



if __name__ == '__main__':
    # After downloading the test_predictions.txt file from Colab, the spece between each sentence 
    # should be removed that might easier to comapre performance of Biobert and SR4GN
    remove_space('test_predictions.txt', 'fold_0', './biobert')
    # Read the output xml file from SR4GN and fold_name.tsv, token each abstract and save to fold_[0-9].tsv
    preprocess('./split_fold', './NER_species')
    # compare SR4GN & Biobert
    compare('fold_0')
    
