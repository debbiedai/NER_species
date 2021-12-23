from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import inflect
from numpy.core.numeric import allclose
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv


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
            if index > len(all_token)-1:
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




def split_fold(all_file_list, n_fold, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)    
    all_data = []
    for file in all_file_list:
        if file.endswith('.xml'):
            all_data.append(file)
    kf = KFold(n_splits=n_fold, shuffle=False)
    train_ids, test_ids = [], []
    for train_id, test_id in kf.split(all_data):
        train_ids.append(train_id)
        test_ids.append(test_id)

    all_data = np.array(all_data)
    for i in range(n_fold):
        fold_data = all_data[test_ids[i]]        
        fold_data = fold_data.reshape((-1,1))
        df = pd.DataFrame(fold_data)
        df.to_csv(save_path + '/fold_name_' + str(i) + '.csv', header=['file_name'])

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
            if not os.path.isdir('./plaintext_species'):
                os.mkdir('./plaintext_species')
            with open('./plaintext_species/' + file.replace('.xml', '.txt'), 'w') as f:
                f.write(text)
                f.close()
            with open('./plaintext_species/' + file.replace('.xml', '_chr_annotations.txt'), 'w') as f:
                for anno in annotations:
                    offset, length, label, _ = anno
                    f.write(str(offset) + ' ' + str(length) + ' ' + str(label) + '\n')
                f.close()

            with open('./plaintext_species/' + file.replace('.xml', '_token_annotations.txt'), 'w') as f:
                for ans in token_index:
                    offset, length, label = ans
                    f.write(str(offset) + ' ' + str(length) + ' ' + str(label) + '\n')
                f.close()

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
        log.to_csv(save_path + '/fold_' + str(j) + '.tsv', header=False)


def tsv_to_txt(tsv_path, save_path, fold_num, add_txt_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(fold_num):
        # Open tsv and txt files(open txt file in write mode)
        tsv_file = open(tsv_path + "/fold_" + str(i) + ".tsv")
        txt_file = open(save_path + "/fold_" + str(i) + ".txt", "w")

        # Read tsv file
        read_tsv = csv.reader(tsv_file, delimiter="\t")

        # write data in txt file line by line
        tmp = 1
        tmp_name = 'whatever'
        for row in read_tsv:
            joined_string = "\t".join(row)
            txt_name = joined_string.split(',')[1]
            
            tmp_sent  = int(joined_string.split(',')[2])
            x = joined_string.split(',')
            if x[-2] == '"':
                word = ","
            else:
                word = x[-2]

            tag = x[-1]
            joined_string = word+' '+tag
            # split sentence (add space between different sentence)
            if tmp != tmp_sent:
                txt_file.writelines('\n')
                tmp += 1
            # if True, add txt_name 
            if add_txt_name:
                if txt_name != tmp_name:
                    txt_file.writelines(txt_name + '\n')
                    tmp_name = txt_name
            
            txt_file.writelines(joined_string+'\n')

        txt_file.close()

if __name__ == '__main__':
    files = os.listdir('../')
    # split all .xml to 10 folds, def [split_fold] would generate fold_name_[0-9].csv which store .xml name 
    split_fold(files, 10, './NER_species/split_fold')
    # token each abstract and save to fold_[0-9].csv
    preprocess('./NER_species/split_fold', './NER_species')
    # convert tsv to txt
    tsv_to_txt('./NER_species', './NER_species/preprocessed/name', 10, True)
    tsv_to_txt('./NER_species', './NER_species/preprocessed/no_name', 10, False)




