from bs4 import BeautifulSoup
import os
from nltk.tokenize import word_tokenize
import inflect
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import csv
from transformers import AutoTokenizer
import shutil

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

def preprocess_same_len(max_len, model_name_or_path, read_file, write_file):
    save_data = open(write_file, 'w')
    subword_len_counter = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    max_len -= tokenizer.num_special_tokens_to_add()
    
    with open(read_file, encoding="utf-8") as f_p:
        for line in f_p:
            line = line.rstrip()
            # if space, reset subword_len_counter

            if not line or '.xml' in line:
                save_data.write(line + '\n')
                subword_len_counter = 0
                continue

            token = line.split()[0]
            print(tokenizer.tokenize(token)[:])
            current_subwords_len = len(tokenizer.tokenize(token))


            if current_subwords_len == 0:
                continue

            if (subword_len_counter + current_subwords_len) > max_len:
                print("")
                save_data.write('\n')
                save_data.write(line + '\n')
                subword_len_counter = current_subwords_len
                continue

            subword_len_counter += current_subwords_len
            save_data.write(line + '\n')

def creat_LOOCV(data_path, save_tsv_dir, save_txt_dir, txt_samelen_dir):
    all_files = os.listdir(data_path)    
    for file in all_files:
        if file.endswith('.xml'):
            doc_list, text_list, tag_list, sent_list = [], [], [], []
            doc_num = 1
            sent_num = 1
            print(file)
            text, annotations = xml2plaintext(os.path.join(data_path, file))
            text = text.replace('/', ' / ')
            token_text = word_tokenize(text)    
            token_index = ch2token(token_text, annotations[:])

    #         #####
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
            print('real_id:', real_id)
            print('inside_id:', inside_id)
            for i in range(len(token_text)):
                doc_list.append(file)
                if i in real_id and i not in inside_id:
                    tag_list.append('B')
                elif i in inside_id:
                    tag_list.append('I')
                else:
                    tag_list.append('O')
            print('doc_num:', doc_num)
            doc_num += 1

    #         ### sent_id
            for i in range(len(text_list)-1):
                sent_list.append(sent_num)
                if text_list[i] == '.' and text_list[i+1][0] == text_list[i+1][0].upper():
                    sent_num += 1
            sent_list.append(sent_num)
    #         ###
            word_list = np.array(text_list)
            tags_list = np.array(tag_list)
            name = ['Doc_ID', 'Sent_ID', 'Word', 'tag']
            f = np.stack((doc_list, sent_list, word_list, tags_list), 1)
            log = pd.DataFrame(data = f)
            log.to_csv(save_tsv_dir + str(file.split('.')[0]) + '.tsv', header=False, sep='\t', encoding='utf-8')
    #         ###
            with open(save_txt_dir + str(file.split('.')[0]) + ".txt", "w") as f:
                tmpSentID = 1
                for (s, w, t) in zip(sent_list, word_list, tags_list):
                    if s != tmpSentID:
                        f.writelines('\n')
                        tmpSentID += 1
                    f.write(str(w) + ' ' + str(t) + '\n')
                f.close()
    
    #### same_length
    files = os.listdir(save_txt_dir)
    for file in files:
        if file.endswith('.txt'):
            path = os.path.join(save_txt_dir, file)
            preprocess_same_len(50, 'dmis-lab/biobert-base-cased-v1.1', path, txt_samelen_dir + file)
    
    ###
    all_txt = os.listdir(txt_samelen_dir)

    for i in range(len(all_txt)):
        print(i)
        test_data = all_txt[i]
        train_datas = all_txt[:i]+all_txt[i+1:]


        save_path_dir = os.path.join('./LOOCV_dataset', str(i))
        if not os.path.isdir(save_path_dir):
            os.mkdir(save_path_dir)

        # put labels.txt on the same dir, it will copy labels.txt
        label_data = './labels.txt'
        test_data = os.path.join(txt_samelen_dir,str(test_data))
        devel = os.path.join(save_path_dir, "devel.txt")
        test = os.path.join(save_path_dir, "test.txt")
        label = os.path.join(save_path_dir, "labels.txt")
        shutil.copyfile(test_data, devel)
        shutil.copyfile(test_data, test)
        shutil.copyfile(label_data, label)

        f = open(os.path.join(save_path_dir, "train_dev.txt"), 'w')
        for train_data in train_datas:
            dataset = txt_samelen_dir + str(train_data)
            with open(dataset, "rt") as f_p:
                for line in f_p:
                    line = line.rstrip()
                    f.write(line + '\n')
            f.write('\n')


if __name__ == '__main__':
    creat_LOOCV('./dataset/', './dataset/tsv/', './dataset/txt/', './dataset/txt_same_len/')