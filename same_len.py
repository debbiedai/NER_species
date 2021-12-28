import csv
from transformers import AutoTokenizer
import os
import shutil


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

def creat_train_data(train_fold_list, val_fold, test_fold, data_path, save_path):
    save_path_dir = os.path.join(save_path, "v"+str(val_fold)+"_t"+str(test_fold))
    if not os.path.exists(save_path_dir):
        os.mkdir(save_path_dir)
    val_data = os.path.join(data_path, "fold_"+str(val_fold)+".txt")
    test_data = os.path.join(data_path, "fold_"+str(test_fold)+".txt")
    devel = os.path.join(save_path_dir, "devel.txt")
    test = os.path.join(save_path_dir, "test.txt")
    shutil.copyfile(val_data, devel)
    shutil.copyfile(test_data, test)


    f = open(os.path.join(save_path_dir, "train_dev.txt"), 'w')
    for i in train_fold_list:
        dataset = data_path + "/fold_" + str(i) + ".txt"
        with open(dataset, "rt") as f_p:
            for line in f_p:
                line = line.rstrip()
                f.write(line + '\n')
        f.write('\n')

if __name__ == '__main__':
    # both of no name file and name file need to be processed to same length, so you need to change path directory when run the name file
    # the path of 10 folds no_name/name file, def [preprocess_same_len] truncate each setence to same length
    files = os.listdir('./preprocessed/name')
    for file in files:
        if file.endswith('.txt'):
            path = os.path.join('./preprocessed/name', file)
            preprocess_same_len(50, 'dmis-lab/biobert-base-cased-v1.1', path, './preprocessed/name/same_len/' + file)


    # only no_name file need to run the code below for creating train_dev.txt, devel.txt, test.txt 
    val_list = [0,1,2,3,4,5,6,7,8,9]
    test_list = [1,2,3,4,5,6,7,8,9,0]
    for i in range(len(test_list)):
        val_num = val_list[i]
        test_num = test_list[i]
        train_num = [i for i in range(10)]
        train_num.remove(val_num)
        train_num.remove(test_num)
        print('train_num', train_num)
        creat_train_data(train_num, val_num, test_num, './preprocessed/no_name/same_len/', './preprocessed/training_data/')