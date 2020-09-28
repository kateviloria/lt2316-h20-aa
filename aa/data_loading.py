
#basics
import random
import pandas as pd
import torch


class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)


    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        pass


    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU


    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass




#basics
import random
import pandas as pd
import torch

# own imports
import os
import xml.etree.ElementTree as ET
import re
from nltk.tokenize import WhitespaceTokenizer 
from nltk.tokenize import WordPunctTokenizer 
import string
from random import sample
import pandas as pd

# device = torch.device('cpu')

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):

    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    def get_subdir(self, path_to_folder):
        subfolders = [f.path for f in os.scandir(path_to_folder) if f.is_dir()]
        return subfolders

    def get_xmls(path):
        xml_list = []
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                xml_list.append(os.path.join(path, filename))
        return xml_list

    def process_text(str):
        # tokenizer !
        new_list = []
        word_list = WhitespaceTokenizer().tokenize(str)
        all_punct = list(string.punctuation)
        
        for every_word_idx in range(len(word_list)):
            word = word_list[every_word_idx]
            last_idx = len(word) -1
            start_char = word[0]
            end_char = word[last_idx]
            
            if start_char in all_punct:
                split_punct = WordPunctTokenizer().tokenize(word)
                add_split = [new_list.append(every_newtoken) for every_newtoken in split_punct]

            elif end_char in all_punct:
                if len(word) > 1:
                    before_last = last_idx -1
                    if word[before_last] in all_punct: # '):'
                        new_list.append(word)     
                    else: # word + punct
                        split_punct = WordPunctTokenizer().tokenize(word)
                        add_split = [new_list.append(every_newtoken) for every_newtoken in split_punct]
                
                else: # len == 1
                    new_list.append(word)        
                    
            else: # no punct
                new_list.append(word)
                
        return new_list

    def char_offset(original_string, tokens):
        char_list = []
        start = 0
        for every_token in tokens:
            # start parameter is starting where to look if word is in middle of sentence
            beg = original_string.find(every_token, start) 
            char_length = len(every_token)
            end = beg + char_length -1   
            chars = (beg, end)
            char_list.append(chars)
            start = end
            
        # for checking 
        word2char = zip(tokens, char_list)

        return char_list

    def open_xmls(file_list):
        all_vocab = []
        token_dict = {}
        counter = 0
        
        # list of lists, each list is a token with data for every column
        all_tokens = [['sentence_id', 'token_id', 'char_start_id', 'char_end_id', 'split']] 
        ner_data =[['sentence_id', 'ner_id', 'char_start_id', 'char_end_id']]
                    
        for every_file in file_list:
            
            split = every_file[1]
            file_path = every_file[0]
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # sentence
            root_tag = root[0].tag

            for every_sentence in root:

                sent_id = every_sentence.attrib['id']

                text = every_sentence.attrib['text']
                
                tokenized = process_text(text)

                # get char offset for each token
                char_list = char_offset(text, tokenized)

                assert len(tokenized) == len(char_list),"Token list and character list do not have the same amount of values."

                for every_word_index in range(len(tokenized)):
                    word = tokenized[every_word_index]
                    all_vocab.append(word)

                    if word not in token_dict.keys():
                        token_dict[word] = counter
                        token_id = token_dict[word]
                        counter += 1
                    else: 
                        token_id = token_dict[word]

                    char_start_id = char_list[every_word_index][0]
                    char_end_id = char_list[every_word_index][1]

                    row = [sent_id, token_id, char_start_id, char_end_id, split]
                    all_tokens.append(row)

                # create ner_data
                for every_item in every_sentence.findall('entity'):
                    entity_id = every_item.attrib['id']

                    entity_name = every_item.attrib['text']

                    entity_type = every_item.attrib['type']

                    entity_charoffset = every_item.attrib['charOffset']

                    # entity have two char starts and ends
                    if ';' in entity_charoffset:
                        split_semi = entity_charoffset.split(';')

                        first = split_semi[0].split('-')
                        first_start = first[0]
                        first_end = first[1]

                        second = split_semi[1].split('-')
                        second_start = second[0]
                        second_end = second[1]
                        
                        ner_data.append([sent_id, entity_id, first_start, first_end])
                        ner_data.append([sent_id, entity_id, second_start, second_end])

                    else: # only one char start and end
                        char_list = entity_charoffset.split('-')

                        start = char_list[0]
                        end = char_list[1]
                        
                        ner_data.append([sent_id, entity_id, start, end])

        # turn list of lists into dataframes
        data_df = pd.DataFrame(all_tokens[1:], columns=all_tokens[0])
        ner_df = pd.DataFrame(ner_data[1:], columns=ner_data[0])
        
        return data_df, ner_df

    def _parse_data(self,data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        all_test_files = []
        all_train_files = []
        
        # get subfolders' paths of DDICorpus
        DDI_sub = self.get_subdir(data_dir)
        DDI_testdir = DDI_sub[0]
        DDI_traindir = DDI_sub[1]
        
        # folder paths under test directory
        test_sub = get_subdir(DDI_testdir)

        test_NER = test_sub[0] 
        test_NER_sub = get_subdir(test_NER)
        test_NER_MedLine = test_NER_sub[0]
        test_NER_DrugBank = test_NER_sub[1]
        
        test_DDI = test_sub[1]
        test_DDI_sub = get_subdir(test_DDI)
        test_DDI_MedLine = test_DDI_sub[0]
        test_DDI_DrugBank = test_DDI_sub[1]
        
        # folder paths under train directory
        train_sub = get_subdir(DDI_traindir)
        train_DrugBank = train_sub[0]
        train_MedLine = train_sub[1]
        
        # test xmls
        test_NER_MedLine_files = get_xmls(test_NER_MedLine)
        test_NER_DrugBank_files = get_xmls(test_NER_DrugBank)
        test_DDI_MedLine_files = get_xmls(test_DDI_MedLine)
        test_DDI_DrugBank_files = get_xmls(test_DDI_DrugBank)
        
        test_files = test_NER_MedLine_files + test_NER_DrugBank_files + test_DDI_MedLine_files + test_DDI_DrugBank_files
        
        # taking half of test for validation set
        val_count = len(test_files)//2 # floor division
        val = sample(test_files, val_count)
        all_val_files = []
        
        for every_file in val:
            all_val_files.append((every_file, 'val'))
            test_files.remove(every_file)

        
        [all_test_files.append((every_file, 'test'))for every_file in test_files]

        # train xmls
        train_MedLine_files = get_xmls(train_MedLine)
        train_DrugBank_files = get_xmls(train_DrugBank)

        train_files = train_MedLine_files + train_DrugBank_files
        
        [all_train_files.append((every_file, 'train'))for every_file in train_files]
        
        # list of tuples that has filepath and tag whether it's test, train, or val
        # ie. (filepath, 'val')
        all_data = all_val_files + all_test_files + all_train_files
        

        make_dataframes = open_xmls(all_data)
        data_df = make_dataframes[0]
        ner_df = make_dataframes[1]
        
        return data_df, ner_df

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        pass

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



