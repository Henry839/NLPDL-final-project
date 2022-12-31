# this file is for loading dataset
# available dataset : "acl" : aclarc , "sci" : scierc
from datasets import Dataset,DatasetDict,load_dataset,concatenate_datasets,ClassLabel,Value
import os
import json 
def get_single_dataset(dataset_name):
    if "acl_test" == dataset_name:
        dataset = load_dataset("json",data_files = {"train":"aclarc/train.jsonl","validation" : "aclarc/dev.jsonl","test":"aclarc/test.jsonl"})
        dataset = dataset.map(lambda i : {"text" : i["text"],"label" : i["label"]},remove_columns = ["metadata"])
    elif "sci_test" == dataset_name:
        dataset = load_dataset("json",data_files = {"train":"scierc/train.jsonl","validation" : "scierc/dev.jsonl","test":"scierc/test.jsonl"})
        dataset = dataset.map(lambda i : {"text" : i["text"],"label" : i["label"]},remove_columns = ["metadata"])
    elif "citation_intent" == dataset_name:
        dataset = load_dataset("json",data_files = {"train":"data/citation_intent/train.jsonl","validation" : "data/citation_intent/dev.jsonl","test":"data/citation_intent/test.jsonl"})
        dataset = dataset.map(lambda i : {"text" : i["text"],"label" : i["label"]},remove_columns = ["metadata"])
    elif "sciie" == dataset_name:
        dataset = load_dataset("json",data_files = {"train":"data/sciie/train.jsonl","validation" : "data/sciie/dev.jsonl","test":"data/sciie/test.jsonl"})
        dataset = dataset.map(lambda i : {"text" : i["text"],"label" : i["label"]},remove_columns = ["metadata"])
    elif "ai_corpus" == dataset_name:
        dataset = load_dataset("text",data_files = {"train" : "data/ai_corpus.txt"})
        dataset = dataset['train']
        dataset = dataset.train_test_split( 0.1)
        validation_set = dataset['test'].train_test_split(0.1)
        dataset = DatasetDict({"train":dataset['train'],'validation':validation_set['test'],'test':validation_set['train']})


    
    elif "computer science" in dataset_name:
        i = int(dataset_name[-1])
        if "11" in dataset_name:
            i = 11
        if "12" in dataset_name:
            i = 12
        if "13" in dataset_name:
            i = 13
        if "14" in dataset_name:
            i = 14
        if "15" in dataset_name:
            i = 15
        #print(i)
        dataset = load_dataset("json",data_files = {'train':f"data/metadata/cs/metadata_{i}.json"})['train']
        dataset = dataset.remove_columns(['paper_id', 'title', 'authors', 'year', 'arxiv_id', 'acl_id', 'pmc_id', 'pubmed_id', 'doi', 'venue', 'journal', 'has_pdf_body_text', 'mag_id', 'mag_field_of_study', 'outbound_citations', 'inbound_citations', 'has_outbound_citations', 'has_inbound_citations', 'has_pdf_parse', 'has_pdf_parsed_abstract', 'has_pdf_parsed_body_text', 'has_pdf_parsed_bib_entries', 'has_pdf_parsed_ref_entries', 's2_url'])
        dataset = dataset.rename_column('abstract','text')
        dataset = dataset.train_test_split(0.1)
        print(i)
        
    else :
        print("ERROR : No dataset prepared for this dataset name!")

    if "fs" in dataset_name:
        # print("few shot datset prepared:",dataset_name)
        dataset["train"] = dataset["train"].filter(lambda example , idx : idx < 32,with_indices = True)
    return dataset

def get_dataset(dataset_name):
    dataset = None
    flag = 1
    if type(dataset_name)== type([1]):
        train_list_dataset = []
        test_list_dataset = []
        #validation_list_dataset = []
        label_set = []
        for i in dataset_name:
            a = get_single_dataset(i)
            train_list_dataset.append( a["train"])
            #validation_list_dataset.append(get_single_dataset(i)['validation'])
            test_list_dataset.append( a["test"])
            if "label" not in train_list_dataset[-1][0].keys():
                flag = 0
            if flag:
                label_set = label_set + list(set(test_list_dataset[-1]['label']))

        dataset = DatasetDict({"train":concatenate_datasets(train_list_dataset),"test":concatenate_datasets(test_list_dataset)})
    else:
        dataset = get_single_dataset(dataset_name)
        label_set = set(dataset["test"]["label"])
    new_features = dataset["test"].features.copy()
    if flag:
        new_features['label'] = ClassLabel(num_classes = len(label_set),names = list(label_set))
    # print(new_features)
    dataset = dataset.cast(new_features)
    dataset = dataset.filter(lambda x : x['text'] != None)

    return dataset
