# encoding = "utf-8"
import json
from collections import defaultdict
from tqdm import tqdm
import numpy
from nltk import word_tokenize
import matplotlib.pyplot as plt

from utils.load_dataset import AsciiDataset

def dict_scores(input_dict):

    average_score_dict = {}

    for key in input_dict:
        micro_score = numpy.mean(numpy.array(input_dict[key]))
        average_score_dict[key]=micro_score

    return average_score_dict


if __name__=="__main__":

    test_data = AsciiDataset(data_path="./test_set_all/test.jsonl")
    results_file = open("./evaluations/mm-by-text/VLLM/llava-v1.5-13b-hf-both.jsonl", "r")
    results = []
    for line in results_file:
        results.append(json.loads(line.strip()))


    scores_for_each_concept = defaultdict(list)
    scores_for_each_lengthcounts = defaultdict(list)
    scores_for_each_charcounts = defaultdict(list)

    category2to1 = {}
    category3to2 = {}

    answer_rate = []
    for data, result in tqdm(zip(test_data, results)):
        score = 0
        # '''prob'''
        # if result["pred"]==result["label"]:
        #     score = 1

        letter = None
        cur_choices = eval(data["ori_choices"])
        try:
            pred_words = word_tokenize(result["pred"])

            for word in pred_words:
                if word in ["A", "B", "C", "D"]:
                    letter = word
                    break
            assert letter!=None
        except AssertionError:
            for choice_idx, choice in enumerate(cur_choices):
                if choice in result["pred"]:
                    letter = ["A", "B", "C", "D"][choice_idx]
                    break
        if letter == result["label"]:
            score = 1

        if letter!=None:
            answer_rate.append(1)
        else:
            answer_rate.append(0)

        if data["category-2"] not in category2to1:
            category2to1[data["category-2"]] = data["category-1"]
        else:
            assert category2to1[data["category-2"]] == data["category-1"]

        concept = data["category-2"]+"."+data["category-3"]
        if concept not in category3to2:
            category3to2[concept] = data["category-2"]
        else:
            assert category3to2[concept] == data["category-2"]

        scores_for_each_concept[data["category-2"]+"."+data["category-3"]].append(score)

        scores_for_each_lengthcounts[len(data["ascii_art"].split("\n"))].append(score)
        scores_for_each_charcounts[len(data["ascii_art"])].append(score)

    average_score_for_each_concept = dict_scores(scores_for_each_concept)


    all_micro_scores = []
    all_macro_scores = []

    category_2_micro_scores = {key:[] for key in category2to1.keys()}
    category_2_macro_scores = {key:[] for key in category2to1.keys()}

    category_1_micro_scores = {key:[] for key in list(set(category2to1.values()))}
    category_1_macro_scores = {key:[] for key in list(set(category2to1.values()))}

    for key in scores_for_each_concept:

        all_micro_scores += scores_for_each_concept[key]
        all_macro_scores.append(average_score_for_each_concept[key])

        category_2_micro_scores[category3to2[key]] += scores_for_each_concept[key]
        category_2_macro_scores[category3to2[key]].append(average_score_for_each_concept[key])

        category_1_micro_scores[category2to1[category3to2[key]]] += scores_for_each_concept[key]
        category_1_macro_scores[category2to1[category3to2[key]]].append(average_score_for_each_concept[key])
    
    print("answer ratio is ", sum(answer_rate)/len(answer_rate))
    print("micro average score is ", numpy.mean(numpy.array(all_micro_scores)))
    print("macro average score is ", numpy.mean(numpy.array(all_macro_scores)))

    # print("="*20)
    # for key in category_2_micro_scores:
    #     print(f"micro average score for {key} is ", numpy.mean(numpy.array(category_2_micro_scores[key])))
    #     print(f"macro average score for {key} is ", numpy.mean(numpy.array(category_2_macro_scores[key])))
    # print("="*20)
    # for key in category_1_macro_scores:
    #     print(f"micro average score for {key} is ", numpy.mean(numpy.array(category_1_micro_scores[key])))
    #     print(f"macro average score for {key} is ", numpy.mean(numpy.array(category_1_macro_scores[key])))


    # for key in sorted(list(category_2_micro_scores.keys())):
    #     print("key: {}, macro: {}".format(key, numpy.mean(category_2_macro_scores[key])))

    # print("======")
    # for key in sorted(list(category_1_micro_scores.keys())):
    #     print("key: {}, macro: {}".format(key, numpy.mean(category_1_macro_scores[key])))


    '''ascii art lines'''
    # print(sorted(list(scores_for_each_lengthcounts.keys())))
    
    # length_group = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
    
    # for key in scores_for_each_lengthcounts:
    #     if key<=5:
    #         length_group[0] += scores_for_each_lengthcounts[key]
    #     elif key>5 and key<=10:
    #         length_group[1] += scores_for_each_lengthcounts[key]
    #     elif key>10 and key<=15:
    #         length_group[2] += scores_for_each_lengthcounts[key]
    #     elif key>15 and key<=20:
    #         length_group[3] += scores_for_each_lengthcounts[key]
    #     elif key>20 and key<=25:
    #         length_group[4] += scores_for_each_lengthcounts[key]
    #     else:
    #         length_group[5] += scores_for_each_lengthcounts[key]

    # print("line numbers")
    # for key in length_group:
    #     print(len(length_group[key]))
    #     print(numpy.mean(length_group[key]))
    #     print("===")



    '''ascii art character counts'''
    # print(sorted(list(scores_for_each_charcounts.keys())))
    # keys = list(scores_for_each_charcounts)
    # values = [len(scores_for_each_charcounts[key]) for key in keys]
    # plt.figure(figsize=(10, 5))
    # plt.bar(keys, values)
    # plt.savefig("tmp.jpg")
    # exit()
    
    # char_group = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
    
    # for key in scores_for_each_charcounts:
    #     if key<=50:
    #         char_group[0] += scores_for_each_charcounts[key]
    #     elif key>50 and key<=100:
    #         char_group[1] += scores_for_each_charcounts[key]
    #     elif key>100 and key<=200:
    #         char_group[2] += scores_for_each_charcounts[key]
    #     elif key>200 and key<=400:
    #         char_group[3] += scores_for_each_charcounts[key]
    #     elif key>400 and key<=800:
    #         char_group[4] += scores_for_each_charcounts[key]
    #     elif key>800 and key<=1600:
    #         char_group[5] += scores_for_each_charcounts[key]
    #     else:
    #         char_group[6] += scores_for_each_charcounts[key]
    
    # print("character numbers")
    # for key in char_group:
    #     print(len(char_group[key]))
    #     print(numpy.mean(char_group[key]))
    #     print("===")
