# coding: utf-8

import gensim, logging
import smart_open, os
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#
# def words(text):
#     pattern = r'( |\n|\r|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|，|、|：|；|‘|’|【|】|·|！| |…|（|）)'
#     return re.split(pattern, text)


# def read_file(file_path):
#     result = []
#     pattern = r'(\t|\n|\r|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|，|、|：|；|‘|’|【|】|·|！| |…|（|）)'
#
#     with open(file_path, "r", encoding='UTF-8') as file:
#         line_str = file.readlines()
#         for line in line_str:
#             line_list=[]
#             my_str = re.sub(pattern, " ", line)
#             line2 = my_str.replace(pattern, " ").replace(" +", " ")
#             attr = line2.split(" ")
#             for word in attr:
#                 if len(word) > 0:
#                     line_list.append(word)
#
#
#     return result


def read_file(file_path):
    with open(file_path, "r", encoding='UTF-8') as file:
        lines = file.readlines()
        full_str = ""
        for line in lines:
            if line.strip() == "" or line.strip().startswith("--"):
                continue
            full_str += line.lower()
        return full_str


def get_file_info_list(file_name):
    full_str = remove_noise(read_file(file_name))
    info = full_str.split(";")
    new_info = []
    for each in info:
        line_list = []
        words = each.split(" ")
        for word in words:
            if len(word) > 0:
                line_list.append(word)

        line_list.append(";")
        new_info.append(line_list)

    return new_info


def remove_noise(file_str):
    pattern = r'(\t|\n|\r|,|\.|/|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|，|、|：|；|‘|’|【|】|·|！| |…|（|）)'

    str2 = re.sub(pattern, " ", file_str)
    return re.sub(" +", " ", str2).strip()


def train(sentences):
    model = gensim.models.Word2Vec(sentences, min_count=5)
    # In[7]:

    print(model)
    print(model.wv.vocab)

    # print(model.most_similar(positive=['select', '*', 'from'], topn=5))
    # model.doesnt_match("input is lunch he sentence cat".split())
    # print(model.similarity('human', 'party'))
    # print(model.similarity('tree', 'murder'))
    print(model.predict_output_word(['select', '*', 'from']))
    print(model.predict_output_word(['user', 'name']))
    print(model.predict_output_word(['select', '*', 'from', 'stl']))
    print(model.predict_output_word(['errors','order']))
    # model['tree']


if __name__ == '__main__':
    src_folder = "e://ocr//test//cs_1_1_1_zhanghm_1010010001_0_158.156.1.123_db.ppm_98.254.1.153_1521"
    # src_folder = "e://ocr//new0921//cs_1_1_1_zhanghm_1010010001_0_158.156.1.123_db.ppm_98.254.1.153_1521"
    answer_file = "E://ocr//testsql.txt"
    train_list = get_file_info_list(answer_file)

    train(train_list)
