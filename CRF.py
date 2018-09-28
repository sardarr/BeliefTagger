import argparse
import ast
import os

import nltk
import copy
from nltk.stem.wordnet import WordNetLemmatizer
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem.porter import *
from nltk.tokenize import StanfordTokenizer
#run the following line in terminal inside the stanford direcotry
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000

host='http://localhost'
port=9000
nlp = StanfordCoreNLP(host, port=port,timeout=30000)
stemmer = PorterStemmer()
# print( 'Tokenize:', nlp.word_tokenize(sentence))
# print( 'Part of Speech:', nlp.pos_tag(sentence))
# print ('Named Entities:', nlp.ner(sentence))
# print ('Constituency Parsing:', nlp.parse(sentence))
# print ('Dependency Parsing:', nlp.dependency_parse(sentence))



tagged_sentences = nltk.corpus.treebank.tagged_sents()




def gen_corp(read_file):
    sct_n_tag=[]
    tmp_sent=[]
    for line in read_file:
        if line!='\n':
            tmp_sent.append((line.split(' ')[0],line.split(' ')[1].split('\t')[0]))
        else:
            sct_n_tag.append(tmp_sent)
            tmp_sent = []
    return sct_n_tag

LDC_train=open("/Users/sardarhamidian/Google Drive/PHDthesis/WASSA/NewSeqTAgging/data/data_prp_main/14train.txt").readlines()
LDC_test=open("/Users/sardarhamidian/Google Drive/PHDthesis/WASSA/NewSeqTAgging/data/data_prp_main/14test.txt").readlines()

training_sentences=gen_corp(LDC_train)
test_sentences=gen_corp(LDC_test)
lmtzr = WordNetLemmatizer()


def verb_type(word,pos):
    """
    if the word is moodal/aux/reg or nothin
    :param word: Kill
    :param pos:  VB
    :return:  reg
    """
    # modal=['can','could','may','might','shall','should','will','would','must']
    aux=["be","am","are","is","was","were","being","been","can","could","dare","do","does","did","have","has","had","having","may","might","must","need","ought","shall","should","will","would"]
    if 'vb' not in pos[1].lower():
        return ['nil','nil']
    elif pos=="MD":
        return ['MD',word]
    elif word in aux:
        return ['AUX',word]
    else:
        return ['REG','nil']

def ancestor(sentence,dparse,pos,index):
    """

    :param sentence: The tokenized sentence
    :param dparse: the output of the dependency parset
    :param pos: the list of pos tags of a sentece
    :param index: index of the word in the sentence
    :return: if the ancestor verb is on of the reporting verbs that we have here
    """
    pverbs=('qoute','tell', 'accuse', 'insist', 'seem', 'believe', 'say',\
            'find', 'conclude', 'claim', 'trust', 'think', 'suspect', 'doubt',\
            'report','suppose')
    clausal=("csubj","ccomp","xcomp")
    ances={}
    ances["reporting_ancestor"]=0
    if 'vb' in pos[index][1].lower():
        for item in dparse:
            if item[0] in clausal and item[1] - 1 == index and sentence[item[2] - 1].lower() in pverbs:
                ances["reporting_ancestor"] = 1
                break
            elif item[0] in clausal and item[2] - 1 == index and sentence[item[1] - 1].lower() in pverbs:
                ances["reporting_ancestor"] = 2
                break
            elif item[0] in clausal and item[1]-1<index and sentence[item[1]-1] in pverbs:
                ances["reporting_ancestor"] = 3
                break


    return ances
def parent_pos(dparse,pos,index):
    """
    :param dparse: the output of the dependency parset
    :param pos: the list of pos tags of a sentece
    :param index: index of the word in the sentence
    :return: It returns the ancestors pos for two parents if they exist
    """
    par_pos={}
    par_pos["pos_mother1"]='nil'
    par_pos["pos_mother2"]='nil'

    for item in dparse:
        if item[2] - 1 == index and par_pos["pos_mother1"]=='nil':
            par_pos["pos_mother1"] = pos[item[1] - 1][1].lower()
        if item[2] - 1 == index and par_pos["pos_mother2"]=='nil':
            par_pos["pos_mother2"] = pos[item[1] - 1][1].lower()
    return par_pos
def tupple_allign(word_label,pos_list):
    """
    There are many cases that pos tokenizes 's but tokenization does't. Therefor there is no corresponding cb
    tag for the 's so what we do in this function is to consider 's as a seperate tag with cb with the last LCB label

    :param word_label:List od tokens and cb labels e.g [(he's,LCB)]
    :param pos: List of token with pos tags e.g [(he,pos)('s,pos)]
    :return: list of token with new cb labels e.g [(he,LCB)('s,LCB)]
    """
    aligned_wotd_label=[]
    for item in word_label:
        try:
            if item[0]==pos_list[0][0]   :
                aligned_wotd_label.append(item)
                pos_list.remove(pos_list[0])
            else:
                aligned_wotd_label.append((pos_list[0][0],item[1]))
                aligned_wotd_label.append((pos_list[1][0],item[1]))
                pos_list.remove(pos_list[0])
                pos_list.remove(pos_list[0])
        except:
            print("error")
    return aligned_wotd_label

def daughter(sentence,dparse,pos,index):
    """
    :param sentence: The tokenized sentence
    :param dparse: the output of the dependency parset
    :param pos: the list of pos tags of a sentece
    :param index: index of the word in the sentence
    :return: returns a difctionary of the daughter features in a dictionary format

    """
    aux=["be","am","are","is","was","were","being","been","can","could","dare","do","does","did","have","has","had",\
         "having","may","might","must","need","ought","shall","should","will","would"]
    perfect=('have','has','had')
    whs=("where", "when", "while", "who", "why","what","whom","whose")
    dghter={}
    dghter['daughter_to']=0
    dghter['daughter_perfect'] = 0
    dghter['daughter_should'] = 0
    dghter['daughter_wh'] = 0
    dghter['aux_r'] = 'nil'
    dghter['aux_l'] = 'nil'

    for item in dparse:
        #Perfect
        try:
            if item[1] - 1 == index and sentence[item[2] - 1].lower() in perfect:
                dghter['daughter_perfect'] = 1

            elif item[2] - 1 == index and sentence[item[1] - 1].lower() in perfect:
                dghter['daughter_perfect'] = 2
            #Should
            if item[1] - 1 == index and sentence[item[2] - 1].lower()=='should':
                dghter['daughter_should'] = 1
            elif item[2] - 1 == index and sentence[item[1] - 1].lower() =='should':
                dghter['daughter_should'] = 2
            #Wh*
            if item[1] - 1 == index and sentence[item[2] - 1].lower() in whs:
                dghter['daughter_wh'] = 1
            elif item[2] - 1 == index and sentence[item[1] - 1].lower() in whs:
                dghter['daughter_wh'] = 2
            if item[1] - 1 == index and sentence[item[2] - 1].lower() in aux:
                dghter['aux_r'] = sentence[item[2] - 1].lower()
            elif item[2] - 1 == index and sentence[item[1] - 1].lower() in aux:
                dghter['aux_l'] = sentence[item[1] - 1].lower()
        except:
            print(item)



    #To if it was a verb
    if 'vb' in pos[index][1].lower():
        for item in dparse:
            if item[1]-1==index and sentence[item[2]-1].lower()=='to':
                dghter['daughter_to']=1
            elif item[2]-1==index and sentence[item[1]-1].lower()=='to':
                dghter['daughter_to']=2

    return dghter





def features(words,word_lem,sentence,pos,dparse,index,nf):
    """ sentence: [w1, w2, ...], index: the index of the word """
    daughter_dict=daughter(sentence, dparse, pos, index)
    prnt_pos=parent_pos(dparse, pos, index)

    feature_dic= {
    'is_numeric': sentence[index].isdigit(),
    'pos':pos[index][1],
    'word_lem':word_lem[index],
    'verb_type':verb_type(word_lem[index],pos[index])[0]  ,
    'which_modal':verb_type(word_lem[index],pos[index])[1]  ,
    #################
    'daughter_to':daughter_dict['daughter_to'],
    'daughter_perfect':daughter_dict['daughter_perfect'],
    'daughter_should':daughter_dict['daughter_should'],
    'daughter_wh':daughter_dict['daughter_wh'],
    #################
    'repo_ances':ancestor(word_lem,dparse,pos,index)['reporting_ancestor'],
    #################
    'parent_posm':prnt_pos['pos_mother1'],
    'parent_posf':prnt_pos['pos_mother2'],
    #################
    'aux_r':daughter_dict['aux_r'],
    'aux_l':daughter_dict['aux_l'], }
    if nf==True:
        new_feature={
        ###################
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
        }
        feature_dic.update(new_feature)

    return feature_dic

from nltk.tag.util import untag
# Split the dataset for training and testing
# cutoff = int(.75 * len(tagged_sentences))
# training_sentences = tagged_sentences[:cutoff]
# test_sentences = tagged_sentences[cutoff:]
def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for tagged in tagged_sentences:
        # sent='CNN reported that Republican leader Bill Frist should have what is known to be dangerous.'
        # token=nlp.word_tokenize(sent)
        token=untag(tagged)
        sent = " ".join(token)
        pos=nlp.pos_tag(sent)
        pos_tag=copy.deepcopy(pos)
        tagged=tupple_allign(tagged,pos_tag)
        token=[x[0] for x in pos]
        assert len(tagged)  ==  len(pos)
        dparse=nlp.dependency_parse(sent)
        word_lem= [stemmer.stem(x) for x in token]
        nf=True
        X.append([features(sent,word_lem,token,pos,dparse, index,nf) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])
    return X, y
def pos_tag(model, sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))

def main():

    X_train, y_train = transform_to_dataset(training_sentences)
    X_test, y_test = transform_to_dataset(test_sentences)
    print(len(X_train))
    print(len(X_test))
    print(X_train[0])

    from sklearn_crfsuite import CRF
    model = CRF()
    model.fit(X_train, y_train)

    sentence = ['I', 'am', 'Bob', '!']






if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--tmode', type=str, default='ev',help='Transfer learning, evaluation or training regular model')
    # parser.add_argument('--dir_input', type=str, default='None',help='Transfer learning or regular model')
    # parser.add_argument('--ex_param', type=str, default='params/conv_param.txt',help='Transfer learning or regular model')

    main()
    # dinput="/Sardar_Summer/PhD/WASSA/NewSeqTAgging/data/test"
    # args = parser.parse_args()
    # # arg_path=args.args
    # if args.ex_param:
    #     params_file = open(args.ex_param).readlines()
    #     firstline = params_file[0]
    #
    # for parami in params_file:
    #     param = ast.literal_eval(parami)
    #     if not os.path.exists("old_models"):
    #         os.makedirs("old_models")
    #     main(param)

