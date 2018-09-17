import nltk

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

LDC_train=open("/Users/sardarhamidian/Google Drive/PHDthesis/WASSA/NewSeqTAgging/data/data_prp/14train.txt").readlines()
LDC_test=open("/Users/sardarhamidian/Google Drive/PHDthesis/WASSA/NewSeqTAgging/data/data_prp/14test.txt").readlines()

training_sentences=gen_corp(LDC_train)
test_sentences=gen_corp(LDC_test)


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
    'word': sentence[index],
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
    'is_numeric': sentence[index].isdigit(),
    'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

from nltk.tag.util import untag
# Split the dataset for training and testing
# cutoff = int(.75 * len(tagged_sentences))
# training_sentences = tagged_sentences[:cutoff]
# test_sentences = tagged_sentences[cutoff:]
def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])
    return X, y

X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(test_sentences)
print(len(X_train))
print(len(X_test))
print(X_train[0])
print(y_train[0])

from sklearn_crfsuite import CRF
model = CRF()
model.fit(X_train, y_train)

sentence = ['I', 'am', 'Bob', '!']


def pos_tag(sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))


print(pos_tag(sentence))  # [('I', 'PRP'), ('am', 'VBP'), ('Bob', 'NNP'), ('!', '.')]

