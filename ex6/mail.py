import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import svm
import re #regular expression for e-mail processing
# 这个英文算法似乎更符合作业里面所用的代码，与上面效果差不多
import nltk, nltk.stem.porter

with open('emailSample1.txt', 'r') as f:
    email = f.read()
    print(email)

def processEmail(email):
    """做除了Word Stemming和Removal of non-words的所有处理"""
    email = email.lower()
    email = re.sub('<[^<>]>', ' ', email)  # 匹配<开头，然后所有不是< ,> 的内容，知道>结尾，相当于匹配<...>
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email )  # 匹配//后面不是空白字符的内容，遇到空白字符则停止
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email

def email2TokenList(email):
    """预处理数据，返回一个干净的单词列表"""
    # I'll use the NLTK stemmer because it more accurately duplicates the
    # performance of the OCTAVE implementation in the assignment
    stemmer = nltk.stem.porter.PorterStemmer()
    email = processEmail(email)
    # 将邮件分割为单个单词，re.split() 可以设置多种分隔符
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    # 遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token);
        # Use the Porter stemmer to 提取词根
        stemmed = stemmer.stem(token)
        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue
        tokenlist.append(stemmed)
    return tokenlist

def email2VocabIndices(email, vocab):
    """提取存在单词的索引"""
    token = email2TokenList(email)
    index = [i for i in range(len(vocab)) if vocab[i] in token ]
    return index

def email2FeatureVector(email):
    """将email转化为词向量，n是vocab的长度。存在单词的相应位置的值置为1，其余为0"""
    df = pd.read_table('vocab.txt',names=['words'])
    vocab = df.as_matrix()  # return array
    vector = np.zeros(len(vocab))  # init vector
    vocab_indices = email2VocabIndices(email, vocab)  # 返回含有单词的索引
    # 将有单词的索引置为1
    for i in vocab_indices:
        vector[i] = 1
    return vector
vector = email2FeatureVector(email)
print('length of vector = {}\nnum of non-zero = {}'.format(len(vector), int(vector.sum())))

spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

svc = svm.SVC()
svc.fit(X, y)
print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))

kw = np.eye(1899)
kw[:3,:]
spam_val = pd.DataFrame({'idx':range(1899)})
spam_val['isspam'] = svc.decision_function(kw)
spam_val['isspam'].describe()
decision = spam_val[spam_val['isspam'] > -0.55]
path ='vocab.txt'
voc = pd.read_csv(path, header=None, names=['idx', 'voc'], sep = '\t')
spamvoc = voc.loc[list(decision['idx'])]
print(spamvoc)