import jieba
import jieba.posseg
import math
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization.bm25 import BM25
import numpy as np
from transformers import BertTokenizer, BertModel

# 停用词表
stopwords = ["了", "啊", "吗", "的", "呢", "哈", "呵", "，", "。", "、", "？", "！", "：", "“", "”"]


# 分词
def Process(crops):
    filter_document = []
    for text in crops:
        segment = jieba.posseg.cut(text.strip())
        filter_words = []
        for word, flag in segment:
            if not word in stopwords and len(word) >= 1:
                filter_words.append(word)
        filter_document.append(filter_words)
    return filter_document


"""
TF-IDF文本相似度计算相关函数
"""


# 计算IDF
def calIDF(filter_document):
    idf_dict = {}
    for text in filter_document:
        for word in set(text):
            if word not in idf_dict.keys():
                idf_dict[word] = 1
            else:
                idf_dict[word] += 1
    for word in idf_dict.keys():
        idf_dict[word] = math.log((len(file_document) / (idf_dict[word] + 1)), 2)
    # print("idf",idf_dict)
    return idf_dict


# 计算TF
def calTF(text):
    tf_dict = {}
    for word in text:
        if word not in tf_dict:
            tf_dict[word] = 1
        else:
            tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(text)
    # print("tf",tf_dict)
    return tf_dict


# 计算TF-IDF
def calTFIDF(text, idf_dict, tf_dict):
    tf_idf_dict = {}
    for word in text:
        if word not in idf_dict:
            idf_dict[word] = 0
        tf_idf_dict[word] = tf_dict[word] * idf_dict[word]
    # print(tf_idf_dict)
    return tf_idf_dict


"""
bert-whitening相关函数
"""
# 模型与tokenizer导入
model = BertModel.from_pretrained("../../module/bert")
tokenizer = BertTokenizer.from_pretrained("../../module/bert")


# 计算kernel与bias
def compute_kernel_bias(vecs, n_components):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    W = W[:, :n_components]
    return W, -mu


# 归一化
def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


# 获取文档集的文本相似度矩阵
def get_similarity_bert(crops):
    vecs = []
    for text in crops:
        # 由于bert模型限制，需要切分文本长度为512
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model(**inputs)
        # 使用句子级别文本表示
        pooler_output = outputs[1]
        vecs.append(pooler_output.detach().numpy())
    vecs = np.concatenate(vecs, axis=0)
    kernel,bias=compute_kernel_bias(vecs,128)
    normalized_vecs = transform_and_normalize(vecs, kernel, bias)
    similarity_matrix = cosine_similarity(normalized_vecs)
    return similarity_matrix


if __name__ == "__main__":
    # 文档集路径
    data_path = "../../data/homework_4/data1"
    # 文档集构建
    crops = []
    for i in range(1, 6):
        file = open(data_path + "/file" + str(i) + ".txt", 'r', encoding='utf-8').readlines()
        text = ""
        for line in file:
            text = text + line
        crops.append(text)
    # 分词
    file_document = Process(crops)
    print(file_document)


    print("--------------------tf-idf-similarity-------------------")
    # 每个单词idf计算
    idfDic = calIDF(file_document)
    word_list = list(idfDic.keys())
    words_bag = []
    cnt = 0
    for text in file_document:
        # 文本分别统计每个单词的词频
        tfDic = calTF(text)
        # 计算每个单词TF-IDF
        tfIdfDic = calTFIDF(text, idfDic, tfDic)
        # print(tfIdfDic)
        tmp_bag = []
        for word in word_list:
            if word in tfIdfDic.keys():
                tmp_bag.append(tfIdfDic[word])
            else:
                tmp_bag.append(0)
        words_bag.append(tmp_bag)
    # print(words_bag)
    tf_idf_similarity = cosine_similarity(words_bag)
    print(tf_idf_similarity,"\n")

    print("--------------------BM25-similarity-------------------")
    bm25_similarity = []
    bm25 = BM25(file_document)
    for query in file_document:
        scores = bm25.get_scores(query)
        bm25_similarity.append(scores)
    print(np.array(bm25_similarity),"\n")




    print("--------------------bert-whitening-similarity-------------------")
    vecs = []
    for text in crops:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        outputs = model(**inputs)
        pooler_output = outputs[1]
        vecs.append(pooler_output.detach().numpy())
    vecs = np.concatenate(vecs, axis=0)
    similarity_bert = get_similarity_bert(crops)
    print(similarity_bert,"\n")
