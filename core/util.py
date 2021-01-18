from ternary.common.utils import kd_cut
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from itertools import combinations
import copy
from scipy import sparse
import collections
from multiprocessing import cpu_count, Pool
import re
import queue

from networkx import all_pairs_shortest_path_length


def parallelize_dataframe(data, func, num_cores=cpu_count()):
    partitions = num_cores
    data_split = np.array_split(data, partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return df


def clean_text(text):
    text = str(text)
#     rule = re.compile(r"[^a-zA-Z0-9\u3e00-\u4e00]")
#     text = re.sub(rule, "", text)
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text = text.replace('<br />', ' ')
    text = text.replace(';', ',')
    text = text.replace('\xa0', '')
    text = text.replace('\u3000', '')
    text = text.replace('&amp', '')
    text = text.replace('\r', '')
    text = text.replace('\t', '')
    return text


def tokenize(text):
    return list(kd_cut(clean_text(text), True))


def get_sentences(text):
    text = clean_text(text)
    sentences = re.split('(。|！|\!|？|\?|，|,)', text)
    new_sents = []
    if len(sentences) == 1:
        return sentences
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        new_sents.append(sent)
    return new_sents


def get_sentence_df_content(data):
    return data.assign(content_tokens=data["content"].map(get_sentences)).assign(title_tokens=data['title'].map(get_sentences))


def tokenize_to_string(text):
    res = list(kd_cut(clean_text(text), True))
    return ' '.join(res)


def convert_str_to_array(x):
    res = []
    x = x.replace("[", "")
    x = x.replace("]", "")
    x = x.replace("'", "")
    for c in x.split(" "):
        res.append(c)
    return res


def tokenize(text):
    text = str(text)
    text = text.replace('\n', '')
    text = text.replace('<br />', ' ')
    text = text.replace(';', ',')
    words = kd_cut(str(text), True)
    return list(words)


def news_to_corpus(news, enable_stop_words=True):

    words = kd_cut(news, enable_stop_words=True)
    words = list(words)
    return " ".join(words)


def get_top_n_tfidf(tfidf, n_top, n_features=5000):
    """
    :param tfidf: tfidf value（sparse matrix）
    :param n_top:
    :param n_features:
    :return: The tfidf value of n_top before the tfidf value, the other positions are 0
    """
    descend_index = np.array(np.argsort(- tfidf.todense()))[0]
    topn_descend_index = descend_index[:n_top]
    topn_value = np.array(tfidf.todense())[0][topn_descend_index]  # 前n个tfidf的值

    col = np.array([0] * n_top)
    topn_tfidf = csr_matrix((topn_value, (col, topn_descend_index)), shape=(1, n_features))

    return topn_tfidf


def build_label_str(arr, sep = "|"):
    s = ""
    for i in arr:
        s += str(i) + sep
    if len(s) > 0 and s[-1] == sep:
        s = s[:-1]
    return s


def construct_dict_category_label(df_category_label):
    dict_category_label = {}
    for i in df_category_label.index:
        s = df_category_label.loc[i]
        arr = []
        for x in df_category_label.columns[1:]:
            if s[x] != "":
                arr.append(s[x])
        dict_category_label[s['label']] = arr
    return dict_category_label


def construct_path_labels(labels, dict_category_label):
    arr = []
    for x in labels:
        if x == "": continue
        if x not in dict_category_label:
            print("Cannot find label ", x)
            continue
        path_labels = dict_category_label[x]
        for r in path_labels:
            if r not in arr:
                arr.append(r)
    return arr


def get_cluster_id_index(df, labels_predict):
    """
    :param df:
    :param labels_predict: label prediction results
    :return: A dictionary that stores the news subscript corresponding to the label
    """
    dict_cluster_id_to_index = {}
    cluster_id_list = list(set(labels_predict))
    for i in cluster_id_list:
        result = df[labels_predict == i].index.tolist()
        dict_cluster_id_to_index[i] = result
    return dict_cluster_id_to_index


def calculate_matrix_value(cluster_id_index, dist_matrix):
    """
    :param cluster_id_index: Cluster index dict of class
    :param dist_matrix: Distance matrix
    :return: dict_matrix_value is the value of each class index binary pair in the distance matrix
    """
    # dict_matrix_combinations records the tuple value of each class
    # index to facilitate the output of the specified two-tuple
    dict_matrix_combinations = {}
    dict_matrix_value = {}
    for key, value in cluster_id_index.items():
        if len(value) == 1:
            dict_matrix_combinations[key] = [(value[0], value[0])]
        else:
            dict_matrix_combinations[key] = list(combinations(value, 2))

    for key, value in dict_matrix_combinations.items():
        res = []
        for i in value:
            res.append(dist_matrix[i[0]][i[1]])
        dict_matrix_value[key] = res

    for key in list(dict_matrix_value.keys()):
        if not dict_matrix_value.get(key):
            dict_matrix_value[key] = [0]

    return dict_matrix_value, dict_matrix_combinations


def calculate_inner_max_distance(dict_matrix_value, dict_matrix_combinations):
    # Calculate the maximum distance within the class and the corresponding two-tuple
    dict_inner_cluster_maximun_distance = {}
    dict_maximun_distance_pair_index = {}
    dict_maximun_distance_pair = {}

    for key, value in dict_matrix_value.items():
        dict_inner_cluster_maximun_distance[key] = max(value)
        dict_maximun_distance_pair_index[key] = value.index(max(value))

    for key, value in dict_maximun_distance_pair_index.items():
        dict_maximun_distance_pair[key] = dict_matrix_combinations[key][value]

    return dict_inner_cluster_maximun_distance, dict_maximun_distance_pair


def reorder_predict_labels(labels_predict):
    """
    Prevent the news under cluster_id from being 0 due to iterative clustering
    :param labels_predict: Clustered label
    :return: labels_predict
    """
    labels_predict_copy = copy.deepcopy(labels_predict)
    label_list = sorted(list(set(labels_predict_copy)))  # 升序
    counter = 0
    for label in label_list:
        labels_predict_copy[labels_predict_copy == label] = counter
        counter += 1

    return labels_predict_copy

def get_feature_words(tfidf_value, feature_list):
    """
    get features to analysis
    :param tfidf_value:
    :return: Features corresponding to tfidf values, separated by spaces
    """
    if type(tfidf_value) == sparse.csr_matrix:
        tfidf_value = np.array(tfidf_value.todense())[0]
    # OOV
    len_features = np.nonzero(tfidf_value)[0].shape[0]

    tfidf_descend_index = np.argsort(-tfidf_value)
    word_list = np.array(feature_list)[tfidf_descend_index][:len_features]

    return " ".join(word_list)


def get_target_word(corpus_string, ner_vector, subname):
    words_list = corpus_string.split()
    ner_list = ner_vector.split()

    target_words = [words_list[i] for i in range(len(ner_list)) if subname in ner_list[i]]

    return " ".join(target_words)


def get_top_words(text, topn):
    frequency = collections.defaultdict(int)
    for word in text.split():
        frequency[word] += 1
    if topn <= len(text.split()):
        topn_tuple_list = sorted(frequency.items(), key=lambda item: item[1], reverse=True)[:topn]
    else:
        topn_tuple_list = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    topn_words = [tup[0] for tup in topn_tuple_list]

    return " ".join(topn_words)


def get_topn_target(corpus_string, ner_vector, subname, topn):
    target_words_str = get_target_word(corpus_string, ner_vector, subname)

    topn_target_words_str = get_top_words(target_words_str, topn)
    return topn_target_words_str


def get_ft_avg(doc_corpus, ft_model):
    words_list = doc_corpus.split()
    ft_sum = np.zeros(len(ft_model[" "]))

    for word in words_list:
        ft_sum = ft_model[word] + ft_sum
    if len(words_list) == 0:
        return ft_sum
    return ft_sum / len(words_list)


def callable_matrix(str1, str2):
    if len(str1.split()) == 0 and len(str2.split()) == 0:
        return 0.0
    elif len(str1.split()) == 0 or len(str2.split()) == 0:
        return 1.0
    else:
        for i in str1.split():
            if i in str2.split():
                return 0.0
            else:
                return 1.0


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def convert_event_category_json_to_dict(json):
    key = "ROOT"
    q = queue.Queue()
    hierarchy_dict = {}
    values = []
    for child in json["children"]:
        values.append(child['className'])
        q.put(child)
    hierarchy_dict[key] = values
    while not q.empty():
        node = q.get()
        key = node['className']
        values = []
        for child in node["children"]:
            values.append(child['className'])
            q.put(child)
        if len(values) > 0:
            hierarchy_dict[key] = values
    return hierarchy_dict


def find_array_index(array, element):
    for i in range(len(array)):
        if array[i] == element:
            return i
    return -1


def fill_ancestors(y, graph, labels, copy=True):
    """
    Compute the full ancestor set for y given as a matrix of 0-1.
    Each row will be processed and filled in with 1s in indexes corresponding
    to the (integer) id of the ancestor nodes of those already marked with 1
    in that row, based on the given class hierarchy graph.
    :param y: array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids.
    :param graph: the class hierarchy graph, given as a `networkx.DiGraph` instance
    :param labels: labels corresponding to the columns of matrix y
    :param copy: boolean
    :return: array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids with
        all ancestors of existing labels in matrix filled in, per row.
    """
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        idx = find_array_index(labels, target)
        if idx == -1:
            continue
        if sum(y[:, idx]) == 0:
            continue
        ix_rows = np.where(y[:, idx] > 0)[0]
        ancestors = list(distances.keys())
        for n in ancestors:
            n_idx = find_array_index(labels, n)
            if n_idx != -1:
                y_[ix_rows, n_idx] = 1
        # y_[np.meshgrid(ix_rows, ancestors)] = 1
    graph.reverse(copy=False)
    return y_


def h_precision_score(y_true, y_pred, class_hierarchy, labels):
    """
    :param y_true: array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.
    :param y_pred: array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.
    :param class_hierarchy: the class hierarchy graph, given as a `networkx.DiGraph` instance
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.
    :param labels: array of label names
        the labels are corresponding to the columns of matrix y
    :return: float
        The computed hierarchical precision score.
    """
    y_true_ = fill_ancestors(y_true, graph=class_hierarchy, labels=labels)
    y_pred_ = fill_ancestors(y_pred, graph=class_hierarchy, labels=labels)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    return true_positives / all_results


def h_recall_score(y_true, y_pred, class_hierarchy, labels):
    """
    :param y_true: array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.
    :param y_pred: array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.
    :param class_hierarchy: the class hierarchy graph, given as a `networkx.DiGraph` instance.
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.
    :param labels: array of label names
        the labels are corresponding to the columns of matrix y
    :return: float
        The computed hierarchical recall score.
    """
    y_true_ = fill_ancestors(y_true, graph=class_hierarchy, labels=labels)
    y_pred_ = fill_ancestors(y_pred, graph=class_hierarchy, labels=labels)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)

    return true_positives / all_positives


def hierarchy_metrics(y_true, y_pred, class_hierarchy, labels, beta=1.):
    """
    Calculate the hierarchical metrics based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.
    For motivation and definition details, see:
        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf
    :param y_true: array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.
    :param y_pred: array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.
    :param class_hierarchy: the class hierarchy graph, given as a `networkx.DiGraph` instance
        Node ids must be integer and correspond to the indices into the y_true / y_pred matrices.
    :param labels: array of label names
        the labels are corresponding to the columns of matrix y
    :param beta: float
        the beta parameter for the F-beta score. Defaults to F1 score (beta=1).
    :return: tuple of floats
        Tuple of classification metrics (h-precision, h-recall, h-fscore)
    """
    hP = h_precision_score(y_true, y_pred, class_hierarchy, labels=labels)
    hR = h_recall_score(y_true, y_pred, class_hierarchy, labels=labels)
    hF = (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
    return hP, hR, hF


def get_HMDScore(y_prob, y_test, penalty_coefficient, threshold=0.15, zoom_penalty=1000):
    y_pred = (y_prob > threshold).astype(int)
    y_true = np.stack([y_test] * y_test.shape[1], axis=2)
    y_pred = np.stack([y_pred] * y_test.shape[1], axis=1)
    hmd = zoom_penalty * np.mean(np.mean(np.mean(np.square(y_true - y_pred) * penalty_coefficient, axis=-1), axis=-1))
    hmdscore = 1 - np.tanh(hmd)
    return hmdscore


