import json
import os
from bert_serving.client import BertClient
from tqdm import tqdm
from conf.conf import *
from core.util import *

import keras
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.models import Sequential
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
import matplotlib.pyplot as plt

from networkx import DiGraph
from ternary.htc.metrics import *

tqdm.pandas()


class FHMTC(object, metaclass=Singleton):

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        KTF.set_session(sess)
        self.df_train = pd.read_csv(DATA_DIR + 'fmc_train.csv', delimiter=',', encoding='utf-8')
        self.df_test = pd.read_csv(DATA_DIR + 'fmc_test.csv', delimiter=',', encoding='utf-8')
        self.y_train = np.loadtxt(DATA_DIR + 'fmc_y_train.csv', delimiter=',', encoding='utf-8')
        self.y_test = np.loadtxt(DATA_DIR + 'fmc_y_test.csv', delimiter=',', encoding='utf-8')
        self.labels = np.loadtxt(DATA_DIR + 'fmc_labels.csv', delimiter=',', dtype='str', encoding='utf-8')
        self.event_category_json = open(DATA_DIR+"fmc_event_category.json", encoding="utf-8")
        self.event_category_json = json.load(self.event_category_json)

    @staticmethod
    def get_sentence_mean_embedding(text):
        bc = BertClient(ip='192.168.100.201', port=5555, port_out=5556, check_length=False)
        threshold = 150
        if len(text) > threshold:
            text = text[:threshold]
        vec = bc.encode(text)
        return np.mean(vec, axis=0)

    @staticmethod
    def get_label_level(hier_label_dict):
        label_level = dict()
        label_level['ROOT'] = 0
        for key in hier_label_dict.keys():
            for item in hier_label_dict[key]:
                label_level[item] = label_level[key] + 1
        return label_level

    @staticmethod
    def get_parent_label_index_list(hier_label_dict, labels):
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        parent_label_index_list = [0] * len(labels)
        for parent in hier_label_dict.keys():
            if parent == 'ROOT':
                continue
            parent_index = labels.index(parent)
            parent_label_index_list[parent_index] = parent_index
            for child in hier_label_dict[parent]:
                child_index = labels.index(child)
                parent_label_index_list[child_index] = parent_index
        return parent_label_index_list

    @staticmethod
    def get_hier_label_dict_child(hier_label_dict):
        hier_label_dict_child = dict()
        for key in list(hier_label_dict.keys())[::-1]:
            hier_label_dict_child[key] = hier_label_dict[key]
            for item in hier_label_dict[key]:
                if item in hier_label_dict.keys():
                    hier_label_dict_child[key] += hier_label_dict[item]
        for key in hier_label_dict_child.keys():
            hier_label_dict_child[key] = list(set(hier_label_dict_child[key]))
        return hier_label_dict_child

    @staticmethod
    def get_coefficient_matrix(label_level, labels, hier_label_dict_child, penalty_base_parent=ALPHA_A,
                               penalty_base_child=ALPHA_D, penalty_base_other=ALPHA_O, penalty_ratio=ALPHA_P):
        """
        Used to construct the penalty coefficient matrix, which is convenient for direct application in loss calculation
        """
        dim = len(labels)
        # Initialize the vector with the others penalty coefficient
        coefficient_matrix = np.full((dim, dim), penalty_base_other)
        for key in labels:
            if key == 'ROOT':
                continue
            #         coefficient_matrix[labels.index(key)][labels.index(key)] = 0
            # Try to assign only diagonal values
            coefficient_matrix[labels.index(key)][labels.index(key)] = 1
            if key not in hier_label_dict_child.keys():
                continue
            for item in hier_label_dict_child[key]:
                # Constructing offspring punishment coefficient
                coefficient_matrix[labels.index(key)][labels.index(item)] = penalty_base_child * (
                            penalty_ratio ** (label_level[item] - label_level[key]))
                # Construct ancestor penalty coefficient
                coefficient_matrix[labels.index(item)][labels.index(key)] = penalty_base_parent * (
                            penalty_ratio ** (label_level[item] - label_level[key]))
        return coefficient_matrix

    def train(self):
        df_tokens_train = parallelize_dataframe(self.df_train, get_sentence_df_content, 10)
        df_tokens_test = parallelize_dataframe(self.df_test, get_sentence_df_content, 10)

        wv_title_feature_train = df_tokens_train['title_tokens'].progress_apply(self.get_sentence_mean_embedding)
        wv_content_feature_train = df_tokens_train['content_tokens'].progress_apply(self.get_sentence_mean_embedding)
        wv_title_feature_test = df_tokens_test['title_tokens'].progress_apply(self.get_sentence_mean_embedding)
        wv_content_feature_test = df_tokens_test['content_tokens'].progress_apply(self.get_sentence_mean_embedding)

        X_train = np.column_stack([
            np.array(list(wv_title_feature_train)),
            np.array(list(wv_content_feature_train))
        ])

        X_test = np.column_stack([
            np.array(list(wv_title_feature_test)),
            np.array(list(wv_content_feature_test))
        ])

        print("X_train: ", X_train.shape)
        print("y_train: ", self.y_train.shape)
        print("X_test: ", X_test.shape)
        print("y_test: ", self.y_test.shape)

        train_labels = to_categorical(np.asarray(self.y_train))

        hier_label_dict = convert_event_category_json_to_dict(self.event_category_json)
        label_level = self.get_label_level(hier_label_dict)
        hier_label_dict_child = self.get_hier_label_dict_child(hier_label_dict)
        parent_label_index_list = self.get_parent_label_index_list(hier_label_dict, self.labels)
        coefficient_matrix = self.get_coefficient_matrix(
            label_level=label_level,
            labels=self.labels.tolist(),
            hier_label_dict_child=hier_label_dict_child,
        )

        model = Sequential()
        model.add(Dense(DENSE, input_dim=X_train.shape[1], activation="relu",))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(train_labels.shape[1], activation='sigmoid'))
        model.summary()

        def hmlc_rr_loss(y_true, y_pred, penalty=DECAY):
            penalty_coefficient = tf.constant(coefficient_matrix, dtype=tf.float32)
            y_true = K.stack([y_true] * self.labels.shape[0], axis=2)
            y_pred = K.stack([y_pred] * self.labels.shape[0], axis=1)
            hmlc_res = K.mean(K.sum(K.square(y_true - y_pred) * penalty_coefficient, axis=-1), axis=-1)

            w_child = model.layers[-1].weights[0]
            w_parent = tf.gather(model.layers[-1].weights[0], parent_label_index_list, axis=-1)
            recursive_regularization = penalty * 0.5 * K.sum(K.square(w_parent - w_child))

            return hmlc_res + recursive_regularization

        adam = keras.optimizers.Adam(lr=LEARNING_RATE)
        model.compile(loss=hmlc_rr_loss,
                      optimizer=adam,
                      metrics=['acc'])

        history = model.fit(X_train, self.y_train, epochs=30, batch_size=64, validation_data=(X_test, self.y_test))

        plt.figure(figsize=(10, 8))
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs[1:], loss[1:], '-.', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, '-*', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        y_prob = model.predict(X_test)
        y_pred = (y_prob > PROB_RATE).astype(int)

        y_labels = np.loadtxt(DATA_DIR + 'fmc_labels.csv', delimiter=',', dtype='str', encoding='utf-8')
        hierarchy_graph = DiGraph(convert_event_category_json_to_dict(self.event_category_json))

        print("HTC Evaluation:")
        h_precision, h_recall, h_fscore = hierarchy_metrics(self.y_test, y_pred, hierarchy_graph, y_labels)
        print("h_precision: %2.4f" % h_precision)
        print("h_recall: %2.4f" % h_recall)
        print("h_fscore: %2.4f" % h_fscore)

        hmdscore = get_HMDScore(y_prob=y_prob, y_test=self.y_test, penalty_coefficient=coefficient_matrix,
                                threshold=HMDSCORE_THRESHOLD)
        print("HMDScore : %2.4f" % hmdscore)



