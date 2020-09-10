
from sklearn.metrics import classification_report, f1_score
from roberta_k_fold import KFoldProcessor
import pickle
import argparse
import numpy as np
import json
import os
from itertools import combinations
from collections import Counter

def vote(predictions):
    '''
    投票融合方法
    :param predictions:
    :return:
    '''
    if len(predictions) == 1:  # 没有多个预测结果就直接返回第一个结果
        return predictions[0]
    result = []
    num = len(predictions[0])
    for i in range(num):
        temp = []
        for pred in predictions:
            temp.append(pred[i])
        counter = Counter(temp)
        result.append(counter.most_common()[0][0])
    return result

def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2

class Stacker(object):
    def __init__(self, args):
        self.args = args
        self.X_train_dir = args.X_train_dir
        self.y_train_dir = args.y_train_dir
        self.X_test_dir = args.X_test_dir

    def load_y_train(self):
        args = self.args
        processor = KFoldProcessor(args, data_dir=args.y_train_dir)
        train_examples = processor.get_train_examples()
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        targets = [label_map[example.label] for example in train_examples]
        return np.array(targets)

    def load_X_train(self, path):
        path = os.path.join(path, 'oof_train')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_X_test(self, path):
        path = os.path.join(path, 'oof_test')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def filter_path(self, paths):
        filter_names = ['last3', 'electra', 'xlnet', 'bnm', 'abs', 'raw', 'eda']
        result_paths = []
        for path in paths:
            if not any(fn in path for fn in filter_names):
                result_paths.append(path)
        # return result_paths
        # # usual best 线上0.785，线下0.7799
        result_paths = ['roberta_large', 'roberta_base',
                        'roberta_wwm_ext_large', 'roberta_wwm_ext_large_lstm_attention',
                        'roberta_wwm_ext_transfer_learning1', 'roberta_wwm_ext_large_attention',
                        'uer_large',
                        ]

        # virus 8-5 0.703
        # result_paths = [
        #     'roberta_large',
        #     'roberta_base',
        #     # 'roberta_wwm_ext',
        #     'roberta_base_transfer_learning1',
        #     'roberta_wwm_ext_transfer_learning',
        #     'roberta_wwm_ext_transfer_learning',
        #     'roberta_wwm_ext_transfer_learning1_k10',
        #     'roberta_wwm_ext_transfer_learning1_pseudo',
        # ]

        return result_paths

    def load_data(self):
        args = self.args
        self.y_train = self.load_y_train()
        self.labels = set(self.y_train)
        self.n_classes = len(self.labels)
        pathes = os.listdir(self.X_train_dir)
        pathes = self.filter_path(pathes)
        self.X_train = np.zeros((len(self.y_train), self.n_classes))
        for i, path in enumerate(pathes):
            self.X_train[:, :] += self.load_X_train(os.path.join(self.X_train_dir, path))
        self.X_train = self.X_train/len(pathes)
        # pathes = os.listdir(self.X_test_dir)
        # pathes = self.filter_path(pathes)
        n_test = len(self.load_X_test(os.path.join(self.X_test_dir,pathes[0])))
        n_train = len(self.y_train)
        self.X_test = np.zeros((n_test, self.n_classes))

        for i, path in enumerate(pathes):
            self.X_test[:, :] += self.load_X_test(os.path.join(self.X_train_dir, path))
        self.X_test = self.X_test/len(pathes)
        self.X_train = self.X_train.reshape(n_train, -1)
        self.X_test = self.X_test.reshape(n_test, -1)

    def get_result(self, preds, reals):
        f_score = f1_score(y_true=reals, y_pred=preds, average='macro')
        return f_score

    def stack(self,reload=True):
        if reload:
            self.load_data()
        pred_labels = np.argmax(self.X_train, axis=-1)
        # sep = int(len(self.y_train) * split)
        # clf.fit(self.X_train[:sep], self.y_train[:sep])
        # self.clf = clf
        # pred_labels = clf.predict(self.X_train[sep:])
        real_labels = self.y_train
        # print(pred_labels.shape)
        # print(real_labels.shape)
        f_score = self.get_result(list(pred_labels.tolist()), list(real_labels.tolist()))
        return f_score

    def predict(self):
        print(self.X_test.shape)
        preds = np.argmax(self.X_test, axis=-1)
        return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = 'usual'
    parser.add_argument('--X_train_dir', type=str, default=mode+'_k_fold_model',
                        help='模型的预测输出位置，应为一个examples nums * class nums的矩阵')
    parser.add_argument('--y_train_dir', type=str, default='k_fold/'+mode,
                        help='模型的真实数据，包含句子的真正标签')
    parser.add_argument('--X_test_dir', type=str, default=mode+'_k_fold_model',
                        help='模型的test数据，生成的oof_test')
    parser.add_argument('--output_dir', type=str, default='voting_results/'+mode,
                        help='输出的目录')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    st = Stacker(args)
    st.load_data()
    f_score = st.stack()
    print(f_score)
    preds = st.predict()

    preds = preds.tolist()

    label_list = ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
    label_map = {label:id for id, label in enumerate(label_list)}
    # 计算和之前test最佳的f score
    # with open('voting_results2/virus/virus_result.txt', 'r', encoding='utf8') as f:
    #     target = json.load(f)
    # target_label = [int(label_map[example['label']]) for example in target]
    # print("相似度")
    # print(f1_score(y_true=target_label, y_pred=preds, average='macro'))

    with open('k_fold/'+mode+'/eval.txt') as f:
        content_data = json.load(f)
    results = []
    for i, pred in enumerate(preds, start=1):
        if len(content_data[i-1]['content']) == 0:
            label = 'neural'
            print(i)
            print(content_data[i-1])
        else:
            label = label_list[pred]
        results.append({'id': i,
                        "label": label})

    with open(os.path.join(args.output_dir, mode+'_result.txt'), 'w', encoding='utf8') as fw:
        json.dump(results, fw, indent=4, ensure_ascii=False)

