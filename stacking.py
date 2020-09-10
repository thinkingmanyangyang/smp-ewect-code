from sklearn.externals import joblib
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from roberta_k_fold import KFoldProcessor
import pickle
import argparse
import numpy as np
import json
import os
from tqdm import tqdm

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
        filter_names = ['last3', 'electra', 'xlnet', 'attention_bnm', 'bnm', 'abs', 'raw', 'uer']
        result_paths = []
        for path in paths:
            if not any(fn in path for fn in filter_names):
                result_paths.append(path)
        # usual
        # result_paths = ['roberta_large', 'roberta_base',
        #                 'roberta_wwm_ext_large', 'roberta_wwm_ext_large_lstm_attention',
        #                 'roberta_wwm_ext_transfer_learning1', 'roberta_wwm_ext_large_attention',
        #                 ]
        # virus
        # result_paths = [
        #     'roberta_large',
        #     'roberta_base',
        #     'roberta_wwm_ext',
        #     'roberta_wwm_ext_140',
        #     'roberta_wwm_ext_large',
        #     'roberta_wwm_ext_transfer_learning',
        #     'roberta_wwm_ext_transfer_learning2',
        #     'roberta_wwm_ext_transfer_learning3',
        #     'roberta_wwm_ext_transfer_learning4',
        #     'roberta_wwm_ext_transfer_learning1_k10',
        #     'roberta_wwm_ext_eda_lstm_attention_transfer_learning1',
        #     'roberta_wwm_ext_eda_attention_transfer_learning1',
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
        self.X_train = np.zeros((len(self.y_train), len(pathes), self.n_classes))
        for i, path in enumerate(pathes):
            X_train_temp = self.load_X_train(os.path.join(self.X_train_dir, path))
            self.X_train[:, i, :] = X_train_temp

        pathes = os.listdir(self.X_test_dir)
        pathes = self.filter_path(pathes)
        n_test = len(self.load_X_test(os.path.join(self.X_test_dir,pathes[0])))
        n_train = len(self.y_train)
        self.X_test = np.zeros((n_test, len(pathes), self.n_classes))
        for i, path in enumerate(pathes):
            X_test_temp = self.load_X_test(os.path.join(self.X_train_dir, path))
            self.X_test[:, i, :] = X_test_temp

        self.X_train = self.X_train.reshape(n_train, -1)
        self.X_test = self.X_test.reshape(n_test, -1)

    def get_result(self, preds, reals):
        f_score = f1_score(y_true=reals, y_pred=preds, average='macro')
        # f_score = accuracy_score(y_true=reals, y_pred=preds)
        return f_score

    def stack(self, clf, split=0.7, reload=True):
        if reload:
            self.load_data()
        param_test1 = {
                        'n_estimators': range(20, 81, 10),
                       'max_depth':range(8,16,3),
                       'min_samples_split':range(45,85,5),
                       'min_samples_leaf':range(1,25,5),
                       'max_features': range(1, 15, 3),
                       }
        # virus best param
        # best_param = {'max_depth': 13, 'max_features': 5, 'min_samples_leaf': 10, 'min_samples_split': 70, 'n_estimators': 50}
        # usual best param
        # best param = {'max_features': 7, 'n_estimators': 30, min_samples_split:70, max_depth:13, min_samples_leaf:10}
        # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=70,
        #                                                          min_samples_leaf=10,
        #                                                          max_depth=13,
        #                                                          max_features=7,
        #                                                          n_estimators=30,
        #                                                          random_state=10),
        #                         param_grid=param_test1, scoring='f1_macro', cv=5,
        #                         n_jobs=32)
        # gsearch1.fit(self.X_train, self.y_train)
        # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
        sep = int(len(self.y_train) * split)
        clf.fit(self.X_train[:sep], self.y_train[:sep])
        self.clf = clf
        pred_labels = clf.predict(self.X_train[sep:])
        real_labels = self.y_train[sep:]
        # print(pred_labels.shape)
        # print(real_labels.shape)
        f_score = self.get_result(list(pred_labels.tolist()), list(real_labels.tolist()))
        print(f_score)
        return f_score

    def predict(self, clf):
        clf.fit(self.X_train, self.y_train)
        return np.array(clf.predict(self.X_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = 'virus'
    parser.add_argument('--X_train_dir', type=str, default=mode+'_k_fold_model',
                        help='模型的预测输出位置，应为一个examples nums * class nums的矩阵')
    parser.add_argument('--y_train_dir', type=str, default='k_fold/'+mode,
                        help='模型的真实数据，包含句子的真正标签')
    parser.add_argument('--X_test_dir', type=str, default=mode+'_k_fold_model',
                        help='模型的test数据，生成的oof_test')
    parser.add_argument('--output_dir', type=str, default='stacking_results/'+mode,
                        help='输出的目录')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    st = Stacker(args)
    st.load_data()
    # clf = ExtraTreesClassifier()
    # clf = GradientBoostingClassifier()
    # clf = AdaBoostClassifier()
    # virus
    # clf = RandomForestClassifier(min_samples_split=70,
    #                              min_samples_leaf=10,
    #                              max_depth=13,
    #                              max_features=5,
    #                              n_estimators=50,
    #                              random_state=10)
    # usual
    # usual_param = {'max_features': 7, 'n_estimators': 30, 'min_samples_split':70, 'max_depth':13, 'min_samples_leaf':30}
    # virus best param
    best_param = {'max_depth': 13, 'max_features': 5, 'min_samples_leaf': 10, 'min_samples_split': 70, 'n_estimators': 50, 'random_state':10}
    clf = RandomForestClassifier(**best_param)
    f_score = st.stack(clf)
    preds = st.predict(clf)

    preds = preds.tolist()
    label_list = ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
    results = []
    for i, pred in enumerate(preds, start=1):
        results.append({'id': i,
                        "label": label_list[pred]})

    with open(os.path.join(args.output_dir, mode+'_result.txt'), 'w', encoding='utf8') as fw:
        json.dump(results, fw, indent=4, ensure_ascii=False)

