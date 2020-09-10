import logging
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import random
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
from utils import set_logger, set_seed
from transformers import BertConfig, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel
from net.bert_base import BertForSequenceClassification
from net.bert_for_sequence_classification_labelsmooth import BertForSequenceClassificationLabelSmooth
from net.bert_focal_loss import BertForSequenceClassificationFocalLoss
from net.bert_bnm_loss import BertForSequenceClassificationBNMLoss
from net.bert_for_de import BertForSequenceClassificationDE
from net.bert_attention import BertForSequenceClassificationAttention
from net.bert_mean_max import BertForSequenceClassificationMeanMax
from net.bert_lstm import BertForSequenceClassificationLSTM
from net.bert_lstm_attention import BertForSequenceClassificationLSTMAttenion
from net.electra_model_test import ElectraForSequenceClassification
from net.bert_transfer_learning import BertTransferLearning
from net.bert_lstm_attention import BertForSequenceClassificationLSTMAttenionTransferLearning
from net.bert_mean_max import BertForSequenceClassificationMeanMaxTransferLearning
from net.bert_attention import BertForSequenceClassificationAttentionTransferLearning
from net.bert_uer import BertUER
from transformers import ElectraConfig, ElectraTokenizer
from transformers import XLNetConfig, XLNetTokenizer
from net.xlnet_base import XLNetForSequenceClassification
from train_and_eval import trains, test, predict, _predict
from sklearn.model_selection import StratifiedKFold
import argparse
import json
import pickle
from transformers import glue_convert_examples_to_features, GlueDataset, trainer, BertForMaskedLM
logger = logging.getLogger(__name__)

MODELS = {
    'BertBase': BertForSequenceClassification,
    'DebertaBase':AutoModel,
    'BertLabelSmooth': BertForSequenceClassificationLabelSmooth,
    'BertFocalLoss': BertForSequenceClassificationFocalLoss,
    'BertBNMLoss': BertForSequenceClassificationBNMLoss,
    'XlnetBase': XLNetForSequenceClassification,
    'BertForDE': BertForSequenceClassificationDE,
    'ElectraBase': ElectraForSequenceClassification,
    'BertAttention': BertForSequenceClassificationAttention,
    'BertMeanMax': BertForSequenceClassificationMeanMax,
    'BertLSTM': BertForSequenceClassificationLSTM,
    'BertLSTMAttention': BertForSequenceClassificationLSTMAttenion,
    'BertTransferLearning': BertTransferLearning,
    'BertLSTMAttentionTransferLearning': BertForSequenceClassificationLSTMAttenionTransferLearning,
    'BertMeanMaxTransferLearning': BertForSequenceClassificationMeanMaxTransferLearning,
    'BertAttentionTransferLearning': BertForSequenceClassificationAttentionTransferLearning,
    'BertUER':BertUER,
    }
TOKENIZERS = {
    'BertBase': BertTokenizer,
    'DebertaBase': AutoTokenizer,
    'BertLabelSmooth': BertTokenizer,
    'BertFocalLoss': BertTokenizer,
    'BertBNMLoss': BertTokenizer,
    'XlnetBase': XLNetTokenizer,
    'BertForDE': BertTokenizer,
    'ElectraBase': ElectraTokenizer,
    'BertAttention': BertTokenizer,
    'BertMeanMax': BertTokenizer,
    'BertLSTM': BertTokenizer,
    'BertLSTMAttention': BertTokenizer,
    'BertTransferLearning': BertTokenizer,
    'BertLSTMAttentionTransferLearning': BertTokenizer,
    'BertMeanMaxTransferLearning': BertTokenizer,
    'BertAttentionTransferLearning': BertTokenizer,
    'BertUER':BertTokenizer,
    }
CONFIGS = {
    'BertBase': BertConfig,
    'DebertaBase': AutoConfig,
    'BertLabelSmooth': BertConfig,
    'BertFocalLoss': BertConfig,
    'BertBNMLoss': BertConfig,
    'XlnetBase': XLNetConfig,
    'BertForDE': BertConfig,
    'ElectraBase': ElectraConfig,
    'BertAttention': BertConfig,
    'BertMeanMax': BertConfig,
    'BertLSTM': BertConfig,
    'BertLSTMAttention': BertConfig,
    'BertTransferLearning': BertConfig,
    'BertLSTMAttentionTransferLearning': BertConfig,
    'BertMeanMaxTransferLearning': BertConfig,
    'BertAttentionTransferLearning': BertConfig,
    'BertUER':BertConfig,
    }
class SentimentInputExample(object):
    def __init__(self, id, content, label=None):
        self.id = id
        self.content = content
        self.label = label

class SentimentInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

class SentimentDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        return self.features[item]
    def get_labels(self):
        return ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']

class KFoldProcessor:
    def __init__(self, args, data_dir):
        self.args = args
        self.data_dir = data_dir
    def get_train_examples(self):
        logger.info("*" * 10 + 'train dataset' + "*" * 10)
        examples = self._create_examples(
            self._read_file(os.path.join(self.data_dir, 'train.txt')))
        return examples
    def get_predict_examples(self):
        logger.info("*" * 10 + 'predict dataset' + "*" * 10)
        examples = self._create_examples(
            self._read_file(os.path.join(self.data_dir, 'eval.txt')),
            do_predict=True)
        return examples
    def get_pseudo_data(self):
        logger.info("*"*10 + 'use pseudo' + "*"*10)
        examples = self._create_examples(
            self._read_file(os.path.join(self.data_dir, 'pseudo_train.txt')))
        return examples
    def get_labels(self):
        return ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, encoding='utf-8') as f:
            data_list = json.load(f)
        return data_list
    @classmethod
    def _create_examples(cls, data_list, do_predict=False):
        examples = []
        for data in data_list:
            id = data['id']
            content = data['content']
            if do_predict:
                label = 'neural'
            else:
                label = data['label']
            examples.append(
                SentimentInputExample(
                    id=id,
                    content=content,
                    label=label
                ))
        return examples

def sentiment_convert_examples_to_features(examples,
                                           tokenizer,
                                           max_length=256,
                                           label_list=None,
                                           pad_token=0,
                                           pad_token_segment_id = 0,
                                           mask_padding_with_zero = True):
    logging.info('***** converting to features *****')
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    def _truncate(content, max_length):
        while len(content) > max_length:
            content = list(content)
            content.pop(len(content)//2)
        return ''.join(content)
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            text=example.content,
            text_pair= None,
            max_length=max_length,
            truncation=True,
            # truncate_first_sequence=True  # We're truncating the first sequence in priority if True
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),max_length)

        label = int(label_map[example.label])

        features.append(
            SentimentInputFeatures(input_ids, attention_mask, token_type_ids, label)
        )
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="roberta_wwm_sentiment.log", type=str, required=True,
                        help="设置日志的输出目录")
    parser.add_argument("--data_dir", default='data/train/usual', type=str, required=True,
                        help="数据文件目录，应当有train.text dev.text")
    parser.add_argument("--pre_train_path", default='roberta_model', type=str, required=True,
                        help="预训练模型所在的路径，包括 pytorch_model.bin, vocab.txt, bert_config.json")
    parser.add_argument("--model_name", default='RobertaBase', )
    parser.add_argument("--output_dir", default='sentiment_model/usual2', type=str, required=True,
                        help="输出结果的文件")

    # Other parameters
    parser.add_argument("--max_seq_length", default=140, type=int,
                        help="输入到bert的最大长度，通常不应该超过512")
    # 这里改成store_false 方便直接运行
    parser.add_argument("--do_train", action='store_true',
                        help="是否进行训练")
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--train_batch_size", default=9, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=512, type=int,
                        help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=3.0, type=float,
                        help="最大的梯度更新")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="epoch 数目")
    parser.add_argument('--seed', type=int, default=233,
                        help="random seed for initialization")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="让学习增加到1的步数，在warmup_steps后，再衰减到0")
    parser.add_argument("--warmup_rate", default=0.00, type=float,
                        help="让学习增加到1的步数，在warmup_steps后，再衰减到0，这里设置一个小数，在总训练步数*rate步时开始增加到1")
    parser.add_argument("--attack", default=None,
                        help = "是否进行对抗样本训练, 选择攻击方式或者不攻击")
    parser.add_argument("--label_smooth", default=0.0, type=float,
                        help="设置标签平滑参数")
    parser.add_argument("--use_pseudo_data", action='store_true',
                        help = '是否使用生成的伪标签数据集')
    parser.add_argument("--k_fold", default=5, type=int,
                        help='k折交叉验证的划分数目')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.pre_train_path)
    assert os.path.exists(args.output_dir)

    # 暂时不写多GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # 这里设置日志目录在sentiment目录下，方便调参时把多个参数的log输入到一个文件里面
    # output_parent_path, _ = os.path.split(args.output_dir)
    log_dir = os.path.join(args.output_dir, args.log_dir)
    set_logger(log_dir)

    k_fold_processor = KFoldProcessor(args, args.data_dir)
    logging.info('loading model... ...')

    if args.do_train:
        tokenizer = TOKENIZERS[args.model_name].from_pretrained(args.pre_train_path)
        # 加载model
        config = CONFIGS[args.model_name].from_pretrained(args.pre_train_path,
                                                          num_labels=len(k_fold_processor.get_labels()))
        # model = MODELS[args.model_name].from_pretrained(args.pre_train_path,
        #                                             config=config, args=args)
        config.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
    else:
        tokenizer = TOKENIZERS[args.model_name].from_pretrained(args.output_dir)
        config = CONFIGS[args.model_name].from_pretrained(args.output_dir,
                                                          num_labels=len(k_fold_processor.get_labels()))
    # model = model.to(args.device)
    logging.info('load model end... ...')
    def convert_to_dataset(examples):
        features = sentiment_convert_examples_to_features(examples=examples,
                                                      tokenizer = tokenizer,
                                                      max_length = args.max_seq_length,
                                                      label_list= k_fold_processor.get_labels())
        return SentimentDataset(features)
    logging.info('dataset loaded...')
    predict_examples = k_fold_processor.get_predict_examples()
    predict_dataset = convert_to_dataset(predict_examples)
    if args.use_pseudo_data:
        pseudo_examples = k_fold_processor.get_pseudo_data()
        pseudo_examples = np.array(pseudo_examples)
    if args.k_fold:
        all_train_examples = k_fold_processor.get_train_examples()
        all_train_examples = np.array(all_train_examples)
        logging.info("start training... ...")
        oof_train = np.zeros((len(all_train_examples), len(k_fold_processor.get_labels())))
        oof_test = np.zeros((len(predict_examples), len(k_fold_processor.get_labels())))
        all_train_labels = [example.label for example in all_train_examples]
        stratified_folder = StratifiedKFold(n_splits=args.k_fold, random_state=args.seed, shuffle=False)
        for fold_num, (train_idx, dev_idx) in enumerate(stratified_folder.split(all_train_examples, all_train_labels), start=1):
            logging.info('start fold ' + str(fold_num) + ' training...')
            train_examples, dev_examples = all_train_examples[train_idx], all_train_examples[dev_idx]
            if args.use_pseudo_data:
                train_examples = np.append(train_examples, pseudo_examples, axis=0)
            train_dataset = convert_to_dataset(train_examples)
            dev_dataset = convert_to_dataset(dev_examples)
            if args.do_train:
                if 'TransferLearning' in args.model_name or 'UER' in args.model_name:
                    config.ignore_weights = ['classifier', 'pooler']
                    if os.path.exists(os.path.join(args.pre_train_path, 'pytorch_model.bin')):
                        model = MODELS[args.model_name].from_pretrained(
                                args.pre_train_path,
                                config=config, args=args)
                    else:
                        model = MODELS[args.model_name].from_pretrained(
                                os.path.join(args.pre_train_path, '4'),
                                config=config, args=args)
                    config.ignore_weights = None
                else:
                    model = MODELS[args.model_name].from_pretrained(args.pre_train_path,
                                                                config=config, args=args)

                model = model.to(args.device)
                trains(args, train_dataset, dev_dataset, model, str(fold_num))
            model = MODELS[args.model_name].from_pretrained(os.path.join(args.output_dir, str(fold_num)),
                                                            config=config, args=args)
            model = model.to(args.device)
            train_probs = test(args, model, dev_dataset)
            oof_train[dev_idx] = train_probs

            predict_probs = _predict(args, model, predict_dataset)
            oof_test += predict_probs
        oof_test = oof_test/args.k_fold
    # 用于stacking
    with open(os.path.join(args.output_dir, 'oof_train'), 'wb') as wf:
        pickle.dump(oof_train, wf)
    # 用于stacking 或者直接根据均值输出结果
    with open(os.path.join(args.output_dir, 'oof_test'), 'wb') as wf:
        pickle.dump(oof_test, wf)
    logging.info('processing end... ...')
if __name__ == '__main__':
    main()

