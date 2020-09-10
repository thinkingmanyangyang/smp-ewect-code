import os
import json
import logging
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

LABEL_LIST = ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
SMP_2019_LABEL_LIST = ['0', '1', '2']
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def collate_batch(features):
    # In this method we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    first = features[0]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    # 将features的label属性转换为labels, 以匹配模型的输入参数名称
    if hasattr(first, "label") and first.label is not None:
        if type(first.label) is int:
            labels = torch.tensor([f.label for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label for f in features], dtype=torch.float)
        batch = {"labels": labels}
    elif hasattr(first, "label_ids") and first.label_ids is not None:
        if type(first.label_ids[0]) is int:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        else:
            labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
        batch = {"labels": labels}
    else:
        batch = {}

    # Handling of all other possible attributes.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in vars(first).items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
    return batch



# def convert_example_to_features(example, max_seq_length, tokenizer):
#     """
#     Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
#     IDs, LM labels, input_mask, CLS and SEP tokens etc.
#     :param example: InputExample, containing sentence input as strings and is_next label
#     :param max_seq_length: int, maximum length of sequence.
#     :param tokenizer: Tokenizer
#     :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
#     """
#     tokens_a = example.tokens_a
#     tokens_b = example.tokens_b
#     # Modifies `tokens_a` and `tokens_b` in place so that the total
#     # length is less than the specified length.
#     # Account for [CLS], [SEP], [SEP] with "- 3"
#     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 1)
#
#     tokens_a, t1_label = random_word(tokens_a, tokenizer)
#     tokens_b, t2_label = random_word(tokens_b, tokenizer)
#     # concatenate lm labels and account for CLS, SEP, SEP
#     lm_label_ids = ([-1] + t1_label + [-1] + t2_label + [-1])
#
#     # The convention in BERT is:
#     # (a) For sequence pairs:
#     #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#     #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
#     # (b) For single sequences:
#     #  tokens:   [CLS] the dog is hairy . [SEP]
#     #  type_ids: 0   0   0   0  0     0 0
#     #
#     # Where "type_ids" are used to indicate whether this is the first
#     # sequence or the second sequence. The embedding vectors for `type=0` and
#     # `type=1` were learned during pre-training and are added to the wordpiece
#     # embedding vector (and position vector). This is not *strictly* necessary
#     # since the [SEP] token unambigiously separates the sequences, but it makes
#     # it easier for the model to learn the concept of sequences.
#     #
#     # For classification tasks, the first vector (corresponding to [CLS]) is
#     # used as as the "sentence vector". Note that this only makes sense because
#     # the entire model is fine-tuned.
#     tokens = []
#     segment_ids = []
#     tokens.append("[CLS]")
#     segment_ids.append(0)
#     for token in tokens_a:
#         tokens.append(token)
#         segment_ids.append(0)
#     tokens.append("[SEP]")
#     segment_ids.append(0)
#
#     assert len(tokens_b) > 0
#     for token in tokens_b:
#         tokens.append(token)
#         segment_ids.append(1)
#     tokens.append("[SEP]")
#     segment_ids.append(1)
#
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#     # The mask has 1 for real tokens and 0 for padding tokens. Only real
#     # tokens are attended to.
#     input_mask = [1] * len(input_ids)
#
#     # Zero-pad up to the sequence length.
#     while len(input_ids) < max_seq_length:
#         input_ids.append(0)
#         input_mask.append(0)
#         segment_ids.append(0)
#         lm_label_ids.append(-1)
#
#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length
#     assert len(lm_label_ids) == max_seq_length
#
#     if example.guid < 5:
#         logger.info("*** Example ***")
#         logger.info("guid: %s" % (example.guid))
#         logger.info("tokens: %s" % " ".join(
#                 [str(x) for x in tokens]))
#         logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#         logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#         logger.info(
#                 "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#         logger.info("LM label: %s " % (lm_label_ids))
#         logger.info("Is next sentence label: %s " % (example.is_next))
#
#     features = InputFeatures(input_ids=input_ids,
#                              input_mask=input_mask,
#                              segment_ids=segment_ids,
#                              lm_label_ids=lm_label_ids,
#                              is_next=example.is_next)
#     return features
