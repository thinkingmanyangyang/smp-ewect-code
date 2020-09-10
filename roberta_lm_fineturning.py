# 一个只进行mask语言模型，没有nsp任务的 fine-turning代码
import logging
import os
import random
import json
from torch.utils.data import Dataset
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
from net.utils.fgm import FGM
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import argparse
from utils import set_logger, collate_batch

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, id, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        self.id = id
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # nextSentence

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, masked_lm_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.masked_lm_labels = masked_lm_labels

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logging.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)
    return tokens, output_label

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_convert_example_to_feature(example,
                                        tokenizer,
                                        max_seq_length = 256,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        mask_padding_with_zero=True
                                        ):
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    if tokens_b == None:
        tokens_a = tokenizer.tokenize(tokens_a)
        tokens_a = tokens_a[: max_seq_length-2]
        tokens_a, t1_label = random_word(tokens_a, tokenizer)
        lm_label_ids = ([-100] + t1_label + [-100])
    else:
        tokens_a = tokenizer.tokenize(tokens_a)
        tokens_b = tokenizer.tokenize(tokens_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 1)
        tokens_a, t1_label = random_word(tokens_a, tokenizer)
        tokens_b, t2_label = random_word(tokens_b, tokenizer)
        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = ([-100] + t1_label + [-100] + t2_label + [-100])
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0 if mask_padding_with_zero else 1)
        segment_ids.append(pad_token_segment_id)
        lm_label_ids.append(-100)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    # if example.id < 3:
    #     logging.info("*** Example ***")
    #     logging.info("id: %s" % (example.id))
    #     logging.info("tokens: %s" % " ".join(
    #         [str(x) for x in tokens]))
    #     logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logging.info(
    #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logging.info("LM label: %s " % (lm_label_ids))
    #     logging.info("Is next sentence label: %s " % (example.is_next))
    features = InputFeatures(input_ids=input_ids,
                             attention_mask=input_mask,
                             token_type_ids=segment_ids,
                             masked_lm_labels=lm_label_ids,
                             )
    return features

class SimpleBERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_length, encoding="utf-8"):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.all_corpus = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                self.all_corpus.append(line.strip())
        self.corpus_lines = len(self.all_corpus)

        examples = []
        for corpus in tqdm(self.all_corpus, desc='convert to features'):
            example = InputExample(id = corpus['id'], tokens_a=corpus['content'])
            examples.append(example)
        self.examples = examples
        logging.info('loading examples end... ...')

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines

    def __getitem__(self, item):
        example = self.examples[item]
        feature = simple_convert_example_to_feature(example, self.tokenizer, max_seq_length=self.max_seq_length)
        return feature

def train_mlm(args,train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_batch)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight','transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_rate * t_total,
                                                num_training_steps=t_total)

    logging.info("***** Running training *****")
    for k,v in args.__dict__.items():
        logging.info("  {:18s} = {}".format(str(k), str(v)))
    logging.info('*'*30)
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    best_f_score = 0.
    best_epoch = 0.

    if args.attack == 'fgm':
        fgm = FGM(model)
        logging.info('*** attack method = fgm ***')

    for epoch in range(args.num_train_epochs):
        logging.info('  Epoch [{}/{}]'.format(epoch + 1, args.num_train_epochs))
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(args.device)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            logging_loss += loss.item()
            tr_loss += loss.item()

            if args.attack == 'fgm':
                # fgm 攻击
                fgm.attack()
                loss_adv, _other_val = model(**inputs)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数
                # fgm 攻击 end

        # 过gradient_accumulation_steps后才将梯度清零，不是每次更新/每过一个batch清空一次梯度，即每gradient_accumulation_steps次更新清空一次
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
                optimizer.step()
                scheduler.step() #更新学习率
                model.zero_grad()
                global_step += 1
                # 每隔一定步数评估一下
                if global_step % 50 == 0 :
                    logging.info("EPOCH = [%d/%d] global_step = %d loss = %f", epoch + 1, args.num_train_epochs,
                                 global_step, logging_loss)
                    logging_loss = 0.0
                    # 这里直接保存，没有eval过程
                    # torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
                    model.save_pretrained(args.output_dir)
    logging.info("train end ... ... ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="roberta_wwm_pretrain.log", type=str, required=True,
                        help="设置日志的输出目录")
    parser.add_argument("--data_dir", default='', type=str, required=True,
                        help="预训练语言模型的数据，应该包括全部的训练数据")
    parser.add_argument("--pre_train_path", default='roberta_model', type=str, required=True,
                        help="预训练模型所在的路径，包括 pytorch_model.bin, vocab.txt, bert_config.json")
    parser.add_argument("--output_dir", default='sentiment_model/usual2', type=str, required=True,
                        help="输出结果的文件")
    # Other parameters
    parser.add_argument("--max_seq_length", default=140, type=int,
                        help="输入到bert的最大长度，通常不应该超过512")
    # 这里改成store_false 方便直接运行
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="是否进行训练")
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=9, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
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
                        help="是否进行对抗样本训练, 选择攻击方式或者不攻击")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert os.path.exists(args.data_dir)
    assert os.path.exists(args.pre_train_path)
    assert os.path.exists(args.output_dir)
    # 暂时不写多GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置随机种子
    set_seed(args.seed)

    log_dir = os.path.join(args.output_dir, args.log_dir)
    set_logger(log_dir)

    tokenizer = BertTokenizer.from_pretrained(args.pre_train_path)
    pretrain_dataset = SimpleBERTDataset(args.data_dir,
                                         tokenizer,
                                         args.max_seq_length,)

    # 加载model
    config = BertConfig.from_pretrained(args.pre_train_path)

    model = BertForMaskedLM.from_pretrained(args.pre_train_path,
                                                    config=config)
    model = model.to(args.device)
    model.config.save_pretrained(args.output_dir)
    tokenizer.save_vocabulary(args.output_dir)

    logging.info("start training... ...")
    train_mlm(args, pretrain_dataset, model)

if __name__  == '__main__':
    main()