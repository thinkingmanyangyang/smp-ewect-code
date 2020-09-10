import pickle
import argparse
import numpy as np
import json
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='转换之前的输入，应为一个probs矩阵')
parser.add_argument('--output_dir', type=str, required=True, help='转换后的输出，符合提交系统的格式')
parser.add_argument('--weight_dir', type=str, default=None, help='各个probs的权重')
args = parser.parse_args()
with open(args.input_dir, 'rb') as f:
    predict_probs = pickle.load(f)
predict_probs = np.array(predict_probs)
if args.weight_dir:
    with open(args.weight_dir, 'rb') as f:
        predict_weight = pickle.load(f)
        predict_weight = np.array(predict_weight)
        predict_probs = predict_weight * predict_probs

predict_label_id = np.argmax(predict_probs, axis=1)
label_list = ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
results = []
for i, pred in enumerate(predict_label_id, start=1):
    results.append({'id': i,
                    "label": label_list[pred]})

with open(args.output_dir, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)