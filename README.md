# smp-ewect-code
这是2020年smp-ewect情感分析评测的代码，在最终排行第6，获得三等奖。
这题虽然是情感分析，但是可以简化为一个文本分类问题。
本次比赛在普通的bert for sequence classification代码进行改进，主要包括一下几点：
- 网络结构的改进，这部分的定义主要定义在net中，主要包括pooling方法改进，loss function的改进。
- 数据清洗，对原始数据中的乱码字符进行清洗，这部分主要定义在clean_data.py中。
- k折交叉训练，这部分主要在roberta_k_fold.py中。
- 多模型投票，主要采用stacking和voting两种方法。

如何运行
- 运行 run_k_fold.sh 显存不足需要调整batch size，本队训练过程中均通过软batch将size设置为64，随机种子也已经固定。
- 运行 run_k_fold_test.sh 预测模型的oof文件，train过程也会生成对应结果。
如果想要生产提交的结果，
- 运行 voting.py 会根据目录下各个模型的oof文件进行投票集成，并生成最终的结果。
