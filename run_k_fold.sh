#!/bin/bash
# usual
python roberta_k_fold.py \
	--pre_train_path roberta_large \
	--model_name BertBase \
	--output_dir usual_k_fold_model/roberta_large \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_large.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path roberta_base \
	--model_name BertBase \
	--output_dir usual_k_fold_model/roberta_base \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_base.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path roberta_wwm_ext_large \
	--model_name BertBase \
	--output_dir usual_k_fold_model/roberta_wwm_ext_large \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_wwm_ext_large.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path roberta_large \
	--model_name BertLSTMAttention \
	--output_dir usual_k_fold_model/roberta_large_lstm_attention \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_large_lstm_attention.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path roberta_large \
	--model_name BertAttention \
	--output_dir usual_k_fold_model/roberta_large_attention \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_large_attention.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path uer_large \
	--model_name BertUER \
	--output_dir usual_k_fold_model/uer_large \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir uer_large.log \
	--k_fold 5 \
	--do_train

python roberta_k_fold.py \
	--pre_train_path virus_k_fold_model/roberta_wwm_ext \
	--model_name BertTransferLearning \
	--output_dir usual_k_fold_model/roberta_wwm_ext_transfer_learning1 \
	--data_dir k_fold/usual \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_wwm_ext_transfer_learning1.log \
	--k_fold 5 \
	--do_train


	# virus
	python roberta_k_fold.py \
	--pre_train_path roberta_large \
	--model_name BertBase \
	--output_dir virus_k_fold_model/roberta_large \
	--data_dir k_fold/virus \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_large.log \
	--k_fold 5 \
	--do_train

	python roberta_k_fold.py \
	--pre_train_path usual_k_fold_model/roberta_wwm_ext \
	--model_name BertTransferLearning \
	--output_dir virus_k_fold_model/roberta_wwm_ext_transfer_learning \
	--data_dir k_fold/virus \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_wwm_ext_transfer_learning.log \
	--k_fold 5 \
	--do_train

	python roberta_k_fold.py \
	--pre_train_path usual_k_fold_model/roberta_wwm_ext \
	--model_name BertTransferLearning \
	--output_dir virus_k_fold_model/roberta_wwm_ext_transfer_learning_pseudo \
	--data_dir k_fold/virus \
	--max_seq_length 140 \
	--train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 20 \
	--learning_rate 1e-5 \
	--warmup_rate 0.3 \
	--weight_decay 0.02 \
	--attack fgm \
	--log_dir roberta_wwm_ext_transfer_learning_pseudo.log \
	--k_fold 5 \
	--use_pseudo_data pseudo \
	--do_train