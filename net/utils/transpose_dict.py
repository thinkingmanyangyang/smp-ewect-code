
# OLD_KEYS = [
# 'embedding.word_embedding.weight',
# 'embedding.position_embedding.weight',
# 'embedding.segment_embedding.weight',
# 'embedding.layer_norm.gamma',
# 'embedding.layer_norm.beta',
# 'encoder.transformer.0.self_attn.linear_layers.0.weight',
# 'encoder.transformer.0.self_attn.linear_layers.0.bias',
# 'encoder.transformer.0.self_attn.linear_layers.1.weight',
# 'encoder.transformer.0.self_attn.linear_layers.1.bias',
# 'encoder.transformer.0.self_attn.linear_layers.2.weight',
# 'encoder.transformer.0.self_attn.linear_layers.2.bias',
# 'encoder.transformer.0.self_attn.final_linear.weight',
# 'encoder.transformer.0.self_attn.final_linear.bias',
# 'encoder.transformer.0.layer_norm_1.gamma',
# 'encoder.transformer.0.layer_norm_1.beta',
# 'encoder.transformer.0.feed_forward.linear_1.weight',
# 'encoder.transformer.0.feed_forward.linear_1.bias',
# 'encoder.transformer.0.feed_forward.linear_2.weight',
# 'encoder.transformer.0.feed_forward.linear_2.bias',
# 'encoder.transformer.0.layer_norm_2.gamma',
# 'encoder.transformer.0.layer_norm_2.beta',
# ]
# NEW_KEYS = [
# 'bert.embeddings.word_embeddings.weight',
# 'bert.embeddings.position_embeddings.weight',
# 'bert.embeddings.token_type_embeddings.weight',
# 'bert.embeddings.LayerNorm.weight',
# 'bert.embeddings.LayerNorm.bias',
# 'bert.encoder.layer.0.attention.self.query.weight',
# 'bert.encoder.layer.0.attention.self.query.bias',
# 'bert.encoder.layer.0.attention.self.key.weight',
# 'bert.encoder.layer.0.attention.self.key.bias',
# 'bert.encoder.layer.0.attention.self.value.weight',
# 'bert.encoder.layer.0.attention.self.value.bias',
# 'bert.encoder.layer.0.attention.output.dense.weight',
# 'bert.encoder.layer.0.attention.output.dense.bias',
# 'bert.encoder.layer.0.attention.output.LayerNorm.weight',
# 'bert.encoder.layer.0.attention.output.LayerNorm.bias',
# 'bert.encoder.layer.0.intermediate.dense.weight',
# 'bert.encoder.layer.0.intermediate.dense.bias',
# 'bert.encoder.layer.0.output.dense.weight',
# 'bert.encoder.layer.0.output.dense.bias',
# 'bert.encoder.layer.0.output.LayerNorm.weight',
# 'bert.encoder.layer.0.output.LayerNorm.bias',
# ]
import torch

def transpose_name(name):
    # bert encoder part
    name = name.replace('encoder.transformer', 'bert.encoder.layer')
    name = name.replace('self_attn.linear_layers.0', 'attention.self.query')
    name = name.replace('self_attn.linear_layers.1', 'attention.self.key')
    name = name.replace('self_attn.linear_layers.2', 'attention.self.value')
    name = name.replace('self_attn.final_linear', 'attention.output.dense')
    name = name.replace('layer_norm_1.gamma', 'attention.output.LayerNorm.weight')
    name = name.replace('layer_norm_1.beta', 'attention.output.LayerNorm.bias')
    name = name.replace('feed_forward.linear_1', 'intermediate.dense')
    name = name.replace('feed_forward.linear_2', 'output.dense')
    name = name.replace('layer_norm_2.gamma', 'output.LayerNorm.weight')
    name = name.replace('layer_norm_2.beta', 'output.LayerNorm.bias')
    # bert embedding part
    name = name.replace('embedding.word_embedding.weight', 'bert.embeddings.word_embeddings.weight')
    name = name.replace('embedding.position_embedding.weight', 'bert.embeddings.position_embeddings.weight')
    name = name.replace('embedding.segment_embedding.weight', 'bert.embeddings.token_type_embeddings.weight')
    name = name.replace('embedding.layer_norm.gamma', 'bert.embeddings.LayerNorm.weight')
    name = name.replace('embedding.layer_norm.beta', 'bert.embeddings.LayerNorm.bias')

    # bert target part
    name = name.replace('target.mlm_linear_1', 'bert.pooler.dense')
    name = name.replace('target.mlm_linear_1.weight','cls.predictions.transform.dense.weight')
    name = name.replace('target.mlm_linear_1.bias', 'cls.predictions.transform.dense.bias')
    name = name.replace('target.layer_norm.gamma', 'cls.predictions.transform.LayerNorm.weight')
    name = name.replace('target.layer_norm.beta','cls.predictions.transform.LayerNorm.bias')
    name = name.replace('target.mlm_linear_2.weight', 'cls.predictions.decoder.weight')
    name = name.replace('target.mlm_linear_2.bias', 'cls.predictions.bias')
    name = name.replace('target.nsp_linear_1.weight', 'bert.pooler.dense.weight')
    name = name.replace('target.nsp_linear_1.bias', 'bert.pooler.dense.bias')
    name = name.replace('target.nsp_linear_2.weight', 'cls.seq_relationship.weight')
    name = name.replace('target.nsp_linear_2.bias', 'cls.seq_relationship.bias')
    return name










