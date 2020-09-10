import torch

class DataGen(object):
    def __init__(self, model):
        self.model = model
        self.backup = {}


    def attack(self, epsilon=0.5, emd_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emd_name='word_embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def vir_adv(self, epsilon=1, xi=10, iters=1, emd_name = 'word_embedding',
                is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emd_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if is_first_attack:
                    delta1, delta2 = 0.0, torch.randn_like(param.data)
                else:
                    delta1 = delta2
                    delta2 = param.grad
                delta2 = xi * torch.norm(delta2)
                param.data.add_(-delta1 + delta2)

        # for _ in range(iters):  # 迭代求扰动
        #     delta2 = xi * l2_normalize(delta2)
        #     K.set_value(embeddings, K.eval(embeddings) - delta1 + delta2)
        #     delta1 = delta2
        #     delta2 = embedding_gradients(inputs)[0]  # Embedding梯度
        # delta2 = epsilon * l2_normalize(delta2)

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

    # def rand_attack(self, std=1e-3, emb_name='word_embedding'):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emb_name in name:
    #             self.backup[name] = param.data.clone()
    #             noise = torch.normal(mean=torch.zeros_like(param.data), std=std)
    #             noise = noise.to(param.device)
    #             noise = self._dropout(noise, 0.2)
    #             param.data.add_(noise)
    #
    # def rand_restore(self, emd_name='word_embeddings'):
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad and emd_name in name:
    #             assert name in self.backup
    #             param.data = self.backup[name]
    #     self.backup = {}
    #
    # def _dropout(self, X, drop_prob=0.1):
    #     X = X.float()
    #     assert 0 <= drop_prob < 1
    #     keep_prob = 1 - drop_prob
    #     mask = (torch.rand(X.shape[0], 1) < keep_prob).float()
    #     mask = mask.to(X.device)
    #     return mask * X
