import numpy as np
from mindspore import nn, train
import os
from utils import linear_rampup, rda_consistency_loss, scmatch_loss
import mindspore as ms

ndim_feat = 32

class SimpleMLP(nn.Cell):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(SimpleMLP, self).__init__()
        
        self.layers = nn.SequentialCell([
            nn.Dense(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dense(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dense(hidden_sizes[1], num_classes),
            nn.Softmax()
        ])
    
    def construct(self, x):
        return self.layers(x)


class RDA:
    def __init__(self, num_classes, ema_m, device_id, train_model, eval_model, epochs, T, p_cutoff, lambda_u,
                 min_clust_K, max_clust_K, lambda_c, clust_cutoff, clust_cont_temp,lb_loader, ulb_loader, val_loader, lr,
                 rampup_ratio=0.8, num_train_iter=1):

        self.num_classes = num_classes
        self.ema_m = ema_m
        self.device_id = device_id

        self.train_model = train_model
        self.A = SimpleMLP(train_model.input_channel, 128, num_classes)
        self.eval_model = eval_model
        self.epochs = epochs
        self.iterations = num_train_iter

        self.lb_loader = lb_loader
        self.ulb_loader = ulb_loader
        self.val_loader = val_loader


        self.t_fn = T
        self.p_fn = p_cutoff
        self.lambda_u = lambda_u

        # 定义优化器和损失函数
        self.opt = nn.Momentum(params=self.train_model.trainable_params(), learning_rate=lr, momentum=0.9)

        # for param_q, param_k in zip(self.train_model.trainable_params(), self.eval_model.trainable_params()):
        #     param_k.data.copy_(param_q.detach().data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient for eval_net
        # self.eval_model.eval()

        self.min_k = min_clust_K
        self.max_k = max_clust_K

        self.func_rampup = linear_rampup(int(self.epochs * rampup_ratio))
        self.lambda_c = lambda_c
        self.clust_cutoff = clust_cutoff
        self.clust_cont_temp = clust_cont_temp

        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        # if hasattr(self.train_model, "module"):
        #     tmp_para = self.train_model.module.parameters()
        # else:
        #     tmp_para = self.train_model.parameters()

        # for param_train, param_eval in zip(tmp_para, self.eval_model.parameters()):
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            alpha = min(1 - 1 / (self.it + 1), self.ema_m)
            # alpha = self.ema_m
            param_eval.copy_(param_eval * alpha + param_train.detach() * (1 - alpha))

        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def forward_fn(self, inputs, targets, num_lb, num_ulb, distri, distri_):
        logits, feature = self.train_model(inputs, feature=True)
        logits_A = self.A(feature)

        logits_x_lb = logits[:num_lb]
        logits_x_ulb = logits[num_lb:]
        logits_x_ulb_w = logits_x_ulb[:num_ulb]
        logits_x_ulb_s = logits_x_ulb[num_ulb:]

        logits_x_lb_A = logits_A[:num_lb]
        logits_x_ulb_A = logits_A[num_lb:]
        logits_x_ulb_w_A = logits_x_ulb_A[:num_ulb]
        logits_x_ulb_s_A = logits_x_ulb_A[num_ulb:]

        feature_x_ulb = feature[num_lb:num_ulb+num_lb]

        distri_tmp = ms.ops.reduce_mean(ms.ops.softmax(logits_x_ulb_w))
        distri_A_tmp = ms.ops.reduce_mean(ms.ops.softmax(logits_x_ulb_w_A))
        del feature

        # loss = self.loss_fn(logits_x_lb, targets) + consistency_loss(logits_x_ulb_w, logits_x_ulb_s) + \
        #        scmatch_loss(logits_x_ulb_w, logits_x_ulb_s, feature_x_ulb, n_clusts=3)
        loss = self.loss_fn(logits_x_lb, targets) + rda_consistency_loss(logits_x_ulb_w, logits_x_ulb_s, logits_x_ulb_w_A, logits_x_ulb_s_A, distri, distri_)

        return loss, distri_tmp, distri_A_tmp

    def evaluate(self):
        acc = 0
        nums = 0
        for i,(inputs, targets) in enumerate(self.val_loader):
            logtis = self.train_model(inputs)
            logtis = ms.ops.softmax(logtis)
            predicts, _ = ms.ops.max(logtis, axis=-1)
            acc = acc + (predicts==targets).sum()
            nums = nums + len(targets)
            print(acc, nums, float(acc)/nums)

        return  float(acc)/nums

    def train(self):
        # self.train_model.train()
        grad_fn = ms.value_and_grad(self.forward_fn, None, self.opt.parameters)

        train_iteration = 0

        # for it in range(1):
        while True:
            if train_iteration > self.iterations:
                break

            for (i, (lb_w, lb_s, lb_targets)), (i, (ulb_w, ulb_s, ulb_targets)) in zip(enumerate(self.lb_loader), enumerate(self.ulb_loader)):

                num_lb = lb_w.shape[0]
                num_ulb = ulb_w.shape[0]
                op = ms.ops.Concat()

                inputs = op((lb_w,ulb_w,ulb_s))


                loss, grads, distri_tmp, distri_A_tmp = grad_fn(inputs, lb_targets, num_lb, num_ulb)
                self.opt(grads)

                train_iteration += 1

                if train_iteration % 10 == 0:

                    acc = self.evaluate()
                    print('training iteration: {} / {}, acc: {}'.format(train_iteration, self.iterations, acc))
                # self._eval_model_update()
            # acc = self.train_model.eval(self.val_dst)['Accuracy']
            # print(acc)
