from torch.autograd import Variable
from gnn.gnn_utils import OP_PRIMITIVES
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class FloatToLongSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播：执行从 float 到 long 的转换
        # 注意：这里实际上返回的仍然是 float 类型的数据，因为 long 类型不支持梯度
        # 但是，我们模拟这一操作，实际应用中可能根据转换结果进行下一步的处理
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：直接传递梯度
        return grad_output.clone()

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播时执行非可微分操作，例如 argmax
        return input.argmax(dim=-1).long()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时直接传递梯度
        return grad_output

class Gumbel_Sample_Model(torch.nn.Module):
    def __init__(self, edge_attr):
        super(Gumbel_Sample_Model, self).__init__()
        self.log_alpha = Parameter(
            torch.zeros([edge_attr.size(0), len(OP_PRIMITIVES)]).normal_(1, 0.01),requires_grad=True
        )

        self._temp = 1
        self.indices = torch.arange(0, len(OP_PRIMITIVES)).repeat(edge_attr.size(0), 1)

        # self.normal_identity_s, self.normal_identity_e = 8, 12
        # self.reduction_identity_s, self.reduction_identity_e = 20, 24
        # self.graph_identity_s, self.graph_identity_e = 24, 29
        #
        # with torch.no_grad():
        #     self.log_alpha[self.normal_identity_s:self.normal_identity_e, :].zero_()  # 设置前5行为0
        #     self.log_alpha[self.reduction_identity_s:self.reduction_identity_e, :].zero_()  # 设置前5行为0
        #     self.log_alpha[self.graph_identity_s:self.graph_identity_e, :].zero_()  # 设置前5行为0

        self.mask = edge_attr == 0
        with torch.no_grad():
            for i, m in enumerate(self.mask):
                if m:  # 如果这一行应该被置0
                    self.log_alpha[i,:].zero_()

    def _get_gumbel_dist(self, log_alpha):
        # log_alpha 2d one_hot 2d
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        r = softmax((log_alpha + (-((-(u.log())).log()))) / self._temp)
        return r
    def get_indices(self, r):
        # r_hard = (r == r.max(1, keepdim=True)[0]).float()  #  STE 能否解决 ？
        r_hard = torch.argmax(r, dim=1)
        # r_re = (r_hard - r).detach() + r
        r_hard_one_hot = F.one_hot(r_hard,num_classes=r.size(1)).float()  # 转换为 one-hot 编码，形状为 [batch_size, num_classes]
        # 使用直通估计器（STE）的技巧
        r_re = (r_hard_one_hot - r).detach() + r
        return r_re

    def forward(self):
        gumbel_distribution = self._get_gumbel_dist(self.log_alpha)
        arch_indices = self.get_indices(gumbel_distribution)
        # re_tensor =  one_hot*self.indices
        # re = torch.sum(re_tensor, dim=1)
        # long_like_re = FloatToLongSTE.apply(re)
        # long_like_re = STEFunction.apply(one_hot)
        # return long_like_re
        # return re
        return arch_indices

    def get_cur_arch_attr(self):
        # 使用argmax获取每一行最大值的索引
        _, indices = self.log_alpha.max(dim=1)
        # 将索引转换为对应的操作
        return [OP_PRIMITIVES[index] for index in indices]

    def get_cur_attr(self):
        return torch.argmax(self.log_alpha, dim=1)

    def zero_grad_identity_edge(self) -> None:
        # 在梯度更新前手动将特定行的梯度设置为0
        with torch.no_grad():
            for i, m in enumerate(self.mask):
                if m:  # 如果这一行应该被置0
                    self.log_alpha.grad[i,:].zero_()

class Search_Model(torch.nn.Module):
    def __init__(self, surrogate_model, Gumbel_Sample_Model):
        super(Search_Model, self).__init__()
        self.surrogate_model = surrogate_model
        self.sample_model = Gumbel_Sample_Model
    def forward(self, x, edge_index, batch):
        edge_attr = self.sample_model()
        return self.surrogate_model.model(x, edge_index, edge_attr, batch)

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp