import torch
import torch.nn as nn
import torch.nn.functional as F


# KL Divergence calculator. alpha shape(batch_size, num_classes)
def KL(alpha):
    ones = torch.ones([1, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl.reshape(-1)


def loss_log(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.log(alpha.sum(dim=-1, keepdim=True)) - torch.log(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_digamma(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.digamma(alpha.sum(dim=-1, keepdim=True)) - torch.digamma(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_mse(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    err = (y - alpha / sum_alpha) ** 2
    var = alpha * (sum_alpha - alpha) / (sum_alpha ** 2 * (sum_alpha + 1))
    loss = torch.sum(err + var, dim=-1)
    loss = loss + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class InferNet(nn.Module):
    def __init__(self, layers_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential()
        for i in range(len(layers_dim) - 1):
            self.fc.add_module(f'infer{i}', nn.Linear(layers_dim[i], layers_dim[i + 1]))
            self.fc.add_module(f'dropout{i}', nn.Dropout(dropout))
            self.fc.add_module(f'relu{i}', nn.ReLU())

    def forward(self, x):
        return self.fc(x)


class EML(nn.Module):
    def __init__(self, sample_shapes: list, num_classes: int, delta=0.1):
        super().__init__()
        self.num_views = len(sample_shapes)
        self.num_classes = num_classes
        self.delta = delta
        # cnn for eeg
        self.c = nn.ModuleList([
            nn.Sequential(
                Reshape([-1, 1, 23, 256]),
                nn.Conv2d(1, 1, (1, 128)),
                nn.BatchNorm2d(1),
                nn.Tanh(),
                nn.Conv2d(1, 30, (1, 65)),
                nn.BatchNorm2d(30),
                nn.Tanh(),
                nn.Conv2d(30, 20, (4, 33)),
                nn.BatchNorm2d(20),
                nn.Tanh(),
                nn.Conv2d(20, 10, (8, 18)),
                nn.BatchNorm2d(10),
                nn.Tanh(),
                Reshape([-1, 13 * 16 * 10]),
                nn.Linear(13 * 16 * 10, 1024),
                nn.Tanh()
            ),
            nn.Sequential(
                Reshape([-1, 1, 23, 27]),
                nn.Conv2d(1, 20, (4, 4)),
                nn.BatchNorm2d(20),
                nn.Tanh(),
                nn.Conv2d(20, 10, (8, 8)),
                nn.BatchNorm2d(10),
                nn.Tanh(),
                Reshape([-1, 13 * 17 * 10]),
                nn.Linear(13 * 17 * 10, 1024),
                nn.Tanh()
            ),
            nn.Sequential(
                Reshape([-1, 1, 23, 14, 256]),
                nn.Conv3d(1, 1, (1, 1, 129)),
                nn.BatchNorm3d(1),
                nn.Tanh(),
                nn.Conv3d(1, 30, (4, 4, 65)),
                nn.BatchNorm3d(30),
                nn.Tanh(),
                nn.Conv3d(30, 20, (4, 4, 33)),
                nn.BatchNorm3d(20),
                nn.Tanh(),
                nn.Conv3d(20, 10, (8, 1, 17)),
                nn.BatchNorm3d(10),
                nn.Tanh(),
                Reshape([-1, 10 * 10 * 8 * 16]),
                nn.Linear(10 * 10 * 8 * 16, 2048),
                nn.Tanh(),
                nn.Linear(2048, 1024),
                nn.Tanh()
            ),
        ])  # 3 views with outputing 1024 features.
        self.f = nn.ModuleList([InferNet([1024, 512, 128]) for i in range(self.num_views)])
        self.g = nn.ModuleList([InferNet([128, 64, num_classes]) for i in range(self.num_views)])

    def forward(self, x: dict, target=None, kl_penalty=0):
        view_x = dict()
        for v in x.keys():
            view_x[v] = self.c[v](x[v])  # CNN

        view_h = dict()
        for v in view_x.keys():
            view_h[v] = self.f[v](view_x[v])

        view_e = dict()
        for v in view_h.keys():
            view_e[v] = self.g[v](view_h[v])

        fusion_e = torch.zeros_like(view_e[0])
        for v in view_e.keys():
            fusion_e = (fusion_e + view_e[v]) / 2

        loss = None
        if target is not None:
            # Loss of classifying
            loss_fusion = loss_digamma(fusion_e + 1, target, kl_penalty)

            # Loss of classifying for each view
            loss_views = 0
            for v in view_e.keys():
                loss_views += loss_digamma(view_e[v] + 1, target, kl_penalty)

            # Loss of graph
            # Firstly, We calculate affinity matrix W.
            y_row = target.unsqueeze(-1).expand(-1, len(target))
            y_col = target.unsqueeze(0).expand(len(target), -1)
            num_each_cate = torch.zeros(self.num_classes, device=target.device)  # Quantity for each category
            target_with_num = torch.zeros_like(target)
            for i in range(len(target)):
                num_each_cate[target[i]] += 1
            for i in range(len(target)):
                target_with_num[i] = num_each_cate[target[i]]  # The i-th sample's number of same categories.
            W = torch.where(y_row == y_col, 1 / target_with_num.unsqueeze(0).expand(len(target), -1) - 1 / len(target), 0)
            # secondly, We calculate loss of graph
            loss_G = torch.zeros_like(W)
            for v in view_h.keys():
                D2 = self.calculate_vector_distance(view_h[v], return_square=True)
                loss_G += W * D2  # n*n, each element is Wij*||hi-hj||^2
            loss_G /= len(view_h.keys())  # mean value

            # Total loss
            loss = loss_fusion.mean() + loss_views.mean() + self.delta * loss_G.mean()
        return view_e, fusion_e, loss

    def get_dc_loss(evidences, device):
        num_views = len(evidences)
        batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
        p = torch.zeros((num_views, batch_size, num_classes)).to(device)
        u = torch.zeros((num_views, batch_size)).to(device)
        for v in range(num_views):
            alpha = evidences[v] + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            p[v] = alpha / S
            u[v] = torch.squeeze(num_classes / S)
        dc_sum = 0
        for i in range(num_views):
            pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
            cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
            dc = pd * cc
            dc_sum = dc_sum + torch.sum(dc, dim=0)
        dc_sum = torch.mean(dc_sum)
        return dc_sum

    def calculate_vector_distance(self, X, return_square=False):
        ''''''
        Calculate distance between each row vector and each other.
        return a matrix D with shape n*n where Dij means distance between i and j.
        Refer to <https://blog.csdn.net/LoveCarpenter/article/details/85048291/#t6> 2.4
        Attention! DO NOT use sqrt if results need to backward. Because sqrt is not differentiable at x=0.
        ''''''
        G = X @ X.T  # n*n, gram matrix
        H = torch.diag(G).unsqueeze(0).expand(len(X), -1)  # n*n
        D2 = H + H.T - 2 * G  # n*n
        if return_square:
            return D2
        return torch.sqrt(D2)
