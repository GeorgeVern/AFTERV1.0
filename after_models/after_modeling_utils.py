import torch


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        # return x or x.clone()
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # lambda_ = ctx.lambd
        # lambda_ = grad_output.new_tensor(lambda_)
        # dx = -lambda_ * grad_output
        # print(ctx.lambd)
        return grad_output * -ctx.lambd, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambd=1):
        super(GradientReversal, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

    def extra_repr(self):
        return 'lambda={}'.format(self.lambd)