import torch.nn as nn
import torch
import itertools

# def get_outputs(out_trend, out1, out2, out3):
#     softmax = nn.Softmax(dim=1)
#     out_trend_sf = softmax(out_trend)
#     out = torch.stack((out1, out2, out3), 1)
#     out=out.view(out_trend_sf.size())
#     output = out_trend_sf * out
#     output = torch.sum(output, 1)
#     return output

def get_outputs(out_trend, out1, out2, out3):
    # softmax = nn.Softmax(dim=1)
    # out_trend_sf = softmax(out_trend)
    out_trend_nor = normalize(out_trend)
    out = torch.stack((out1, out2, out3), 1)
    out = out.view(out_trend_nor.size())
    output = out_trend_nor * out
    output = torch.sum(output, 1)
    return output

def normalize(out_trend):
    sum_out = torch.sum(out_trend,1)
    sum_out_mat = torch.stack((sum_out,sum_out,sum_out),1)
    out_trend = out_trend/sum_out_mat
    return out_trend

if __name__ == '__main__':
    combinations = list(itertools.combinations(range(10), 3))
    print(len(combinations))
    # trend = torch.tensor([[0.5,0.1,0.4],[0.6,0.4,0.8]])
    # out1 = torch.tensor([[117.0],[125.0]])
    # out2 = torch.tensor([[114.0], [155.0]])
    # out3 = torch.tensor([[157.0], [165.0]])
    # print(get_outputs(trend, out1, out2, out3))
