import torch
import torch.nn.functional as F
import yacs.config
import math
import torch.nn as nn


class KL_Loss(nn.Module):
    def __init__(self, temperature=4):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        # loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        loss = self.T * self.T * torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch))/teacher_outputs.size(0)
        return loss


class KL_Loss2(nn.Module):
    def __init__(self, temperature=4):
        super(KL_Loss2, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        # teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        teacher_outputs = teacher_outputs + 10 ** (-7)
        # loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        loss = self.T * self.T * torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch))/teacher_outputs.size(0)
        return loss