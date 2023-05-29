# coding:utf-8
import torch
import torch.nn.functional as F
import numpy as np
import math
from model import MFMCGCNABSA
from utils import torch_utils


class ABSATrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = MFMCGCNABSA(args, emb_matrix=emb_matrix)

        self.model.cuda()


    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer = checkpoint['optimizer']
       
        print(checkpoint['config'],"checkpoint")

    # save model_state and args
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.args,
            'optimizer':self.optimizer,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.args.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def different_loss(self, Z, ZC):
        diff_loss = torch.mean(torch.matmul(Z.permute(0, 2, 1), ZC) ** 2)
        return diff_loss

    def similarity_loss(self, ZCSY, ZCSE):
        ZCSY = F.normalize(ZCSY, p=2, dim=1)
        ZCSE = F.normalize(ZCSE, p=2, dim=1)
        similar_loss = torch.mean((ZCSY - ZCSE) ** 2)
        return similar_loss


    def update(self, batch,optimizer):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:12]
        label = batch[-1]

        # step forward
        self.model.train()
        optimizer.zero_grad()
        logits, outputs, h_sy, h_se, h_csy, h_cse= self.model(inputs)
        diff_loss = self.args.beta1 * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)

        loss = F.cross_entropy(logits, label, reduction='mean') #  + 0.5 * (diff_loss + similar_loss)
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        
        # backward
        loss.backward()
        optimizer.step()

        return loss.data, acc

    def predict(self, batch):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:12]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, g_outputs,h_sy, h_se, h_csy, h_cse = self.model(inputs)

        diff_loss = self.args.beta * (self.different_loss(h_sy, h_csy) + self.different_loss(h_se, h_cse))
        similar_loss = self.args.theta * self.similarity_loss(h_csy, h_cse)

        loss = F.cross_entropy(logits, label, reduction='mean') # + diff_loss + similar_loss
        
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        
        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, g_outputs.data.cpu().numpy()

    def show_error(self, batch, vocab=None):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:12]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, g_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        # wrongs = (torch.max(logits, 1)[1].view(label.size()).data != label.data)
        # print('batch', batch)
        # print('run error', wrongs)
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        
        # print('acc', acc)
        # print('predictions', predictions)
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        
        for i in range(len(batch)):
            tokids = batch[0][i]
            aspids = batch[1][i]
            ithlabel = batch[-1][i]
            pridict = predictions[i] 
            if vocab is not None:
                # print(tokids)
                tok = [vocab.itos[idx] for idx in tokids]
                asp_tok = [vocab.itos[idx] for idx in aspids]
                # strline = ' '.join(tok) + ' '.join(asp_tok) + str(label.item()) + str(pridict.item())
                # if ithlabel.item() != pridict:
                strline = '{} {} {} {} {}'.format(' '.join(tok), ' '.join(asp_tok), ithlabel.item(), pridict, ithlabel.item() == pridict)
                print(strline)
        
        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, g_outputs.data.cpu().numpy()

