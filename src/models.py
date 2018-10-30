import torch
from torch import nn, LongTensor
from torch.autograd import Variable
from torch.nn import functional

USE_GPU = torch.cuda.is_available()

class MF(nn.Module):
    def __init__(self, n_job, n_geek, layers_dim):
        super(MF, self).__init__()
        self.job_emb = nn.Embedding(n_job, int(layers_dim))
        self.geek_emb = nn.Embedding(n_geek, int(layers_dim))
        self.criterion = nn.MSELoss()
        self.shape = (n_job, n_geek)

    def forward(self, job, geek):
        job = self.job_emb(job)
        geek = self.geek_emb(geek)
        x = functional.cosine_similarity(job, geek)
        return x

    def predict(self, test_data):
        n_job, n_geek = self.shape
        predictions = []
        # with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            if job >= n_job:
                continue
            geeks = [x for x in sample[1:] if x < n_geek]
            job_tensor = Variable(LongTensor([job] * len(geeks)))
            geeks_tensor = Variable(LongTensor(geeks))
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            scores = self.forward(job_tensor, geeks_tensor)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample):
        job, geek, label = sample.t()
        # job, geek = feature
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        # label = label.view(label.shape[0], 1)
        label = label.float()
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek)
        loss = model.criterion(out, label)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * label.size(0)


class GMF(nn.Module):
    def __init__(self, n_job, n_geek, layers_dim):
        super(GMF, self).__init__()
        self.job_emb = nn.Embedding(n_job, int(layers_dim))
        self.geek_emb = nn.Embedding(n_geek, int(layers_dim))
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()
        self.shape = (n_job, n_geek)
        self.out = nn.Sequential(
            nn.Linear(layers_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, job, geek):
        job = self.job_emb(job)
        geek = self.geek_emb(geek)
        x = self.out(job * geek)
        return x

    def predict(self, test_data):
        n_job, n_geek = self.shape
        predictions = []
        # with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            if job >= n_job:
                continue
            geeks = [x for x in sample[1:] if x < n_geek]
            job_tensor = Variable(LongTensor([job] * len(geeks)))
            geeks_tensor = Variable(LongTensor(geeks))
            # job_tensor = job_tensor.view(job_tensor.shape[0], 1)
            # geeks_tensor = geeks_tensor.view(geeks_tensor.shape[0], 1)
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            scores = self.forward(job_tensor, geeks_tensor)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample):
        # job = sample[:, 0].unsqueeze(dim=1)
        # geek = sample[:, 1].unsqueeze(dim=1)
        # label = sample[:, 2].unsqueeze(dim=1)
        job, geek, label = sample.t()
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        label = label.view(label.shape[0], 1)
        label = label.float()
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek)
        loss = model.criterion(out, label)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * label.size(0)


class MLP(nn.Module):
    def __init__(self, n_job, n_geek, emb_dim):
        super(MLP, self).__init__()
        self.shape = (n_job, n_geek)
        self.job_emb = nn.Embedding(n_job, emb_dim)
        self.geek_emb = nn.Embedding(n_geek, emb_dim)
        dim = emb_dim * 2
        self.hidens = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward(self, *input):
        job, geek = input
        job = self.job_emb(job)
        geek = self.geek_emb(geek)
        x = torch.cat((job, geek), dim=-1).squeeze(dim=1)
        x = self.hidens(x)
        return x

    def predict(self, test_data):
        n_job, n_geek = self.shape
        predictions = []
        # with torch.no_grad():
        for sample in test_data:
            job = sample[0]
            if job >= n_job:
                continue
            geeks = [x for x in sample[1:] if x < n_geek]
            job_tensor = Variable(LongTensor([job] * len(geeks)))
            geeks_tensor = Variable(LongTensor(geeks))
            # job_tensor = job_tensor.view(job_tensor.shape, 1)
            # geeks_tensor = geeks_tensor.view(geeks_tensor.shape, 1)
            if USE_GPU:
                job_tensor = job_tensor.cuda()
                geeks_tensor = geeks_tensor.cuda()
            scores = self.forward(job_tensor, geeks_tensor)
            # print(scores)
            scores = scores.cpu()
            scores = scores.detach().numpy()
            predictions.append(scores)
        return predictions

    @staticmethod
    def batch_fit(model, optimizer, sample):
        # job, geek, label = sample.t()
        job = sample[:, 0].unsqueeze(dim=1)
        geek = sample[:, 1].unsqueeze(dim=1)
        label = sample[:, 2].unsqueeze(dim=1)
        # print('job size \n', job.shape)
        job = Variable(LongTensor(job))
        geek = Variable(LongTensor(geek))
        # label = label.view(label.shape[0], 1)
        label = label.float()
        if USE_GPU:
            job = job.cuda()
            geek = geek.cuda()
            label = label.cuda()
        # 前向传播计算损失
        out = model(job, geek)
        loss = model.criterion(out, label)
        # 后向传播计算梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item() * label.size(0)

