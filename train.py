import random
import os
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE
import torch.nn.functional as F
from test import test
import deepmatcher as dm
from deepmatcher.data import MatchingIterator
import torch.nn as nn



class SoftNLLLoss(nn.NLLLoss):
    def __init__(self, label_smoothing=0, weight=None, num_classes=2, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.register_buffer('weight', Variable(weight))

        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

        self.criterion = nn.KLDivLoss(**kwargs)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)

        if self.weight is not None:
            one_hot.mul_(self.weight)

        return self.criterion(input, one_hot)
class Statistics(object):
    def __init__(self):
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0
        self.start_time = time.time()

    def update(self, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples



    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples

    def examples_per_sec(self):
        return self.examples / (time.time() - self.start_time + 1)
def print_final_stats(epoch,  stats):
    print(('Finished Epoch {epoch} ||'
               '|| F1: {f1:6.2f} | Prec: {prec:6.2f} | '
               'Rec: {rec:6.2f} ||\n').format(
                   epoch=epoch,
                   f1=stats.f1(),
                   prec=stats.precision(),
                   rec=stats.recall()))
def compute_scores(output, target):
    predictions = output.max(1)[1].data
    correct = (predictions == target.data).float()
    incorrect = (1 - correct).float()
    positives = (target.data == 1).float()
    negatives = (target.data == 0).float()

    tp = torch.dot(correct, positives)
    tn = torch.dot(correct, negatives)
    fp = torch.dot(incorrect, negatives)
    fn = torch.dot(incorrect, positives)

    return tp, tn, fp, fn
######################
# params             #
######################


model_root = 'model'
cuda = True
cudnn.benchmark = True
lr = 1e-2
batch_size = 16
n_epoch = 50
step_decay_weight = 0.95
lr_decay_step = 5000
active_domain_loss_step = 2500
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9
device=None
if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif device == 'gpu':
    device = 'cuda'
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################
# load data           #
#######################

train, validation, test3 = dm.data.process(path='/home/learn/VAE/quan-vae-dataset/cit1',
        train='train.csv', validation='valid.csv', test='test.csv')
train2, validation2, test2 = dm.data.process(path='/home/learn/VAE/quan-vae-dataset/cit2',
    train='train.csv', validation='valid.csv', test='test.csv')
model = dm.MatchingModel(attr_summarizer='rnn')
model.run_train(train2 )
model.to('cuda')


sort_in_buckets = True
run_iter = MatchingIterator(
            train,
            train,
            True,
            batch_size=batch_size,
            device=device,
            sort_in_buckets=sort_in_buckets)

run_iter2 = MatchingIterator(
            train2,
            train2,
            True,
            batch_size=batch_size,
            device=device,
            sort_in_buckets=sort_in_buckets)
def attribut(fields,embeddings):
	batch = embeddings[fields[0]].data.shape[0]
	batch_list = torch.zeros(1,batch,300).to('cuda')
	for field in fields:
		attribut_tensor = torch.zeros(1,300).to('cuda')
		# value = getattr(input, field)
		# print('value',field,value)


				# Get token embedding
		value_embedding = embeddings[field]
	#     print('valuee',field,value_embedding)
		h = 0
		for i in value_embedding.data:
			d = value_embedding.lengths

			c = torch.zeros(300).to('cuda')
					# for j in i:
					#     # print(j)
			if (int(d[h]) !=2):
				for j in range(1,int(d[h])-1):
					c.add_(i[j])
			
				c = torch.div(c,d[h]-2)         #做平均
			# print(c)
			h = h+1
			c = torch.unsqueeze(c, 0).to('cuda')              #增加一维
			attribut_tensor = torch.cat((attribut_tensor,c),dim=0)
			# print(field,'1',attribut_tensor.shape)
		attribut_tensor = torch.unsqueeze(attribut_tensor[1:],0)
		# print(field,'2',attribut_tensor.shape)
		batch_list = torch.cat((batch_list,attribut_tensor),dim=0)
		# print(field,'3',batch_list.shape)
	batch_list = batch_list[1:]
	return batch_list.permute(1,0,2)
def getbatch(input_batch,model,train):
	embeddings = {}
	for name in model.meta.all_text_fields:
			
		attr_input = getattr(input_batch, name)
				
		embeddings[name] = model.embed[name](attr_input)
	meta_data = []
	for name in model.meta.canonical_text_fields:
		left, right = model.meta.text_fields[name]

            # new_add
		len1 = embeddings[left].lengths
		len2 = embeddings[right].lengths
		temp = []
		for i, j in zip(len1, len2):
			if i == 2 or j == 2:
				temp.append(0)
			else:
				temp.append(1)
		meta_data.append(temp)
				# print(embeddings['left_title'].data.shape)
	left_fields = train.all_left_fields
	right_fields = train.all_right_fields
	batch_left_tensor = attribut(left_fields,embeddings)
	batch_right_tensor = attribut(right_fields,embeddings)
	# print(model.meta.canonical_text_fields)
	return batch_left_tensor,batch_right_tensor,len(left_fields*2)
#####################
#  load model       #
#####################
field_sizee=8
my_net = DSN(field_size=field_sizee)

#####################
# setup optimizer   #
#####################


def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print ('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer

label_smoothing=0.05
pos_neg_ratio=3


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
loss_classification=None
if loss_classification is None:
    if pos_neg_ratio is None:
        pos_neg_ratio = 1
    else:
        assert pos_neg_ratio > 0
    pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)

    neg_weight = 2 - pos_weight
loss_classification = SoftNLLLoss(label_smoothing,torch.Tensor([neg_weight, pos_weight]))
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

#############################
# training network          #
#############################


len_dataloader = min(len(run_iter), len(run_iter2))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in range(n_epoch):

    data_source_iter = iter(run_iter)
    data_target_iter = iter(run_iter2)

    i = 0
    cum_stats = Statistics()
    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = data_target_iter.__next__()

        # target_inputv_img = torch.zeros(1,300).to('cuda')
        batch_left_tensor,batch_right_tensor,field_size = getbatch(data_target,model,train)
        # for bb,dd in zip(batch_left_tensor,batch_right_tensor):
        #     target_inputv_img=torch.cat((target_inputv_img,torch.cat((bb,dd),dim=0)),dim=0)
        my_net.zero_grad()
        loss = 0
        label_attr = model.meta.label_field
        batch_size = len(getattr(data_target,label_attr))
        t_label = getattr(data_target, label_attr)
        class_label = torch.LongTensor(batch_size)

        # print(domain_label.shape)

        if cuda:
            t_label = t_label.cuda()
            # target_inputv_img = target_inputv_img.cuda()
            class_label = class_label.cuda()
            # domain_label = domain_label.cuda()
        class_label.resize_as_(t_label).copy_(t_label)
        target_classv_label = Variable(class_label)


        for ii,jj in zip(batch_left_tensor,batch_right_tensor):
            target_inputv_img=torch.cat((ii,jj),dim=0)

            domain_label = torch.ones(field_size)
            domain_label = domain_label.long()
            if cuda:
                domain_label = domain_label.cuda()
            target_domainv_label = Variable(domain_label)
        # target_inputv_img=target_inputv_img[1:]

            if current_step > active_domain_loss_step:
                p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
                p = 2. / (1. + np.exp(-10 * p)) - 1

                # activate domain loss
                # print(111)
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', field_size=field_size,p=p)
                target_privte_code, target_share_code, target_domain_label, target_rec_code,mu1,var1,mu2,var2 = result
                target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
                loss += target_dann
            else:
                target_dann = Variable(torch.zeros(1).float().cuda())
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all',field_size=field_size)
                target_privte_code, target_share_code, _, target_rec_code,mu1,var1,mu2,var2 = result

            # target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
            # loss += target_diff
            # zero_set=set()
            # for i,j in enumerate(bb):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            # for i,j in enumerate(dd):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            mu = (mu1 - mu2).pow(2)
            var = (var1 - var2).pow(2)
            muu = mu +var
            #print(muu.shape)
            # muu_sum=(muu.sum(dim=1)).mean()
            target_diff= beta_weight * (muu.sum(dim=1)).mean()
            loss += target_diff
            # for ii,jj in enumerate(muu_sum):
            #     if (ii in zero_set):
            #         muu_sum[ii]=torch.tensor(0).to('cuda')

            # print(muu_sum)

            target_mse = alpha_weight * F.mse_loss(target_rec_code, target_inputv_img,size_average=False)
            loss += target_mse
            target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
            loss += target_simse
            kl_div1 = - 0.5 * torch.sum(1 + var1 - mu1.pow(2) - var1.exp())
            kl_div2 = - 0.5 * torch.sum(1 + var2 - mu2.pow(2) - var2.exp())
            # loss+=kl_div1
            # loss+=kl_div2
        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = data_source_iter.__next__()

        # source_inputv_img=torch.zeros(1,300).to('cuda')
        batch_left_tensor,batch_right_tensor,field_size = getbatch(data_source,model,train)
        # for bb,dd in zip(batch_left_tensor,batch_right_tensor):
        #     source_inputv_img=torch.cat((source_inputv_img,torch.cat((bb,dd),dim=0)),dim=0)
        my_net.zero_grad()
        label_attr = model.meta.label_field
        batch_size = len(getattr(data_source,label_attr))
        s_label = getattr(data_source, label_attr)
        class_label = torch.LongTensor(batch_size)
        if cuda:
            s_label = s_label.cuda()
            # target_inputv_img = target_inputv_img.cuda()
            class_label = class_label.cuda()
            # domain_label = domain_label.cuda()


        loss = 0
        label_tol=torch.zeros(1,2).to('cuda')


        class_label.resize_as_(s_label).copy_(s_label)
        source_classv_label = Variable(class_label)


        for ii,jj in zip(batch_left_tensor,batch_right_tensor):
            source_inputv_img=torch.cat((ii,jj),dim=0)
            domain_label = torch.zeros(field_size)
            domain_label = domain_label.long()
            if cuda:
                domain_label = domain_label.cuda()
            source_domainv_label = Variable(domain_label)
        # source_inputv_img=source_inputv_img[1:]

            if current_step > active_domain_loss_step:

                # activate domain loss
                # print(222)
                result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', field_size=field_size,p=p)
                source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code,mu1,var1,mu2,var2 = result
                source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
                loss += source_dann
            else:
                source_dann = Variable(torch.zeros(1).float().cuda())
                result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all',field_size=field_size)
                source_privte_code, source_share_code, _, source_class_label, source_rec_code,mu1,var1,mu2,var2 = result
            label_tol=torch.cat((label_tol,source_class_label),dim=0)


            # source_classification = loss_classification(source_class_label, source_classv_label)

            # loss += source_classification

            # source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
            # loss += source_diff
            # zero_set=set()
            # for i,j in enumerate(bb):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            # for i,j in enumerate(dd):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            mu = (mu1 - mu2).pow(2)
            var = (var1 - var2).pow(2)
            muu = mu +var
            #print(muu.shape)
            # muu_sum=(muu.sum(dim=1)).mean()
            source_diff = beta_weight * (muu.sum(dim=1)).mean()
            loss += source_diff
            # for ii,jj in enumerate(muu_sum):
            #     if (ii in zero_set):
            #         muu_sum[ii]=torch.tensor(0).to('cuda')

            # print(muu_sum)
            source_mse = alpha_weight * F.mse_loss(source_rec_code, source_inputv_img,size_average=False)
            loss += source_mse
            source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
            loss += source_simse
            kl_div1 = - 0.5 * torch.sum(1 + var1 - mu1.pow(2) - var1.exp())
            kl_div2 = - 0.5 * torch.sum(1 + var2 - mu2.pow(2) - var2.exp())
            # loss+=kl_div1
            # loss+=kl_div2
        source_classification = loss_classification(label_tol[1:], source_classv_label)
        loss += source_classification
        scores = compute_scores(label_tol[1:], getattr(data_source, label_attr))
        cum_stats.update(*scores)
        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1


    print_final_stats(epoch + 1, cum_stats)
    # print ('source_classification: %f, source_dann: %f, source_diff: %f, ' \
    #       'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
    #       'target_mse: %f, target_simse: %f' \
    #       % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
    #          source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
    #          target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))

    # print 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    torch.save(my_net.state_dict(), '/home/learn/VAERDSN/' + 'dsn_mnist_epoch_' + str(epoch) + '.pth')
    test(epoch=epoch, field_sizee=field_sizee)
    # test(epoch=epoch, name='mnist_m')

print ('done')


