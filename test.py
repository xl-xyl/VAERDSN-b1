import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from model_compat import DSN
import time
import deepmatcher as dm
from deepmatcher.data import MatchingIterator

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
				# print(embeddings['left_title'].data.shape)
	left_fields = train.all_left_fields
	right_fields = train.all_right_fields
	batch_left_tensor = attribut(left_fields,embeddings)
	batch_right_tensor = attribut(right_fields,embeddings)
	return batch_left_tensor,batch_right_tensor,len(left_fields*2)

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
def test(epoch,field_sizee):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 16
    image_size = 28

    ###################
    # load data       #
    ###################

    model_root='model'
    ####################
    # load model       #
    ####################

    my_net = DSN(field_size=field_sizee)
    checkpoint = torch.load(os.path.join('/home/learn/VAERDSN', 'dsn_mnist_epoch_' + str(epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    ####################
    # transform image  #
    ####################



    device=None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'gpu':
        device = 'cuda'
    train, _, test2 = dm.data.process(path='/home/learn/VAE/quan-vae-dataset/cit2',
    train='train.csv', validation='valid.csv', test='test.csv')
    model = dm.MatchingModel(attr_summarizer='rnn')
    model.run_train(train )
    model.to('cuda')


    sort_in_buckets = True
    run_iter = MatchingIterator(
                test2,
                train,
                False,
                batch_size=batch_size,
                device=device,
                sort_in_buckets=sort_in_buckets)

    len_dataloader = len(run_iter)
    data_iter = iter(run_iter)

    i = 0

    cum_stats = Statistics()
    while i < len_dataloader:

        data_input = data_iter.__next__()
        # img, label = data_input
        batch_left_tensor,batch_right_tensor,field_size = getbatch(data_input,model,test2)
        # for bb,dd in zip(batch_left_tensor,batch_right_tensor):
        #     source_inputv_img=torch.cat((source_inputv_img,torch.cat((bb,dd),dim=0)),dim=0)
		
        label_attr = model.meta.label_field

        label_tol=torch.zeros(1,2).to('cuda')



        for ii,jj in zip(batch_left_tensor,batch_right_tensor):
            source_inputv_img=torch.cat((ii,jj),dim=0)
            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='share',field_size=field_size)
            label_tol=torch.cat((label_tol,result[3]),dim=0)





        # result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='share',field_size=field_size)
        # pred = result[3]








        i += 1
        scores = compute_scores(label_tol[1:], getattr(data_input, label_attr))
        cum_stats.update(*scores)
    print_final_stats(epoch + 1, cum_stats)


