import torch.nn as nn
import torch
from functions import ReverseLayerF
import torch.nn.functional as F


class DSN(nn.Module):
    def __init__(self, field_size,code_size=100, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################
        self.s_encoder=nn.Sequential()
        self.s_encoder.add_module('sencoder',nn.Linear(300,200))
        self.s_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(200,100)


        #########################################
        # private target encoder
        #########################################
        self.t_encoder=nn.Sequential()
        self.t_encoder.add_module('sencoder',nn.Linear(300,200))
        self.t_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(200,100)



        ################################
        # shared encoder (dann_mnist)
        ################################
        self.sh_encoder=nn.Sequential()
        self.sh_encoder.add_module('sencoder',nn.Linear(300,200))
        self.sh_encoder.add_module('srelu',nn.ReLU(True))
        # self.fc1 = nn.Linear(image_size, h_dim)
        self.fc5 = nn.Linear(200,100)
        self.fc6 = nn.Linear(200,100)





        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se4', nn.Linear(in_features=int(field_size/2)*100, out_features=300))
        self.shared_encoder_pred_class.add_module('relu_se4', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se5', nn.Linear(in_features=300, out_features=2))
        self.shared_encoder_pred_class.add_module('d_softmax',nn.LogSoftmax(dim=1))

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se6', nn.Linear(in_features=100, out_features=100))
        self.shared_encoder_pred_domain.add_module('relu_se6', nn.ReLU(True))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('fc_se7', nn.Linear(in_features=100, out_features=2))

        ######################################
        # shared decoder (small decoder)
        ######################################
        self.sh_decoder=nn.Sequential()
        self.sh_decoder.add_module('shdecoder',nn.Linear(in_features=100, out_features=200))
        self.sh_decoder.add_module('shdecoder2',nn.ReLU(True))
        self.sh_decoder.add_module('shdecoder3',nn.Linear(in_features=200, out_features=300))
        # self.sh_decoder.add_module('shdecoder4',nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_data, mode, rec_scheme,field_size,p=0.0):

        result = []

        if mode == 'source':

            # source private encoder
            private_feat = self.s_encoder(input_data)
            mu1,var1=self.fc1(private_feat),self.fc2(private_feat)
            private_code = self.reparameterize(mu1, var1)

        elif mode == 'target':

            # target private encoder
            private_feat = self.t_encoder(input_data)
            mu1,var1=self.fc3(private_feat),self.fc4(private_feat)
            private_code = self.reparameterize(mu1,var1)

        result.append(private_code)

        # shared encoder
        shared_feat = self.sh_encoder(input_data)
        mu2,var2=self.fc5(shared_feat),self.fc6(shared_feat)
        shared_code = self.reparameterize(mu2, var2)
        result.append(shared_code)
        # print(shared_code)
        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        # print(domain_label.shape)
        result.append(domain_label)

        if mode == 'source':
            muu1 = torch.zeros(1,100).to('cuda')
            muu2 = torch.zeros(1,100).to('cuda')
            varr1 = torch.zeros(1,100).to('cuda')
            varr2 = torch.zeros(1,100).to('cuda')
            for hh in range(int(field_size/2)):
                muu1 = torch.cat((muu1,torch.unsqueeze(mu2[hh], 0).to('cuda')),dim=0)
                muu2 = torch.cat((muu2,torch.unsqueeze(mu2[hh+int(field_size/2)], 0).to('cuda')),dim=0)
            for hh in range(int(field_size/2)):
                varr1 = torch.cat((varr1,torch.unsqueeze(var2[hh], 0).to('cuda')),dim=0)
                varr2 = torch.cat((varr2,torch.unsqueeze(var2[hh+int(field_size/2)], 0).to('cuda')),dim=0)
            aa = torch.zeros(100).to('cuda')
            # zero_set=set()
            # for i,j in enumerate(bb):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            # for i,j in enumerate(dd):
            #     if (j.equal(nm)):
            #         zero_set.add(i)
            mu = (muu1[1:] - muu2[1:]).pow(2)
            var = (varr1[1:] - varr2[1:]).pow(2)
            muu = mu +var
            #print(muu.shape)
            for ii,jj in enumerate(muu):
                # if (ii in zero_set):
                #     aa = torch.cat((aa,torch.zeros(100).to('cuda')),dim=0)
                # else:
                if (ii==0):
                    aa = torch.cat((aa,0.6*jj),dim=0)
                elif (ii==1):
                    aa = torch.cat((aa,0.3*jj),dim=0)
                elif (ii==2):
                    aa = torch.cat((aa,0.1*jj),dim=0)
                elif (ii==3):
                    aa = torch.cat((aa,0*jj),dim=0)
                # aa = torch.cat((aa,jj),dim=0)


            # print(aa.shape)
            aa = torch.unsqueeze(aa[100:], 0).to('cuda')              #增加一维
            # print(aa.shape)

            # print(aa[100:].shape)
            # print(cc.shape)
            class_label = self.shared_encoder_pred_class(aa)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = self.reparameterize((mu1+mu2),(var1+var2))
            # union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        # print(union_code.shape)
        rec_code = self.sh_decoder(union_code)

        result.append(rec_code)
        result.append(mu1)
        result.append(var1)
        result.append(mu2)
        result.append(var2)

        return result



