import torch
import torch.nn as nn
import torch.nn.functional as F



def linear_block(in_features, out_features, dropout=0.5, final_layer=False):
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    nn.init.kaiming_normal_(linear_layer.weight, a=0.25)
    nn.init.constant_(linear_layer.bias, val=0)

    if final_layer:
        return linear_layer
    else:
        layers = [linear_layer, 
                nn.Dropout(p=dropout, inplace=True),
                nn.BatchNorm1d(out_features), 
                nn.ReLU()]
        
    return nn.Sequential(*layers)




def maxpool1d(n_channels, pool_size=2, relu=True):
    if relu:
        layers = [nn.BatchNorm1d(n_channels), 
                #   nn.ReLU(inplace=True),
                  nn.PReLU(),
                  nn.MaxPool1d(kernel_size = pool_size, stride = 2)]
    else:
        layers = [nn.BatchNorm1d(n_channels),
                  nn.ReLU(inplace=True),
                  nn.MaxPool1d(kernel_size = pool_size, stride = 2)]
    
    return nn.Sequential(*layers)




def avgpool1d(n_channels, pool_size=2, relu=True):
    if relu:
        layers = [nn.BatchNorm1d(n_channels), 
                #   nn.ReLU(inplace=True),
                  nn.PReLU(),
                  nn.AvgPool1d(kernel_size = pool_size, stride = 2)]
    else:
        layers = [nn.BatchNorm1d(n_channels),
                  nn.ELU(inplace=True),
                  nn.AvgPool1d(kernel_size = pool_size, stride = 2)]
    
    return nn.Sequential(*layers)




class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 

        self.W_s = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.kaiming_normal_(self.W_s.data)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a_s = nn.Parameter(torch.empty(size=(out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.kaiming_normal_(self.a_s.data)                           ## initializing the attention-coefficient weights
        
        self.W_n = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.kaiming_normal_(self.W_n.data)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a_n = nn.Parameter(torch.empty(size=(out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.kaiming_normal_(self.a_n.data)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, inp, adj, inds, first_gat=True):
        h_s = torch.mm(inp, self.W_s)    
        h_n = torch.mm(inp, self.W_n)   
        e = self._prepare_attentional_mechanism_input(h_s, h_n, inds, first_gat)

        zero_vec = -9e15*torch.ones_like(e)                        
        
        if not first_gat: adj = adj[:,inds]
        attention = torch.where(adj > 0, e*adj, zero_vec)             ## assigning values of 'e' to those which has value>0 in adj matrix
        attention = F.softmax(attention, dim=0) 

        attention_sum = torch.sum(attention, axis = -1, keepdims = True)
        attention_s = 1/(1 + attention_sum)  
        attention_n = attention
        
        h_s = F.dropout(h_s, self.dropout, training=self.training)
        h_n = F.dropout(h_n, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention_n, h_n)              ## multiplying attention co-efficients with the input  -- dimension (#input X out_features)
        if first_gat: h_s = h_s[inds]
        h_prime = attention_s*h_s +  h_prime                     

        if self.concat:
            return self.relu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, h_s, h_n, inds, first_gat):
        
        h1 = torch.matmul(h_s, self.a_s)
        h2 = torch.matmul(h_n, self.a_n)
        
        if first_gat: h1=h1[inds]
        e = h1 + h2.T   # broadcast add  [inds]
        return self.relu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    




class GAT(nn.Module):
    def __init__(self, in_feat, embed_size, nheads=8, dropout=0.6, l_alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        ## creating attention layers for given number of heads
        self.attentions = [GraphAttentionLayer(in_feat, embed_size, dropout=dropout, alpha=l_alpha, concat=True) for _ in range(nheads)] 
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)     ## adding the modules for each head

    def forward(self, x, adj, inds, first_gat=True):
        x = torch.cat([att(x, adj, inds, first_gat) for att in self.attentions], dim=1)  

        return x  
    



class hic_model(nn.Module):
    def __init__(self, in_channels, nb_embed, nb_heads, dropout=0.5, l_alpha=0.2):
        super(hic_model, self).__init__()

        self.embed = nb_embed
        self.heads = nb_heads

        self.pool1 = nn.Sequential(
                    maxpool1d(n_channels=in_channels, pool_size=2),
                    maxpool1d(n_channels=in_channels, pool_size=2)
                    )
        
        self.pool2 = nn.Sequential(
                    avgpool1d(n_channels=in_channels, pool_size=2),
                    avgpool1d(n_channels=in_channels, pool_size=2)
                    )
        
        self.gat1 = GAT(in_feat = nb_embed,
                      embed_size=self.embed, 
                      nheads=self.heads, 
                      dropout=dropout, 
                      l_alpha=l_alpha)
        
        
        self.linear = nn.Sequential(
                    nn.Flatten(),
                    linear_block(self.embed*self.heads, 512, dropout=dropout),
                    linear_block(512, 128, dropout=dropout),
                    linear_block(128, 32, dropout=dropout),
                    linear_block(32, 8, dropout=dropout),
                    linear_block(8,1, dropout=dropout, final_layer=True),
                    nn.Sigmoid())
        
    def forward(self, x, adj, inds):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x = torch.cat([x1,x2],dim=1)

        x = torch.squeeze(x)
        x = x.T
        
        x = self.gat1(x, adj, inds, first_gat = True)
        x = self.linear(x)
        
        return x
    

