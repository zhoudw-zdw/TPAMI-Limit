import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np
from copy import deepcopy

def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []

    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))      

    return permuted_ids, permuted_labels

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)
        
        hdim=self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5) 
        self.seminorm=True 
        if args.dataset == 'cifar100':
            self.seminorm=False
        
    def split_instances(self, support_label,epoch):
        args = self.args
        #crriculum for num_way:
        total_epochs=args.epochs_base
        
        #Linear increment
        #self.current_way=int(5+float((args.sample_class-5)/total_epochs)*epoch)
        #Linear drop
        #self.current_way=int(args.sample_class-float((args.sample_class-5)/total_epochs)*epoch)
        #Equal
        #self.current_way=10
        #Random Sample
        #self.current_way=np.random.randint(5,args.sample_class)
        self.current_way=args.sample_class
        permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, num_shot=args.sample_shot, num_way=self.current_way, num_class=args.sample_class)
        index_label=(permuted_ids.view(args.num_tasks, args.sample_shot, self.current_way), torch.stack(permuted_labels))
        
        return index_label

    def forward(self, x_shot, x_query=None, shot_label=None,epoch=None):

        if self.mode == 'encoder':
            x_shot = self.encode(x_shot)
            return x_shot
        else:
            support_emb = self.encode(x_shot)
            query_emb = self.encode(x_query)
            index_label = self.split_instances(shot_label,epoch)
            logits = self._forward(support_emb, query_emb, index_label)
            return logits

    def _forward(self, support,query,index_label):
        
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        num_dim = support.shape[-1]
        # organize support data
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]     
        logit = []

        num_batch=1
        num_proto=self.args.num_classes
        num_query=query.shape[0]
        emb_dim = support.size(-1)
        query=query.unsqueeze(1)

        for tt in range(num_task):
            # combine proto with the global classifier
            global_mask = torch.eye(self.args.num_classes).cuda()  
            whole_support_index = support_labels[tt,:]
            global_mask[:, whole_support_index] = 0
            # construct local mask
            local_mask = one_hot(whole_support_index, self.args.num_classes)
            current_classifier = torch.mm(self.fc.weight.t(), global_mask) + torch.mm(proto[tt,:].t(), local_mask)            
            current_classifier = current_classifier.t() #100*64
            current_classifier=current_classifier.unsqueeze(0)
            
            current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            current_classifier = current_classifier.view(num_batch*num_query, num_proto, emb_dim)
            
            combined = torch.cat([current_classifier, query], 1) # Nk x (N + 1) x d, batch_size = NK
            combined = self.slf_attn(combined, combined, combined)
            # compute distance for all batches
            current_classifier, query = combined.split(num_proto, 1)
            

            if self.seminorm:
                #norm classifier
                current_classifier = F.normalize(current_classifier, dim=-1) # normalize for cosine distance
                logits = torch.bmm(query, current_classifier.permute([0,2,1])) /self.args.temperature
                logits = logits.view(-1, num_proto)
            else:
                #cosine
                logits=F.cosine_similarity(query,current_classifier,dim=-1)
                logits=logits*self.args.temperature

            
            logit.append(logits)
        logit = torch.cat(logit, 1)
        logit = logit.view(-1, self.args.num_classes)   

        return logit
    
    def updateclf(self,data,label):
        support_embs = self.encode(data)  
        num_dim = support_embs.shape[-1]
        #proto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d
        proto = support_embs.reshape(5, -1, num_dim).mean(dim=0) # N x d
        cls_unseen, _, _ = self.slf_attn(proto.unsqueeze(0), self.shared_key, self.shared_key)
        #cls_unseen = F.normalize(cls_unseen.squeeze(0), dim=1)
        cls_unseen=cls_unseen.squeeze(0)
        self.fc.weight.data[torch.min(label):torch.max(label)+1]=  cls_unseen
        
    def forward_many(self, query):
        #cls_seen = F.normalize(self.fc.weight, dim=1)   
        num_batch=1
        num_proto=self.args.num_classes
        
        emb_dim = query.size(-1)
        query=query.view(-1,1,emb_dim)
        num_query=query.shape[0]

        current_classifier = self.fc.weight.unsqueeze(0)
        current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        current_classifier = current_classifier.view(num_batch*num_query, num_proto, emb_dim)
        
        combined = torch.cat([current_classifier, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        # compute distance for all batches
        current_classifier, query = combined.split(num_proto, 1)
       
        if self.seminorm:
            #norm classifier
            current_classifier = F.normalize(current_classifier, dim=-1) # normalize for cosine distance
            logits = torch.bmm(query, current_classifier.permute([0,2,1])) /self.args.temperature
            logits = logits.view(-1, num_proto)
        else:
            #cosine
            logits=F.cosine_similarity(query,current_classifier,dim=-1)
            logits=logits*self.args.temperature
        return logits


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output