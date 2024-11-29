import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.layers.layer_norm import LayerNorm
from uer.utils import *

class TinyBertTarget(nn.Module):
    """
    TinyBERT exploits masked language modeling (MLM)
    and knowledge distillation (KD) for pretraining.
    """

    def __init__(self, args, vocab_size, teacher_model):
        super(TinyBertTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]
        
        if self.factorized_embedding_parameterization:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.layer_norm = LayerNorm(args.emb_size)
            self.mlm_linear_2 = nn.Linear(args.emb_size, self.vocab_size)
        else:
            self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.layer_norm = LayerNorm(args.hidden_size)
            self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        self.teacher_model = teacher_model
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: hidden, attention_list, hidden_list
            tgt_mlm: [batch_size x seq_length]
        """
        tgt_mlm = tgt[0]
        self.teacher_model.eval()
        tgt_mlm_1 = tgt_mlm
        hidden, attention_list, hidden_list, emb, seg = memory_bank
        teacher_outputs = self.teacher_model.encoder(emb, seg)
        output_mlm = self.teacher_model.target.act(self.teacher_model.target.mlm_linear_1(teacher_outputs[0]))
        output_mlm = self.teacher_model.target.layer_norm(output_mlm)
        if self.teacher_model.target.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        teacher_logits = self.teacher_model.target.mlm_linear_2(output_mlm)
        output_mlm = self.act(self.mlm_linear_1(hidden))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm_1 = tgt_mlm_1.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm_1 > 0, :]
        tgt_mlm_1 = tgt_mlm_1[tgt_mlm_1 > 0]
        student_logits = self.mlm_linear_2(output_mlm)

        student_layers = len(hidden_list)
        teacher_layers = len(teacher_outputs[2])
        assert teacher_layers % student_layers == 0
        mapping = [i * teacher_layers // student_layers for i in range(student_layers)]
         
        # Cross entropy loss.
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.log_softmax(student_logits, dim=1)
        logit_loss = -torch.mean(torch.sum(teacher_probs * student_probs, dim=1))

        # Attention loss.
        attention_loss = 0
        teacher_att = teacher_outputs[1]
        student_att = attention_list
        new_teacher_att = [teacher_att[mapping[i]] for i in range(student_layers)]
        for i in range(student_layers):
            student_att[i] = torch.where(student_att[i] < -1e2, torch.zeros_like(student_att[i]), student_att[i])
            new_teacher_att[i] = torch.where(new_teacher_att[i] < -1e2, torch.zeros_like(new_teacher_att[i]), new_teacher_att[i])
            #torch.all(new_teacher_att[i] == 0)
            is_s_z = torch.all(student_att[i] == 0)
            #print(is_t_z, is_s_z)
            attention_loss += nn.MSELoss()(student_att[i], new_teacher_att[i])
        
        # Hidden loss.
        hidden_loss = 0
        teacher_hidden = teacher_outputs[2]
        student_hidden = hidden_list
        new_teacher_hidden = [teacher_hidden[mapping[i]] for i in range(student_layers)]
        for i in range(student_layers):
            hidden_loss += nn.MSELoss()(student_hidden[i], new_teacher_hidden[i])
        
        return logit_loss, attention_loss, hidden_loss
            
        


        
