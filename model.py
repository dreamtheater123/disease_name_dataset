import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from transformers import set_seed

with open('./vocab/pure_weights.pkl', 'rb') as file:
    embedding = pkl.load(file)


class BILSTM_classi(nn.Module):
    """
    classification model for BILSTM as the backbone
    """
    def __init__(self, args):
        super(BILSTM_classi, self).__init__()

        self.args = args
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(embedding), padding_idx=0)
        self.LSTM = nn.LSTM(
            input_size=embedding.shape[1],
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.pred_fc = nn.Linear(512, args['dataset']['num_label'])
        # self.softmax = nn.Softmax()
        # parameter initialization
        self.parameter_init()

    def parameter_init(self):
        # ----- BILSTM Layer 1 -----
        # forward
        nn.init.kaiming_normal_(self.LSTM.weight_ih_l0)
        nn.init.constant_(self.LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.LSTM.weight_hh_l0)
        nn.init.constant_(self.LSTM.bias_hh_l0, val=0)
        # backward
        nn.init.kaiming_normal_(self.LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.LSTM.bias_hh_l0_reverse, val=0)

        # ----- BILSTM Layer 2 -----
        # forward
        nn.init.kaiming_normal_(self.LSTM.weight_ih_l1)
        nn.init.constant_(self.LSTM.bias_ih_l1, val=0)
        nn.init.orthogonal_(self.LSTM.weight_hh_l1)
        nn.init.constant_(self.LSTM.bias_hh_l1, val=0)
        # backward
        nn.init.kaiming_normal_(self.LSTM.weight_ih_l1_reverse)
        nn.init.constant_(self.LSTM.bias_ih_l1_reverse, val=0)
        nn.init.orthogonal_(self.LSTM.weight_hh_l1_reverse)
        nn.init.constant_(self.LSTM.bias_hh_l1_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc.bias, val=0)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.training)

    def forward(self, x):
        x = self.word_emb(x)
        x, _ = self.LSTM(x)
        x = torch.mean(x, dim=1)
        x = self.pred_fc(x)
        # x = self.softmax(x)

        return x


class Bert_CN(nn.Module):
    def __init__(self, args, bert_model):
        super(Bert_CN, self).__init__()
        set_seed(42)  # 设置这个可以在每次运行的时候固定transformer head的随机初始化参数，从而使每次运行结果一致
        self.args = args
        # self.bert_tokenizer = AutoTokenizer.from_pretrained("./bert_base_chinese")
        self.bert_model = bert_model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pt_batch):
        x = self.bert_model(**pt_batch)
        x = x.logits
        # x = self.softmax(x.logits)

        return x
