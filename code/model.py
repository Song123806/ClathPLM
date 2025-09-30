import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


class T5_CNN_ATT(nn.Module):
    def __init__(self, input_size=1024, reduced_dim=128):
        super(T5_CNN_ATT, self).__init__()

        # CNN layers (no initial dimensionality reduction)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(768),
            nn.ELU()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU()
        )

        # CNN layers (no initial dimensionality reduction)
        self.cnn3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU()
        )

        self.selfattention = selfAttention(8, 256, 256)
        # Dimensionality reduction after CNN
        self.fc_reducer = nn.Sequential(
            nn.Linear(256, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.ELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Add channel dimension for CNN
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)

        # Residual CNN
        cnn_out = self.cnn1(x)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.cnn3(cnn_out)

        cnn_out = cnn_out.permute(0, 2, 1)
        att_out = self.selfattention(cnn_out)
        # Remove channel dimension and reduce
        att_out = att_out.squeeze(1)
        reduced_out = self.fc_reducer(att_out)

        return reduced_out


class BERT_CNN_ATT(nn.Module):
    def __init__(self, input_size=1024, reduced_dim=128):
        super(BERT_CNN_ATT, self).__init__()

        # CNN layers (no initial dimensionality reduction)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1024, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(768),
            nn.ELU()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU()
        )

        # CNN layers (no initial dimensionality reduction)
        self.cnn3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU()
        )

        self.selfattention = selfAttention(8, 256, 256)
        # Dimensionality reduction after CNN
        self.fc_reducer = nn.Sequential(
            nn.Linear(256, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.ELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Add channel dimension for CNN
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        # Residual CNN
        cnn_out = self.cnn1(x)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.cnn3(cnn_out)

        cnn_out = cnn_out.permute(0, 2, 1)
        att_out = self.selfattention(cnn_out)
        # print(att_out.shape)
        # Remove channel dimension and reduce
        att_out = att_out.squeeze(1)
        # print(att_out.shape)
        reduced_out = self.fc_reducer(att_out)

        return reduced_out
class ESM3_CNN_ATT(nn.Module):
    def __init__(self, input_size=1536, reduced_dim=128): #256):
        super(ESM3_CNN_ATT, self).__init__()

        # CNN layers (no initial dimensionality reduction)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1536, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(768),
            nn.ELU()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU()
        )

        # CNN layers (no initial dimensionality reduction)
        self.cnn3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU()
        )


        self.selfattention = selfAttention(8, 256, 256)
        # Dimensionality reduction after CNN
        self.fc_reducer = nn.Sequential(
            nn.Linear(256, reduced_dim),
            nn.BatchNorm1d(reduced_dim),
            nn.ELU(),
            nn.Dropout(0.1)
        )



    def forward(self, x):
        # Add channel dimension for CNN
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        #print(x.shape)
        # Residual CNN
        cnn_out = self.cnn1(x)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.cnn3(cnn_out)

        cnn_out = cnn_out.permute(0, 2, 1)
        att_out = self.selfattention(cnn_out)
        #print(att_out.shape)
        # Remove channel dimension and reduce
        att_out = att_out.squeeze(1)
        #print(att_out.shape)
        reduced_out = self.fc_reducer(att_out)

        return reduced_out


# 三特征模型
class TripleFeatureModel(nn.Module):
    def __init__(self, t5_dim=1024, bert_dim=1024, esm3_dim=1536, num_classes=2):
        super(TripleFeatureModel, self).__init__()

        # 三个独立的特征提取分支
        self.t5_extractor = T5_CNN_ATT(t5_dim)
        self.bert_extractor = BERT_CNN_ATT(bert_dim)
        self.esm3_extractor = ESM3_CNN_ATT(esm3_dim)

        # 最终分类器（调整输入维度）
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),  # 三个特征的输出拼接
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )

    def forward(self, t5_features, bert_features, esm3_features):
        # 分别处理三个特征
        t5_out = self.t5_extractor(t5_features)
        bert_out = self.bert_extractor(bert_features)
        esm3_out = self.esm3_extractor(esm3_features)

        # 拼接特征
        combined = torch.cat((t5_out, bert_out, esm3_out), dim=1)

        # 最终分类
        output = self.classifier(combined)

        return output
