import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from model.linear_crf_inferencer import LinearCRF
from typing import Tuple, Optional, Union
import numpy as np


class HierarchicalLSTMBertCRF(BertPreTrainedModel):
    def __init__(self, cfig):
        super(HierarchicalLSTMBertCRF, self).__init__(cfig)
        self.num_labels = len(cfig.label2idx)
        self.bert = BertModel(cfig)

        # 主LSTM层（处理原始序列）
        self.lstm = nn.LSTM(
            input_size=cfig.hidden_size,
            hidden_size=cfig.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=cfig.hidden_dropout_prob if cfig.hidden_dropout_prob > 0 else 0.1
        )

        self.edu_window_size = cfig.edu_window_size if hasattr(cfig, 'edu_window_size') else 3
        self.edu_stride = cfig.edu_stride if hasattr(cfig, 'edu_stride') else 1

        self.edu_lstm = nn.LSTM(
            input_size=cfig.hidden_size,
            hidden_size=cfig.hidden_size // 2,  # 双向LSTM，最终输出hidden_size
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=cfig.hidden_dropout_prob if cfig.hidden_dropout_prob > 0 else 0.1
        )

        self.edu_attention = nn.MultiheadAttention(
            embed_dim=cfig.hidden_size,
            num_heads=8,
            dropout=cfig.hidden_dropout_prob
        )
        self.sentence_attention = nn.MultiheadAttention(
            embed_dim=cfig.hidden_size,
            num_heads=8,
            dropout=cfig.hidden_dropout_prob
        )

        # 特征融合层（调整为融合4种特征）
        self.feature_fusion = nn.Sequential(
            nn.Linear(cfig.hidden_size * 4, cfig.hidden_size),  # BERT + LSTM + EDU-LSTM + EDU-Attn
            nn.ReLU(),
            nn.Dropout(cfig.hidden_dropout_prob)
        )

        # 其他层
        self.dropout = nn.Dropout(cfig.hidden_dropout_prob)
        self.classifier = nn.Linear(cfig.hidden_size, self.num_labels)
        self.inferencer = LinearCRF(cfig)
        self.layer_norm = nn.LayerNorm(cfig.hidden_size)

    def split_into_edus(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = sequence_output.shape

        # 计算有效EDU数量
        num_edus = (seq_len - self.edu_window_size) // self.edu_stride + 1

        # 初始化输出张量（明确保持维度）
        edu_features = torch.zeros((batch_size, num_edus, hidden_size),
                                   device=sequence_output.device)
        edu_mask = torch.zeros((batch_size, num_edus),
                               device=attention_mask.device)

        # 滑动窗口处理
        for i in range(num_edus):
            start = i * self.edu_stride
            end = start + self.edu_window_size
            window = sequence_output[:, start:end, :]  # (batch, window, hidden)
            edu_features[:, i, :] = window.mean(dim=1)  # 保持hidden维度

            # 处理mask
        unfolded_mask = attention_mask.unfold(1, self.edu_window_size, self.edu_stride)
        edu_mask = torch.any(unfolded_mask.bool(), dim=2)

        return edu_features, edu_mask

    def forward(
            self,
            input_ids: torch.Tensor,
            input_seq_lens: Optional[torch.Tensor] = None,
            annotation_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # BERT特征提取
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        sequence_output = bert_outputs[0]  # (batch, seq_len, hidden)
        edu_features, edu_mask = self.split_into_edus(sequence_output, attention_mask)  # (batch, num_edus, hidden)
        self.edu_lstm.flatten_parameters()
        edu_lstm_output, _ = self.edu_lstm(edu_features)  # (batch, num_edus, hidden)
        edu_attn_output, _ = self.edu_attention(
            edu_features.transpose(0, 1),
            edu_features.transpose(0, 1),
            edu_features.transpose(0, 1),
            key_padding_mask=~edu_mask
        )
        edu_attn_output = edu_attn_output.transpose(0, 1)  # (batch, num_edus, hidden)
        # 主LSTM
        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, hidden)
        # 句子级注意力
        sentence_attn_output, _ = self.sentence_attention(
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            sequence_output.transpose(0, 1),
            key_padding_mask=~attention_mask
        )
        sentence_attn_output = sentence_attn_output.transpose(0, 1)  # (batch, seq_len, hidden)

        # 5. 特征对齐（统一到seq_len长度）
        target_len = sequence_output.size(1)

        def align_features(x, original_len, target_len):
            if original_len == target_len:
                return x
            return F.interpolate(
                x.permute(0, 2, 1),
                size=target_len,
                mode='linear'
            ).permute(0, 2, 1)

        edu_lstm_output = align_features(edu_lstm_output, edu_lstm_output.size(1), target_len)
        edu_attn_output = align_features(edu_attn_output, edu_attn_output.size(1), target_len)

        # 6. 特征融合（BERT + LSTM + EDU-LSTM + EDU-Attn）
        fused_input = torch.cat([
            sequence_output,  # BERT原始特征
            lstm_output,  # 句子级LSTM特征
            edu_lstm_output,  # EDU级LSTM特征
            edu_attn_output  # EDU注意力特征
        ], dim=-1)

        fused_features = self.feature_fusion(fused_input)

        # 7. CRF解码
        logits = self.classifier(fused_features)
        if self.training:
            if labels is not None:
                unlabed_score, labeled_score = self.inferencer(logits, input_seq_lens, labels, attention_mask)
                return unlabed_score - labeled_score
        else:
            return self.inferencer.decode(logits, input_seq_lens, annotation_mask)