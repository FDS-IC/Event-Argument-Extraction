class Config:
    def __init__(self, args):
        # ... existing code ...
        
        # 新增配置参数
        self.label_smoothing = 0.1  # 标签平滑系数
        self.num_attention_heads = 8  # 注意力头数
        self.use_focal_loss = True  # 是否使用Focal Loss
        self.focal_loss_gamma = 2.0  # Focal Loss的gamma参数
        self.focal_loss_alpha = 0.25  # Focal Loss的alpha参数
        
        # 更改为FinBERT模型路径
        self.bert_model_dir = "ProsusAI/finbert"
        
        # 添加FinBERT特定配置
        self.max_position_embeddings = 512
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1 