import torch
import torch.nn as nn
import csv
from torchtext.legacy.data import Field, BucketIterator, Example, Dataset

import spacy
from spacy.tokenizer import Tokenizer
import numpy as np

import random
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn.functional as F

SEED = 1234

random.seed(SEED)  # 生成了多个随机的种子，以后想要复现相同的结果，设置数字一样为1234即可，1234会生成特定的随机数
np.random.seed(SEED)
torch.manual_seed(SEED)  # cpu设置
torch.cuda.manual_seed(SEED)  # gpu设置
torch.backends.cudnn.deterministic = True




def tokenize_en(text):
    breakp_loc = []
    times = 0
    times2 = 0
    doc = spacy_en.tokenizer(text)  # 对合并后的文本进行分词操作
    
    for t in doc:
        times += 1
        
        #如果已经结束了 board end 应该跳过
        if t.text == 'end':
            breakp_loc.append(times)
        #每条网络划分为一个词，当看到网络结束标志 end
        #elif t.text == 'CLINE':
        #   breakp_loc.append(times)
        #if t.text == ''
        if t.text == 'boardend':
            #print(times)
            #第一次看到board 以后代表board end 记录下board end 代表板子结束就可以 不必记录很多board end 
            breakp_loc.append(times)
            
            break
    
            
    
    with doc.retokenize() as retokenizer:
        
        retokenizer.merge(doc[0:breakp_loc[0]])
        
        for i in range(len(breakp_loc)):

            if i + 1 < len(breakp_loc):
                retokenizer.merge(doc[breakp_loc[i]:breakp_loc[i + 1]])
    for t in doc:
        times2 += 1
        if t.text == 'boardend':
            #print(doc[times2])
            break
    doc = doc[0: times2]
    
    return [t.text for t in doc]



def tokenize_enm(text):
    
    doc = spacy_en.tokenizer(text)
    '''
    
    '''
    #breakp_loc = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    #breakp_loc = [0, 150, 300, 450]
    #with doc.retokenize() as retokenizer:
        
        #for i in range(len(breakp_loc)):
        #    if i + 1 < len(breakp_loc):
        #        retokenizer.merge(doc[breakp_loc[i]:breakp_loc[i + 1]])
        #retokenizer.merge(doc[breakp_loc[-1]:])
    #    retokenizer.merge(doc[0:])
    return [t.text for t in doc]


spacy_en = spacy.load('en_core_web_sm')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SRC = Field(tokenize=tokenize_en,
            init_token='<start>',
            eos_token='<end>',
            sequential=True,
            lower=False,
            batch_first=True)

TRG = Field(tokenize=tokenize_enm,
            init_token='<start>',
            eos_token='<end>',
            sequential=True,
            lower=False,
            batch_first=True)

data = []
train_path = 'D:\LCY\simulation\\train_set'
test_path = 'D:\LCY\simulation\\test_set'

#这里遍历文件夹主要是处理形成数据集

train_examples = []
test_examples = []
for filename in os.listdir(train_path):
 
 #if filename.endswith('csv'):  #这个条件视情况决定加不加
  with open(os.path.join(train_path, filename), 'r', encoding='utf-8') as csvfile:
    
    reader = csv.reader(csvfile)
    
    merged_nets = ""
    merged_third_col = ""
    last_val = ""
    skip_first = True  # 跳过第一个非空数据的标志
    for i, row in enumerate(reader):
        if skip_first:
            skip_first = False
            continue  # 跳过第一个非空数据
        if row[0]:  # 检查第一列是否为空
            merged_nets += row[0] + ' '  # 合并nets列的数据
        #if row[1] and not last_val:  # 检查第二列是否为空且last_val尚未被赋值
        if row[1]:    
            #last_val = row[1]   # 更新last_val为第二列的第一个非空值
            last_val += row[1] + ' '
            
        
    example = {'nets': merged_nets.strip(), 'val': last_val.strip()}
   
    src_text = example['nets']
    
    
    trg_text = example['val']
    example2 = Example.fromlist([src_text, trg_text], fields=[('nets', SRC), ('val', TRG)])
    train_examples.append(example2)


for filename in os.listdir(test_path):
 #print(filename)
 #if filename.endswith('csv'):  #这个条件视情况决定加不加
 with open(os.path.join(test_path, filename), 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    merged_nets = ""
    merged_third_col = ""
    last_val = ""
    skip_first = True  # 跳过第一个非空数据的标志
    for i, row in enumerate(reader):
        if skip_first:
            skip_first = False
            continue  # 跳过第一个非空数据
        if row[0]:  # 检查第一列是否为空
            merged_nets += row[0] + ' '  # 合并nets列的数据
        #if row[1] and not last_val:  # 检查第二列是否为空且last_val尚未被赋值
        if row[1]:    
            #last_val = row[1]   # 更新last_val为第二列的第一个非空值
            last_val += row[1] + ' '
            
        
    example = {'nets': merged_nets.strip(), 'val': last_val.strip()}
   
    src_text = example['nets']
    

    trg_text = example['val']
    example2 = Example.fromlist([src_text, trg_text], fields=[('nets', SRC), ('val', TRG)])
    test_examples.append(example2)

    


# 创建数据集


train_set = Dataset(train_examples, fields=[('nets', SRC), ('val', TRG)])
test_set = Dataset(test_examples, fields=[('nets', SRC), ('val', TRG)])
# 构建词汇表
SRC.build_vocab(train_set, min_freq=1)
TRG.build_vocab(train_set, min_freq=1)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# 创建迭代器
BATCH_SIZE = 64
train_iterator = BucketIterator(train_set,
                          batch_size=BATCH_SIZE,
                          sort=False,
                          device=device,
                          shuffle=True)

test_iterator = BucketIterator(test_set,
                          batch_size=BATCH_SIZE,
                          sort=False,
                          device=device,
                          shuffle=True)
#def test_iterator(iterator):
#    for i, batch in enumerate(iterator):
#        print(f"Batch {i+1}:")
#        print("Source Text (src):", [SRC.vocab.itos[token] for token in batch.nets[0].tolist()])
#        print("Target Text (trg):", [TRG.vocab.itos[token] for token in batch.val[0].tolist()])
#        # 如果你的目标值是数值，可以直接打印数值
#        # print("Target Values (trg):", batch.val)
#        print(batch.val.shape)
#        print("\n")
#        # 限制输出的批次数量，防止输出太多
#        if i == 2:  # 只输出前3个批次
#            break
#
# #调用测试函数
#test_iterator(train_iterator)

# print(valid_data.vocab_itos)
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=501):
        super().__init__()

        self.device = device
        self.hid_dim = hid_dim
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.max_length = max_length

        #self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src1, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src1.shape[0]
        src_len1 = src1.shape[1]

         #生成正弦余弦位置编码  
        
        position_encodings = self.positional_encoding_pytorch(src_len1, self.hid_dim)  
        position_encodings = position_encodings.to(self.device)  
        #print((self.tok_embedding(src1) * self.scale).shape)
        #print(position_encodings.shape)
        # 将位置编码加到词嵌入上  
        src1 = self.dropout((self.tok_embedding(src1) * self.scale) + position_encodings)  
        #pos = torch.arange(0, src_len1).unsqueeze(0).repeat(batch_size, 1).to(self.device)


        # pos = [batch size, src len]

        #src1 = self.dropout((self.tok_embedding(src1) * self.scale) + self.pos_embedding(pos))
        #src1 = self.dropout(self.tok_embedding(src1) + self.pos_embedding(pos))
        #经过dropout层后变成了三维的张量
        for layer in self.layers:
            src1 = layer(src1, src_mask)
        return src1
        # src = [batch size, src len, hid dim]
    @staticmethod  
    def positional_encoding_pytorch(seq_len, d_model):  
        """  
        使用PyTorch生成位置编码  
        """  
        pe = torch.zeros(seq_len, d_model)  
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        return pe  
        

    


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()
        # 层归一化
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        # 也是层归一化
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        # 多头自注意力层，输入维度是hid_dim,头数量是n_heads,dropout的概率为"dropout"
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x




class transformerRegression(nn.Module):  
    def __init__(self, encoder, src_pad_idx, device):  
        super().__init__()  
  
        self.encoder = encoder  
        self.src_pad_idx = src_pad_idx  
        self.device = device  
        self.relu = nn.ReLU()
        self.fc_encoder_output = nn.Linear(HID_DIM, HID_DIM*2)  # 假设我们想要扩展特征维度  
        
        self.fc2 = nn.Linear(HID_DIM*2, HID_DIM*3)
        self.fc3 = nn.Linear(HID_DIM*3, HID_DIM*4)
        self.fc4 = nn.Linear(HID_DIM*4, HID_DIM*8)
        # 通常只需要一个线性层来将编码器的输出映射到回归目标的维度  
         # 假设OUTPUT_DIM是500
        self.output_layer = nn.Linear(HID_DIM*8, 501)
        
  
    def make_src_mask(self, src):  
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  
        return src_mask  
  
    def forward(self, src):  
        # src = [batch size, src len]  
        src_mask = self.make_src_mask(src)  
  
        # 编码器处理输入序列  
        enc_src = self.encoder(src, src_mask)  
  
        # 编码器输出通常是[batch size, src len, hid dim]，但我们可能想要一个固定的输出维度  
        # 我们可以选择对编码器的输出进行平均或最大池化等操作来获取一个固定大小的表示  
        # 这里我们选择平均池化作为示例  
        
        # 添加的全连接层处理编码器输出  
        enc_src_processed = self.relu(self.fc_encoder_output(enc_src))  
        enc_src_fc2 = self.relu(self.fc2(enc_src_processed))
        enc_src_fc3 = self.relu(self.fc3(enc_src_fc2))
        enc_src_fc4 = self.relu(self.fc4(enc_src_fc3))
        #enc_src_processed = F.gelu(self.fc_encoder_output(enc_src))  
        #enc_src_fc2 = F.gelu(self.fc2(enc_src_processed))
        #enc_src_fc3 = F.gelu(self.fc3(enc_src_fc2))
        #enc_src_fc4 = F.gelu(self.fc4(enc_src_fc3))
        # 对处理后的编码器输出应用平均池化  
        enc_src_avg_pooled = torch.mean(enc_src_fc4, dim=1)  
  
        # 通过线性层将编码器的输出映射到所需的输出维度  
        output = self.output_layer(enc_src_avg_pooled)  
        
        #output = torch.sigmoid(output)

        
        return output  



INPUT_DIM = len(SRC.vocab)

OUTPUT_DIM = len(TRG.vocab)
#INPUT_DIM = 114
#OUTPUT_DIM = 177210
HID_DIM = 256 # 之前是256
ENC_LAYERS = 8
 # 之前是3
ENC_HEADS = 2 # 之前是8

ENC_PF_DIM = 512  # 之前是512

ENC_DROPOUT = 0.1  # 之前是0.1


print("INPUT_DIM = {}".format(INPUT_DIM))
print("OUTPUT_DIM = {}".format(OUTPUT_DIM))

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)  # 编码器设置
 
 # 解码器设置

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]


model = transformerRegression(enc, SRC_PAD_IDX, device).to(device)  # 模型设置


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0001 #之前是0.0001
WEIGHT_DECAY = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay = WEIGHT_DECAY)
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
#criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
criterion = nn.MSELoss()



def train(model, iterator, optimizer, criterion, clip, trg_min, trg_max):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):  # 对于每个iterator 而言 会有每个text 和val 从这里着手
        #print("iterator的长度是：")
        #print(len(iterator))
        src1 = batch.nets  # src -> text
        trg = batch.val   # trg -> val
        
        normalized_trg = 10*(trg - trg_min) / (trg_max - trg_min)
        optimizer.zero_grad()
        # 以转换过的张量数据为 输入 与训练数据
        output = model(src1)  

        #print("trg 的具体数值为", normalized_trg.dtype)
        #print(output2)
        #output, _ = model(src, trg[:, :-1])
        # output = [batch size, trg len - 1, output dim]

        #output = [batch size, output dim]
        #trg = [batch size, trg len]
        
        # trg = [batch size, trg len]
        

        output = output.contiguous().view(-1)  # 展平除了最后一个维度以外的所有维度
        normalized_trg = normalized_trg.contiguous().view(-1).float()
        #print("output = ", output)
        #print("trg = ", trg)
        #print(output[:, :-1].shape)
        loss = criterion(output, normalized_trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪

        optimizer.step()

        epoch_loss += loss.item()
    
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, trg_min, trg_max):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src1 = batch.nets  # src -> text
            trg = batch.val  # trg -> val
              # src -> text

            output = model(src1)
            
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]
            normalized_trg = 10*(trg - trg_min) / (trg_max - trg_min)
        
            output = output.contiguous().view(-1)  # 展平除了最后一个维度以外的所有维度
            normalized_trg = normalized_trg.contiguous().view(-1).float()

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, normalized_trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 150
CLIP = 1

best_valid_loss = float('inf')


  
trg_min = float('inf')  # 初始化为正无穷大，用于寻找最小值  
trg_max = float('-inf') # 初始化为负无穷大，用于寻找最大值  
  
# 遍历整个训练数据集来计算trg的最小值和最大值  
for batch in train_iterator:  
    trg = batch.val  # 假设batch.val包含目标变量  
    trg_min = min(trg_min, trg.min().item())  # 更新最小值  
    trg_max = max(trg_max, trg.max().item())  # 更新最大值  
  
# 检查最小值和最大值是否相等（防止除以0错误）  
if trg_min == trg_max:  
    print("Warning: trg_min and trg_max are equal. Min-max scaling is not applicable.")  
    trg_scaled_mean = 0.0  # 或者其他适当的默认值  
    trg_scaled_std = 0.0   # 或者其他适当的默认值  
else:  
    # 进行min-max标准化（这里只计算了标准化后的均值，通常标准化后不需要再计算标准差）  
    trg_scaled_mean = (trg_min + trg_max) / 2.0  # 标准化后的均值理论上是(min+max)/2，但实际上这个值可能并不常用  
    # 遍历数据集再次计算标准化后的值（通常是在需要时才进行，这里只是示例）  
    # 注意：在实际应用中，您可能不需要再次遍历数据集，而是直接在后续处理中使用min和max进行标准化  
    trg_sum_scaled = 0.0  
    trg_count_scaled = 0  
    for batch in train_iterator:  
        trg = batch.val  
        trg_scaled = (trg - trg_min) / (trg_max - trg_min)  # 进行min-max标准化  
        trg_sum_scaled += trg_scaled.sum().item()  # 累加标准化后的值（示例）  
        trg_count_scaled += trg.size(0)  # 累加样本数（示例）  
  
    # 如果需要，可以计算标准化后值的“平均”（虽然这不是标准差，但可以用于了解数据的中心趋势）  
    # 注意：这个“平均”并不是真正的标准差，因为标准差衡量的是数据的离散程度，而不是中心趋势  
    trg_scaled_avg = trg_sum_scaled / trg_count_scaled if trg_count_scaled > 0 else 0.0  
  
    # 打印结果（根据需要调整）  
    print(f"trg_min={trg_min}, trg_max={trg_max}, trg_scaled_mean={trg_scaled_mean}, trg_scaled_avg={trg_scaled_avg}") 



train_losses = []  
test_losses = []
for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    #
    # 假设train_iterator已经加载了数据  
    # 在训练循环之前，我们可以检查trg的范围  
    

    
  
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, trg_min, trg_max)
    valid_loss = evaluate(model, test_iterator, criterion, trg_min, trg_max)
    test_loss = evaluate(model, test_iterator, criterion, trg_min, trg_max)

    train_losses.append(train_loss)  
    test_losses.append(test_loss) 
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    print(f'\t Test Loss: {test_loss:.3f}')
plt.figure(figsize=(10, 5))  
plt.plot(train_losses, label='Training Loss')  
plt.plot(test_losses, label='Validation Loss')  
plt.title('Training and Validation Loss')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.legend()  
plt.show()
model.load_state_dict(torch.load('tut6-model.pt'))
#test_loss = evaluate(model, test_iterator, criterion)
#print(f'| Test Loss: {test_loss:.3f}')


#def translate_sentence(sentence, src_field, trg_field, model, device, max_len=501):
#     model.eval()
#     output_length = 501
#     nlp = spacy.load('de_core_news_sm')
#     if isinstance(sentence, str):
#
#         tokens = tokenize_enm(sentence)
#     else:
#         tokens = [token.lower() for token in sentence]
#
#     # sentence = nlp(sentence)
#     # with sentence.retoknize() as retokenizer:
#     #     retokenizer.merge(sentence[1:-1])
#     
#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#
#     src_indexes = [src_field.vocab.stoi[token] for token in tokens]
#    
#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
#     print("src_tensor = {}".format(src_tensor))
#     
#     src_mask = model.make_src_mask(src_tensor)
#     print("src_mask = {}".format(src_mask))
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)
#     
#     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
#
#    
#
#     with torch.no_grad():
#             
#       output = model(enc_src)
#       
#       print(output)
#    
#
#  
#
#
#     return output

def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
     assert n_rows * n_cols == n_heads

     fig = plt.figure(figsize=(15, 25))

     for i in range(n_heads):
         ax = fig.add_subplot(n_rows, n_cols, i + 1)

         _attention = attention.squeeze(0)[i].cpu().detach().numpy()

         cax = ax.matshow(_attention, cmap='bone')

         ax.tick_params(labelsize=12)
         ax.set_xticklabels([''] + ['<start>'] + [t.lower() for t in sentence] + ['<end>'],
                            rotation=45)
         ax.set_yticklabels([''] + translation)

         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

     plt.show()
     plt.close()

def predict(sentence, src_field, model, device, trg_min, trg_max):    
    model.eval()    

    tokens = src_field.tokenize(str(sentence))    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]    
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)    
        
    src_mask = (src_tensor != src_field.vocab.stoi[src_field.pad_token]).unsqueeze(1).unsqueeze(2)    
        
    with torch.no_grad():    
        prediction = model(src_tensor)    
        
    # 反归一化预测结果  
    denormalized_prediction = (prediction * (trg_max - trg_min) / 30) + trg_min  
    print("init value=", denormalized_prediction)
    denormalized_prediction = denormalized_prediction.round().to(torch.int)  
    return denormalized_prediction.squeeze().cpu().numpy()
  
  
  


#example_idx = 100#这里数字的含义是预测网络的内容
fre_axis = []
for i in range(499):#500个频点
    fre_axis.append(30 + i * 2)
for example_idx in range(2000):  # 假设我们只预测前2000个样本  
    src = vars(test_set.examples[example_idx])['nets']  # 源文本  
    trg = vars(test_set.examples[example_idx])['val']  # 真实目标值（仅用于比较或展示）  
    
    # 使用模型进行预测  
    prediction = predict(src, SRC, model, device, trg_min, trg_max)  
  
    # 将预测结果转换为适合绘图的格式（如果需要的话）  
    # 例如，如果prediction是一个连续值数组，你可能不需要进行转换  
  
    # 可选：打印预测结果和真实目标值进行比较  
    print(f'Predicted values: {prediction}')  
    print(f'True tensors: {test_set.examples[example_idx]}')
    print(f'True values: {trg}')  # 假设trg是一个由空格分隔的字符串  
    print([TRG.vocab.itos[token] for token in prediction[0]])
    # 绘图比较预测结果和真实值  
    fre_axis = [30 + i * 2 for i in range(len(prediction))]  # 假设频率轴从30开始，步长为2  
    plt.figure()  
    plt.plot(fre_axis, prediction, label='Predicted RE', color='red')  
    plt.plot(fre_axis, list(map(float, trg)), label='True RE', color='blue')  
    plt.legend()  
    plt.title("Radiated Emission Value Prediction")  
    plt.xlabel("Frequency/MHz")  
    plt.ylabel("RE value V/m")  
    plt.savefig(f'D:/LCY/simulation/predicted_pictures/plot_{example_idx}.png')  # 保存具有唯一文件名的图像  
    plt.close()
#display_attention(src, translation, attention)
