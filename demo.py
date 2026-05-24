
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from transformer import Transformer, ModuleConfig


# device = 'cpu'
# device = 'cuda'
device = 'mps'  # macbook的GPU

# transformer epochs
epochs = 100

######
# DATA
######

# S: Symbol that shows starting of decoding input <BOS>
# E: Symbol that shows starting of decoding output <EOS>
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps <PAD>

sentences = [
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
]

src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
             '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
             'friend': 5, 'zero': 6, 'girl': 7,  'boy': 8, 'S': 9, 'E': 10, '.': 11}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
trg_vocab_size = len(tgt_vocab)


def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
 
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        #[[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.extend(enc_input)
        #[[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input)
        #[[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(
    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

#######
# MODEL
#######
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    enc_config=ModuleConfig(
        pad_idx=src_vocab['P'],
    ),
    dec_config=ModuleConfig(
        pad_idx=tgt_vocab['P'],
        sos_idx=tgt_vocab['S'],
        eos_idx=tgt_vocab['E']
    )
).to(device)

# 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['P'])
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  # 用adam的话效果不好

# =======================================================
print("开始训练Transformer模型...")

def train(model: Transformer):
    model.train()

    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            """
            enc_inputs: [bsz, src_len]
            dec_inputs: [bsz, tgt_len]
            dec_outputs: [bsz, tgt_len]
            """
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
                device), dec_inputs.to(device), dec_outputs.to(device)
            # outputs: [bsz * tgt_len, tgt_vocab_size]
            logits = model(enc_inputs, dec_inputs)
            # logits: [bsz, tgt_len, tgt_vocab_size]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),   # [B*T, V]
                dec_outputs.reshape(-1),                # [B*T]
            )
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

model = train(model)

print("训练完成！")

print(model)


# =======================================================

print("开始测试Transformer模型...")

def generate(model: Transformer, prompt: str = '我 有 一 个 男 朋 友'):
    enc_input = torch.LongTensor([src_vocab[n] for n in prompt.split()]).to(device)

    print("SRC:", [src_idx2word[int(i)] for i in enc_input])

    print("GEN:", end=" ")
    for token in model.generate(enc_input.view(1, -1)):
        print(idx2word[token], end=" ")
    print()

generate(model, prompt='我 有 一 个 男 朋 友')
generate(model, prompt='我 有 零 个 男 朋 友')
generate(model, prompt='我 有 一 个 一 朋 友')
