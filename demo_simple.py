
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.simple import *
from transformer import Transformer, ModuleConfig


# device = 'cpu'
# device = 'cuda'
device = 'mps'  # macbook的GPU

# transformer epochs
epochs = 100

######
# DATA
######
loader = DataLoader(
    tokenized_datasets,
    batch_size=2,
    shuffle=True
)

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


# =======================================================
def train(model: Transformer):
    print("开始训练Transformer模型...")

    # 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['P'])
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  # 用adam的话效果不好

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
    print("训练完成！")

    return model

model = train(model)
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
