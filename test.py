import torch
from layers import EncoderDecoderModel
from data_reverse import *

# Test EncoderDecoderModel on "reverse" task
# See if it generalizes to numbers larger than it trains on
vocab_size = 200
seq_len = 100
train_dataset = ReverseDataset(seq_len, vocab_size // 2, 100000)
test_dataset = ReverseDataset(seq_len, vocab_size, 10000)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

model = EncoderDecoderModel(
    vocab_size = vocab_size, 
    enc_seq_len = seq_len,
    n_enc_layers = 12, 
    dec_seq_len = seq_len + 1, 
    n_dec_layers = 12,                 
    embed_dim = 128, 
    d_qkv = 8, 
    n_heads = 16, 
    d_ff = 512, 
    dropout = 0
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1): # Don't repeat data
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        X, Y = batch[0].to(device), batch[1].to(device)
        logits = model(X, Y) 
        loss = torch.nn.functional.cross_entropy(logits[:, :-1, :].flatten(0, 1), Y[:, 1:].flatten())
        loss.backward()
        if i % 10 == 0:
            print(f"Step {i}: loss = {loss.item()}")
        if i % 1000 == 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f"model-{i}.pt")
        optimizer.step()




