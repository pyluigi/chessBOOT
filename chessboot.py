import berserk
import torch
from torch import nn
from torch import optim
import chess
import os
import pandas as pd

device = torch.device("cpu")

def board_to_tensor(board):
    piece_encoding = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }

    tensor = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            tensor[square] = piece_encoding[piece.symbol()]
        else:
            tensor[square] = 0

    return tensor.unsqueeze(0)

class BOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13,64)
        self.attention = nn.MultiheadAttention(embed_dim=64,num_heads=16)
        self.neurons = nn.Sequential(
            nn.Linear(4096,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(1, 0, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.neurons(x)
        return x
model = BOT().to(device)
model = torch.compile(model,mode="max-autotune",dynamic=False)
if os.path.exists("booty.pth"):
    file = torch.load("booty.pth",map_location=device,weights_only=True)
    model.load_state_dict(file)
model.train()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.MSELoss()
num_epochs = 1
df = pd.read_csv("lichess_db_puzzle.csv",nrows=1000)
df = df.sort_values(by="Rating", ascending=True)
t1 = torch.tensor([10.0], dtype=torch.float32, device=device)
t2 = torch.tensor([-10.0], dtype=torch.float32, device=device)
t3 = torch.tensor([0.0], dtype=torch.float32, device=device)
for i in range(num_epochs):
    total_loss = 0.0
    for puzzle in df.iloc:
     board = chess.Board(puzzle["FEN"])
     print(f"Rating: {puzzle['Rating']} elo.")
     n = 0
     for move in puzzle["Moves"].split():
         if n % 2 == 0:
             for movey in list(board.legal_moves):
                 if str(movey.uci()) == move:
                     b = True
                 else:
                     b = False
                 board.push(movey)
                 tensor = board_to_tensor(board).to(device)
                 evaling = model(tensor)
                 board.pop()
                 optimizer.zero_grad()
                 if b and board.turn == chess.WHITE:
                     loss = criterion(t1, evaling)
                 elif b and board.turn == chess.BLACK:
                     loss = criterion(t2, evaling)
                 else:
                     loss = criterion(t3, evaling)
                 total_loss += loss.item()
                 loss.backward()
                 optimizer.step()
         n += 1
         board.push_uci(move)
    print(f"Epoch [{i+1}/{num_epochs}], Loss: {total_loss}.")
    torch.save(model.state_dict(),"booty.pth")

