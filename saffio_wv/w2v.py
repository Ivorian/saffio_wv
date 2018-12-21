import torch

class W2V(torch.nn.Module):
    def __init__(self, tok_num, code_num):
        super(W2V, self).__init__()
        
        self.embd = torch.nn.Embedding(
            num_embeddings=tok_num, embedding_dim=code_num
        )
        self.fc = torch.nn.Sequential(
        torch.nn.Linear(code_num, tok_num, bias=False)
        )

        self.lo = torch.nn.NLLLoss()
        
    def forward(self, x):
        em = self.embd(x)
        de = self.fc(em)
        log_de = torch.log_softmax(de, 1)
        return log_de
    
    def loss(self, log_act, expected):
        return self.lo(log_act, expected)
    
    def encode(self, x):
        return self.embd(x)