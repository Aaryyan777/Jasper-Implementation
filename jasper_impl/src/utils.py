import torch
import torch.nn.functional as F

class MockTeacher(torch.nn.Module):
    def __init__(self, dim1=4096, dim2=8192):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        # Simulate teacher outputs
        t1 = torch.randn(batch_size, self.dim1, device=input_ids.device)
        t2 = torch.randn(batch_size, self.dim2, device=input_ids.device)
        
        # Normalize individual teacher outputs
        t1 = F.normalize(t1, p=2, dim=1)
        t2 = F.normalize(t2, p=2, dim=1)
        
        # Concatenate
        t_concat = torch.cat([t1, t2], dim=1)
        
        # Normalize combined
        t_final = F.normalize(t_concat, p=2, dim=1)
        
        return t_final

def get_dummy_batch(batch_size=4, seq_len=64, img_size=384):
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values
    }
