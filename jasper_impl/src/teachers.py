import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class DualTeacherWrapper(nn.Module):
    def __init__(self, 
                 stella_path="dunzhang/stella_en_1.5B_v5", 
                 nv_path="nvidia/NV-Embed-v2",
                 load_real_weights=False): # Default to False to prevent accidental 14GB download
        super().__init__()
        
        self.load_real = load_real_weights
        
        if load_real_weights:
            print(f"Loading Teacher 1: {stella_path}...")
            self.stella = AutoModel.from_pretrained(stella_path, trust_remote_code=True)
            
            print(f"Loading Teacher 2: {nv_path}...")
            self.nv = AutoModel.from_pretrained(nv_path, trust_remote_code=True)
        else:
            print("Initializing Dummy Teachers (Configs only)...")
            # Dummy Stella (Optimized for speed)
            cfg_stella = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
            cfg_stella.hidden_size = 4096 
            cfg_stella.num_hidden_layers = 2
            cfg_stella.vocab_size = 1000
            cfg_stella.num_attention_heads = 32 # 4096 / 32 = 128
            cfg_stella.intermediate_size = 11008
            self.stella = AutoModel.from_config(cfg_stella, trust_remote_code=True)
            
            # Dummy NV-Embed (Optimized for speed)
            cfg_nv = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct", trust_remote_code=True) 
            cfg_nv.hidden_size = 4096 
            cfg_nv.num_hidden_layers = 2
            cfg_nv.vocab_size = 1000
            cfg_nv.num_attention_heads = 32
            cfg_nv.intermediate_size = 11008
            self.nv = AutoModel.from_config(cfg_nv, trust_remote_code=True)

        # Freeze teachers
        self.stella.eval()
        self.nv.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None):
        # In a real scenario, you might need different tokenizers for different teachers.
        # Here we assume the input_ids are compatible or re-tokenized externally.
        
        with torch.no_grad():
            # 1. Get Teacher 1 Output (Stella)
            # Stella typically uses mean pooling or last token
            out1 = self.stella(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Mean Pool
            mask_expanded = attention_mask.unsqueeze(-1).expand(out1.size()).float()
            sum_embeddings = torch.sum(out1 * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            vec1 = sum_embeddings / sum_mask
            
            # 2. Get Teacher 2 Output (NV-Embed)
            out2 = self.nv(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Mean Pool
            mask_expanded = attention_mask.unsqueeze(-1).expand(out2.size()).float()
            sum_embeddings = torch.sum(out2 * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            vec2 = sum_embeddings / sum_mask
            
            # 3. Process per Paper Eq: t_x = Norm( Concat( Norm(t1), Norm(t2) ) )
            
            # Normalize individually
            vec1 = F.normalize(vec1, p=2, dim=1)
            vec2 = F.normalize(vec2, p=2, dim=1)
            
            # Concatenate
            vec_cat = torch.cat([vec1, vec2], dim=1)
            
            # Normalize Combined
            vec_final = F.normalize(vec_cat, p=2, dim=1)
            
            return vec_final
