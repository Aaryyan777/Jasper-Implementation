import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, SiglipVisionModel

class JasperModel(nn.Module):
    def __init__(
        self,
        text_model_name_or_path="dunzhang/stella_en_1.5B_v5", # Placeholder
        vision_model_name_or_path="google/siglip-so400m-patch14-384",
        student_embedding_dim=1536, # Inferred from paper "FC3 with a shape of (1536, 512)"
        target_dims={
            "fc1": 12288,
            "fc2": 1024,
            "fc3": 512,
            "fc4": 256
        }
    ):
        super().__init__()
        
        # 1. Text Encoder (Student LLM)
        # Use config-only initialization to avoid downloading 6GB+ weights for logic verification.
        try:
            print(f"Loading config for {text_model_name_or_path}...")
            config = AutoConfig.from_pretrained(text_model_name_or_path, trust_remote_code=True)
            self.text_encoder = AutoModel.from_config(config, trust_remote_code=True)
            print("Initialized Text Encoder with random weights (no download).")
        except Exception as e:
            print(f"Warning: Could not load config for {text_model_name_or_path}. Using dummy Qwen2 config. Error: {e}")
            config = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B", trust_remote_code=True)
            config.hidden_size = student_embedding_dim # Override for testing if needed
            self.text_encoder = AutoModel.from_config(config, trust_remote_code=True)

        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        # 2. Vision Encoder
        try:
            print(f"Loading config for {vision_model_name_or_path}...")
            config = AutoConfig.from_pretrained(vision_model_name_or_path)
            if hasattr(config, "vision_config"):
                config = config.vision_config
            self.vision_encoder = SiglipVisionModel(config)
            print("Initialized Vision Encoder with random weights (no download).")
        except Exception as e:
            print(f"Warning: Could not load config for {vision_model_name_or_path}. Using dummy config. Error: {e}")
            from transformers import SiglipVisionConfig
            config = SiglipVisionConfig() # Default params
            self.vision_encoder = SiglipVisionModel(config)
            
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        # 3. Vision Pooler / Adapter
        # Figure 1 shows AvgPool2d. 
        # We need to adapt vision tokens to text encoder input dimension.
        # Assuming we flatten the spatial dimensions after pooling or before.
        # Paper says: "maps vision token embeddings to the same dimension as the language model's input textual embeddings"
        self.vision_projector = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
        
        # 4. FC Layers (Projection Heads)
        self.fc_layers = nn.ModuleDict()
        for name, dim in target_dims.items():
            self.fc_layers[name] = nn.Linear(self.text_hidden_size, dim)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, modality="text"):
        """
        Args:
            input_ids: Text input
            pixel_values: Image input
            modality: "text" or "image" (for Stage 4) or "multimodal" if integrated.
                      Paper implies separate processing for distillation stages.
        """
        
        if modality == "image" and pixel_values is not None:
            # Vision Path
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state # [B, Seq, Dim]
            
            # Simple average pooling over spatial tokens (like AvgPool2d)
            # Siglip so400m patch14 384 -> 27x27 tokens = 729 tokens.
            # Paper mentions "reducing the length of visual token sequences". 
            # If we strictly follow "AvgPool2d", we might reshape and pool.
            # Let's assume global average pooling for simplicity unless "tokens" are needed for LLM input.
            # Wait, "maps vision token embeddings... while reducing the length". 
            # If it feeds into the LLM, it keeps sequence structure. 
            # But Stage 4 says "image's vector representation acts as the student vector".
            # This implies the final output is a vector.
            # Let's see: Image -> Vision Enc -> Pool -> Project -> LLM -> Mean Pool -> Vector.
            
            # Project to LLM dim
            image_embeds = self.vision_projector(image_embeds) # [B, Seq, LLM_Dim]
            
            # Feed to LLM
            # We need to construct inputs_embeds
            inputs_embeds = image_embeds
            
            # Create attention mask for image tokens (all 1s)
            dummy_mask = torch.ones(inputs_embeds.shape[:-1], device=inputs_embeds.device, dtype=torch.long)
            
            outputs = self.text_encoder(inputs_embeds=inputs_embeds, attention_mask=dummy_mask)
            
        else:
            # Text Path
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean Pooling (Standard for embedding models)
        last_hidden_state = outputs.last_hidden_state
        if modality == "image":
             # Use the dummy mask created
             mask = dummy_mask
        else:
             mask = attention_mask
             
        # Mask out padding tokens
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask # [B, 1536]
        
        # Normalization (usually done before heads? or after? 
        # Equation 1 says s_x is "normalized vector representation".
        # Usually embedding models normalize at the very end.
        # But here we have FC layers. 
        # Paper: "The encoder-based language model that generates text embeddings through mean pooling."
        # Then "Several FC layers project the embeddings".
        # So Mean Pool -> FC -> Normalize.
        
        results = {}
        for name, layer in self.fc_layers.items():
            projected = layer(mean_embeddings)
            # Normalize
            normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
            results[name] = normalized
            
        return results

