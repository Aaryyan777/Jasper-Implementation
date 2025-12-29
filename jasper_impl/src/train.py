import torch
from torch.optim import AdamW
from model import JasperModel
from losses import DistillationLoss
from utils import get_dummy_batch
from teachers import DualTeacherWrapper

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad

def train_stage(stage_name, model, teacher, optimizer, loss_fn, steps=5):
    print(f"\n=== Starting {stage_name} ===")
    print(f"Trainable Parameters: {count_parameters(model)}")
    
    model.train()
    teacher.eval() # Teachers are always frozen
    
    for step in range(steps):
        batch = get_dummy_batch()
        
        # Move to device if available
        # device = next(model.parameters()).device
        # ... (skipping device move for simplicity in CPU dummy run)
        
        optimizer.zero_grad()
        
        # Forward Pass
        # Stage 4 is multimodal, others are text-only distillation
        if stage_name == "Stage 4":
            # Image input -> Student
            student_outputs = model(pixel_values=batch["pixel_values"], modality="image")
            
            # Caption input -> Teacher (Self-Distillation)
            # Paper: "aligned vectors from an earlier stage ... serve as teacher vectors"
            # Specifically: "caption's vector representation serves as the teacher vector"
            # Here we use the text encoder (which is frozen and trained in S1-S3) as the teacher.
            with torch.no_grad():
                teacher_outputs_dict = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], modality="text")
                # Use the FC1 (12288) or highest dim vector as teacher?
                # Paper says: "All fully connected layers ... employed to generate multiple pairs... calculate three losses... averaged"
                # So we align Image_FC_i with Text_FC_i for i in [1,2,3,4].
                pass
        else:
            student_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], modality="text")
            
            # Get Teacher Targets
            with torch.no_grad():
                teacher_target = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        # Loss Calculation
        total_loss = 0
        
        if stage_name == "Stage 4":
            # Average loss across all FC heads
            losses_log = {}
            for name, student_vec in student_outputs.items():
                teacher_vec = teacher_outputs_dict[name] # Text is teacher
                # Note: Dimensions match for each head.
                l, details = loss_fn(student_vec, teacher_vec)
                total_loss += l
                for k, v in details.items():
                    losses_log[f"{name}_{k}"] = v.item()
            total_loss = total_loss / len(student_outputs)
            
        elif stage_name == "Stage 3":
            # Train all FCs.
            # "For the three FC layers ... L_cosine is omitted" (FC2, FC3, FC4)
            # "To ensure accuracy ... FC1 ... continue to be trained using all three loss functions"
            
            # FC1 (12288) -> Matches Teacher
            l1, d1 = loss_fn(student_outputs["fc1"], teacher_target)
            total_loss += l1
            
            # FC2, FC3, FC4 -> Dimension Mismatch with Teacher (12288)
            # We can't use L_cosine or L_sim directly if dimensions differ for L_sim?
            # Eq 2 L_sim uses Matmul(S, S.T) so dimensions cancel out! (BxB)
            # L_resim uses pairwise scores (BxB), so dimensions cancel out!
            # L_cosine uses element-wise, so dimensions MUST match.
            
            # So for FC2, FC3, FC4: Use L_sim and L_resim only.
            for name in ["fc2", "fc3", "fc4"]:
                # Manually calc loss components
                # Reuse loss_fn logic but skip cosine
                # Or pass a flag. Let's manually call for clarity or assume loss_fn handles dim mismatch (it does check shape).
                # Our loss_fn returns 0 for cosine if shape mismatch.
                l, d = loss_fn(student_outputs[name], teacher_target)
                total_loss += l
                
        else: # Stage 1 & 2
            # "student model's vector dimension is adjusted to 12288"
            # Only train FC1 output
            l, details = loss_fn(student_outputs["fc1"], teacher_target)
            total_loss += l
        
        total_loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {total_loss.item():.4f}")

def main():
    print("Initializing Jasper Model...")
    model = JasperModel()
    teacher = DualTeacherWrapper(load_real_weights=False)
    loss_fn = DistillationLoss()
    
    # --- Stage 1: Distillation (Train FC1 only) ---
    # "In stage 1, only the fully connected layer (FC1) is trained"
    set_requires_grad(model, False)
    set_requires_grad(model.fc_layers.fc1, True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    train_stage("Stage 1", model, teacher, optimizer, loss_fn)
    
    # --- Stage 2: Distillation (Train FC1 + Last 3 Layers) ---
    # "FC1 and the last three encoder layers ... are trained"
    # Unfreeze last 3 layers of text encoder
    # Assuming HuggingFace structure: model.text_encoder.layers (or encoder.layer)
    # Check structure via print if unsure, usually 'model.layers' for Qwen/Llama

    # Let's try generic approach or assume 'layers' attribute
    try:
        if hasattr(model.text_encoder, 'layers'):
             layers = model.text_encoder.layers
        elif hasattr(model.text_encoder, 'encoder'):
             layers = model.text_encoder.encoder.layer # BERT style
        elif hasattr(model.text_encoder, 'model'): # Qwen style sometimes wrapped
             layers = model.text_encoder.model.layers
        else:
             print("Warning: Could not identify layers to unfreeze. Skipping layer unfreeze.")
             layers = []
             
        if len(layers) > 0:
            for layer in layers[-3:]:
                set_requires_grad(layer, True)
    except Exception as e:
        print(f"Error unfreezing layers: {e}")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-5)
    train_stage("Stage 2", model, teacher, optimizer, loss_fn)

    # --- Stage 3: Dimension Reduction (Train All) ---
    # "all parameters of the student model are trained"
    set_requires_grad(model, True)
    # Freeze vision just in case (though it's not used in this path)
    set_requires_grad(model.vision_encoder, False)
    set_requires_grad(model.vision_projector, False)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=7e-5)
    train_stage("Stage 3", model, teacher, optimizer, loss_fn)
    
    # --- Stage 4: Multimodal (Train Vision Only) ---
    # "focusing exclusively on training the visual encoder while keeping the other components frozen"
    set_requires_grad(model, False)
    set_requires_grad(model.vision_encoder, True)
    set_requires_grad(model.vision_projector, True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    train_stage("Stage 4", model, teacher, optimizer, loss_fn)

if __name__ == "__main__":
    main()
