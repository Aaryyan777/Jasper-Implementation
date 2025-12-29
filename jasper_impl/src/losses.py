import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, lambda1=10, lambda2=200, lambda3=20, margin=0.015):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, student_embeds, teacher_embeds):
        """
        Args:
            student_embeds: [B, D_s] (Normalized)
            teacher_embeds: [B, D_t] (Normalized)
        """
        
        # 1. Cosine Loss (Eq 1)
        # L_cosine = Sum(1 - s_x . t_x)
        # Note: If dimensions mismatch (e.g. Stage 1/2 vs Stage 3 reduced), we might skip or project.
        # Paper says: "In stage 3... since dimensions ... do not align ... L_cosine is omitted".
        # So we handle that check outside or here.
        
        if student_embeds.shape[1] == teacher_embeds.shape[1]:
            cosine_sim = torch.sum(student_embeds * teacher_embeds, dim=1)
            l_cosine = torch.sum(1 - cosine_sim) / student_embeds.size(0) # Mean over batch
        else:
            l_cosine = torch.tensor(0.0, device=student_embeds.device)

        # 2. Similarity Loss (Eq 2)
        # L_sim = MSE(S @ S.T, T @ T.T)
        sim_matrix_student = torch.matmul(student_embeds, student_embeds.T)
        sim_matrix_teacher = torch.matmul(teacher_embeds, teacher_embeds.T)
        
        l_sim = self.mse(sim_matrix_student, sim_matrix_teacher)

        # 3. Relative Similarity Distillation Loss (Eq 3)
        # Inspired by CoSENT.
        # Paper: "ensure that the similarity between positive pairs exceeds that between negative pairs"
        # identifying positive/negative via teacher scores.
        # Simplified implementation: For each row, pairs (i, j) with high T_ij should have high S_ij.
        # Implementation of strict Eq 3 (pairwise ranking of pairs):
        # pair_diff_teacher = T_ij - T_mn
        # pair_diff_student = S_ij - S_mn
        # If T_diff > 0, we want S_diff > 0 (specifically S_ij > S_mn + margin)
        # This is O(B^4).
        
        # Optimizing:
        # Instead of all pairs of pairs, let's consider for each anchor i:
        # P_i = {j | T_ij is high}, N_i = {k | T_ik is low}
        # Loss = sum_{j in P, k in N} max(0, S_ik - S_ij + margin)
        # We can dynamically define P and N using Teacher Matrix.
        # For each i, let j be the index with max T_ij (j!=i) -> Positive
        # Let k be indices where T_ik < T_ij - threshold? Or just all other k.
        
        # Efficient batched version:
        # 1. Flatten matrices to vectors of pairs (B*B pairs).
        # 2. Filter out self-pairs (diagonal).
        # 3. Use teacher scores to determine valid pairs of pairs? Too big.
        
        # Let's try the CoSENT implementation style which is efficient:
        # CoSENT: logsumexp( lambda * (sim_neg - sim_pos) )
        # Here we have a margin loss.
        # Let's do: For each anchor i:
        # Find j = argmax(T[i, :]) (excluding i)
        # Find all k where T[i, k] < T[i, j]
        # Loss += max(0, S[i, k] - S[i, j] + margin)
        
        batch_size = student_embeds.size(0)
        
        # Mask diagonal
        mask = torch.eye(batch_size, device=student_embeds.device).bool()
        sim_matrix_teacher_masked = sim_matrix_teacher.masked_fill(mask, -1e9) # Exclude self from max
        
        # Find hardest positive (or just max teacher sim)
        # In distillation, the teacher's structure is the ground truth.
        # So the "positive" for i is the j that the teacher thinks is most similar.
        pos_indices = sim_matrix_teacher_masked.argmax(dim=1) # [B]
        
        # Gather S_pos and S_neg
        s_pos = sim_matrix_student[torch.arange(batch_size), pos_indices] # [B]
        s_pos = s_pos.unsqueeze(1) # [B, 1]
        
        # We want S_ik < S_ij - margin for all k != j (and != i)
        # Loss = ReLU(S_ik - (S_ij - margin)) = ReLU(S_ik - S_ij + margin)
        
        # Create mask for k (excluding i and j)
        mask_k = torch.ones_like(sim_matrix_student).bool()
        mask_k[torch.arange(batch_size), pos_indices] = False # Exclude pos
        mask_k = mask_k.masked_fill(torch.eye(batch_size, device=student_embeds.device).bool(), False) # Exclude self
        
        # Calculate loss only on valid negatives
        # Paper implies utilizing "all potential positive and negative pairs".
        # But doing just the hardest positive vs all negatives is a strong proxy.
        # Or we can do: T_ij > T_ik implies S_ij > S_ik.
        # Let's just use all valid comparisons where T_ij > T_ik.
        # Weight by T_ij - T_ik? Eq 3 doesn't say so.
        
        # Let's stick to the "Best Positive vs All Negatives" approximation for efficiency.
        # It's O(B^2).
        
        diff = sim_matrix_student - s_pos + self.margin # [B, B]
        # Only consider where T_ik < T_ij
        t_pos = sim_matrix_teacher[torch.arange(batch_size), pos_indices].unsqueeze(1)
        valid_triplets = sim_matrix_teacher < t_pos # [B, B]
        
        losses = F.relu(diff)
        losses = losses * valid_triplets.float() * mask_k.float()
        
        l_resim = losses.sum() / (valid_triplets.sum() + 1e-9)
        
        total_loss = self.lambda1 * l_cosine + self.lambda2 * l_sim + self.lambda3 * l_resim
        return total_loss, {"l_cosine": l_cosine, "l_sim": l_sim, "l_resim": l_resim}

