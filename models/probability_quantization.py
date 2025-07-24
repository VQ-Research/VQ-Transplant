import torch
from torch.nn import functional as F

def quantize_batch_probs(p, L):
    """
    Quantize multiple sets of probability distributions into fractions with a denominator of L.
    
    Parameters:
        p: A floating-point tensor of shape (n, K), representing n sets of probability distributions.
          Each row represents a set of probability distributions, and the sum of each row is 1.
        L: the denominator is L.
    
    Returns:
        p_quant: A floating-point tensor of shape (n, K), quantized to 0/L, 1/L, ..., L/L.
    """
    n, K = p.shape
    
    # Step 1: Perform initial quantization and calculate the errors
    p_int = torch.round(L * p)
    S = p_int.sum(dim=1)
    delta = L - S  # Adjustment needed per row (shape: [n])
    
    # Step 2: Calculate rounding errors
    p_error_up = p_int - L * p   # >0 means overestimation
    p_error_down = L * p - p_int  # >0 means underestimation
    
    # Step 3: Create masks for valid adjustments
    can_increase = (p_int < L)  # Positions that can be increased
    can_decrease = (p_int > 0)   # Positions that can be decreased
    
    # Step 4: Build priority matrices
    inc_priority = torch.where(can_increase, p_error_down, -torch.inf)
    dec_priority = torch.where(can_decrease, p_error_up, -torch.inf)
    
    # Step 5: Sort priorities (descending order)
    # Use stable sort for deterministic results
    inc_sorted_idx = torch.argsort(inc_priority, dim=1, descending=True, stable=True)
    dec_sorted_idx = torch.argsort(dec_priority, dim=1, descending=True, stable=True)
    
    # Step 6: Vectorized adjustment using scatter
    indices = torch.arange(K, device=p.device).expand(n, K)  # [n, K]
    
    # 6.1 Adjustment for positive delta (increase)
    inc_counts = delta.clamp(min=0, max=K).long()  # Ensure non-negative
    inc_mask = indices < inc_counts.unsqueeze(1)  # True for positions to increase
    adjust_inc = torch.zeros((n, K), dtype=torch.int32, device=p.device)
    adjust_inc = adjust_inc.scatter(1, inc_sorted_idx, inc_mask.to(torch.int32))
    
    # 6.2 Adjustment for negative delta (decrease)
    dec_counts = (-delta).clamp(min=0, max=K).long()  # Absolute value
    dec_mask = indices < dec_counts.unsqueeze(1)  # True for positions to decrease
    adjust_dec = torch.zeros((n, K), dtype=torch.int32, device=p.device)
    adjust_dec = adjust_dec.scatter(1, dec_sorted_idx, dec_mask.to(torch.int32))
    adjust_dec = -adjust_dec  # Convert to negative adjustments
    
    # 6.3 Combine adjustments
    adjustment = adjust_inc + adjust_dec
    
    # Step 7: Apply adjustments and normalize
    p_int_adjusted = p_int + adjustment
    p_quant = p_int_adjusted / L
    
    return p_quant