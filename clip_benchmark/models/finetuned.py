import clip 
import torch
import os
import open_clip
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from tqdm import tqdm
import json
import clip 
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# def load_clip_vit_b_32(model_name=None, pretrained=None, cache_dir=None, device="cuda"):
#     model, preprocess = clip.load("ViT-B/16", device=device)
#     def tokenizer(texts):
#         return clip.tokenize(texts, truncate=True).to(device)
#     i = 4
#     # path = f'/home/aarish/VLM-superstition-analysis/models/clip_fold5.pt'
#     # path = f'/home/aarish/VLM-superstition-analysis/models/clip_fold1_best_16.pt'
#     # path = f'/home/aarish/VLM-superstition-analysis/models/clip_fold1_best_0.pt'
#     # path = f'/home/aarish/VLM-superstition-analysis/models/clip_fold1_best.pt'
#     # path = f'/home/aarish/VLM-superstition-analysis/results/clip_best.pth'
#     # print(f"loading finetuned model from {path}")
#     # path = '/home/aarish/VLM-superstition-analysis/models/clip_fold6_best_7.pt'
#     path = '/home/aarish/VLM-superstition-analysis/models/aarish6_best_29.pt'
#     print("loading model aarish shah mohsin")
#     model.load_state_dict(torch.load(path)['model'])
#     return model, preprocess, tokenizer


CONFIG = {
    # Model settings
    # "model_name": "ViT-B-16-SigLIP",
    # "pretrained_dataset": "webli",

    # "model_name": "ViT-B-32",
    # "pretrained_dataset": "laion2b_s34b_b79k",

    "model_name": "ViT-B-16", 
    "pretrained_dataset": "laion2b_s34b_b88k",

    
    # File paths
    "save_path": "./models_final_debias",
    "faces_csv_path": "/home/aarish/VLM-superstition-analysis/faces_test.csv",

    # --- KEY TUNING PARAMETERS ---
    # Debiasing strength - controls how much to rotate embeddings
    "DEBIAS_STRENGTH": 0.3,  # 0.0 = full debias, 1.0 = no debias (try 0.1-0.5)
    
    # Boost to compensate for overall similarity drop
    "SIMILARITY_COMPENSATION": True,  # Keeps overall mean similarity constant
    
    # Fairness threshold: groups below this similarity get boosted
    "FAIRNESS_THRESHOLD": 0.1,  # Adjust based on your baseline analysis

    # The precise bias definition
    "misclassification_map": {
        'andaman_and_nicobar_islands': 'African',
        'arunachal_pradesh': 'Chinese',
        'assam': 'Bangladeshi',
        'jammu_and_kashmir': 'Pakistani',
        'ladakh': 'Tibetan',
        'manipur': 'Chinese',
        'meghalaya': 'Bangladeshi',
        'mizoram': 'Chinese',
        'nagaland': 'Chinese',
        'sikkim': 'Chinese',
        'tripura': 'Chinese'
    }
}

# ==============================================================================
# 2. DEBIASING MODULES
# ==============================================================================

class NullspaceProjection(torch.nn.Module):
    """Applies Iterative Nullspace Projection to remove specified bias directions."""
    def __init__(self, dim, num_iterations=5):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations
        self.projection_matrices = torch.nn.ParameterList()
        self.fitted = False

    def fit(self, embeddings, labels, device='cpu'):
        print(f"🔍 Computing Directional Debiasing projection...")
        current_embeddings = embeddings.copy()
        projection_matrices = []
        for iteration in range(self.num_iterations):
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(current_embeddings)
            clf = LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, n_jobs=-1)
            clf.fit(scaled_embeddings, labels)
            train_acc = clf.score(scaled_embeddings, labels)
            print(f"  Iteration {iteration + 1}, Classifier Accuracy: {train_acc:.4f}")
            if train_acc < 0.55:
                print(f"  ✅ Bias directions removed (accuracy near random).")
                break
            w = clf.coef_[0]
            norm_w = np.linalg.norm(w)
            if np.isclose(norm_w, 0): break
            w = w / norm_w
            P = np.eye(self.dim) - np.outer(w, w)
            projection_matrices.append(torch.FloatTensor(P))
            current_embeddings = current_embeddings @ P
        for P in projection_matrices:
            self.projection_matrices.append(torch.nn.Parameter(P.to(device), requires_grad=False))
        self.fitted = True
        print(f"✅ Directional debiasing fitted with {len(self.projection_matrices)} matrices.")

    def forward(self, x):
        if not self.fitted: return x
        for P in self.projection_matrices:
            x = torch.matmul(x, P)
        return x

class CompensatedDebiasedSigLIP(torch.nn.Module):
    """
    Applies debiasing with automatic compensation to maintain overall similarity levels.
    """
    def __init__(self, base_model, projection_module=None, debias_strength=0.3, 
                 compensate_similarity=True):
        super().__init__()
        self.base_model = base_model
        self.projection = projection_module
        self.debias_strength = debias_strength
        self.compensate_similarity = compensate_similarity
        self.compensation_vector = None
        for param in self.base_model.parameters():
            param.requires_grad = False

    def compute_compensation(self, target_text_embedding, sample_image_features, device):
        """
        Compute a compensation vector that restores the overall similarity level.
        This is done by analyzing how the projection affects a sample of images.
        """
        if not self.compensate_similarity or self.projection is None:
            return
        
        print(f"🔧 Computing similarity compensation vector...")
        
        with torch.no_grad():
            # Get original and debiased embeddings for sample
            original_features = sample_image_features
            debiased_features = self._apply_debiasing(original_features, compensate=False)
            
            # Normalize for comparison
            original_norm = F.normalize(original_features, dim=-1)
            debiased_norm = F.normalize(debiased_features, dim=-1)
            target_norm = F.normalize(target_text_embedding, dim=-1)
            
            # Compute similarity drop
            original_sim = (original_norm @ target_norm.T).mean()
            debiased_sim = (debiased_norm @ target_norm.T).mean()
            
            print(f"  Original mean similarity: {original_sim:.4f}")
            print(f"  Debiased mean similarity: {debiased_sim:.4f}")
            print(f"  Similarity drop: {original_sim - debiased_sim:.4f}")
            
            # Compute the direction that increases similarity
            # This is the direction from debiased back toward the target
            similarity_direction = target_norm.mean(dim=0, keepdim=True)
            similarity_direction = F.normalize(similarity_direction, dim=-1)
            
            # Compute how much boost is needed
            # We want to add a component in the direction of the target
            boost_magnitude = (original_sim - debiased_sim).item() * 2.0  # Scale factor
            
            self.compensation_vector = (similarity_direction * boost_magnitude).squeeze(0)
            
            print(f"  ✅ Compensation magnitude: {boost_magnitude:.4f}")

    def _apply_debiasing(self, features, compensate=True):
        """
        Apply debiasing as a rotation that preserves vector norms.
        """
        if self.projection is None or self.debias_strength >= 1.0:
            return features
        
        # Get debiased direction
        debiased = self.projection(features)
        
        if self.debias_strength <= 0.0:
            # Full debiasing - but preserve norm
            original_norm = torch.norm(features, dim=-1, keepdim=True)
            debiased_norm = torch.norm(debiased, dim=-1, keepdim=True)
            debiased_norm = torch.clamp(debiased_norm, min=1e-8)
            result = debiased * (original_norm / debiased_norm)
        else:
            # Partial debiasing - interpolate on unit sphere
            features_normalized = F.normalize(features, dim=-1)
            debiased_normalized = F.normalize(debiased, dim=-1)
            
            # Spherical interpolation (slerp)
            dot_product = torch.sum(features_normalized * debiased_normalized, dim=-1, keepdim=True)
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            
            theta = torch.acos(dot_product)
            sin_theta = torch.sin(theta)
            
            safe_sin_theta = torch.where(sin_theta.abs() < 1e-6, 
                                          torch.ones_like(sin_theta), 
                                          sin_theta)
            
            weight_original = torch.sin(self.debias_strength * theta) / safe_sin_theta
            weight_debiased = torch.sin((1 - self.debias_strength) * theta) / safe_sin_theta
            
            interpolated = torch.where(
                sin_theta.abs() < 1e-6,
                features_normalized,
                weight_original * features_normalized + weight_debiased * debiased_normalized
            )
            
            # Scale back to original magnitude
            original_norm = torch.norm(features, dim=-1, keepdim=True)
            result = interpolated * original_norm
        
        # Apply compensation if available and requested
        if compensate and self.compensation_vector is not None:
            result = result + self.compensation_vector.unsqueeze(0)
        
        return result

    def encode_image(self, images):
        features = self.base_model.encode_image(images)
        return self._apply_debiasing(features, compensate=True)

    def encode_text(self, text):
        # Text embeddings stay unchanged
        return self.base_model.encode_text(text)


def load_nullspace_projection(path: str, device='cpu', dim=None, num_iterations=None):
    """
    Returns a NullspaceProjection instance with projection_matrices and fitted flag restored.
    If `dim` is None, tries to infer from saved meta or projection matrix shapes.
    """
    state = torch.load(path, map_location='cpu')
    pm_list = state.get("projection_matrices", [])
    fitted = bool(state.get("fitted", False))
    meta = state.get("meta", {})

    if dim is None:
        if "dim" in meta:
            dim = meta["dim"]
        elif len(pm_list) > 0:
            dim = pm_list[0].shape[0]
        else:
            raise ValueError("dim must be provided if the saved state has no projection matrices or meta['dim'].")

    if num_iterations is None:
        num_iterations = meta.get("num_iterations", None)

    model = NullspaceProjection(dim=dim, num_iterations=num_iterations or 0)
    # clear any existing list and append saved matrices as Parameters (requires_grad=False)
    model.projection_matrices = torch.nn.ParameterList()
    for P in pm_list:
        model.projection_matrices.append(torch.nn.Parameter(P.to(device), requires_grad=False))
    model.fitted = fitted

    # load other model params (if any) into model -- allow strict=False to avoid mismatch
    model_state = state.get("model_state", {})
    if model_state:
        try:
            model.load_state_dict(model_state, strict=False)
        except Exception as e:
            print("Warning: failed to load model_state with strict=False:", e)

    model.to(device)
    print(f"Loaded NullspaceProjection from {path} (matrices={len(model.projection_matrices)}, fitted={model.fitted})")
    return model    

def load_clip_vit_b_32(model_name=None, pretrained=None, cache_dir=None, device="cuda"):
    """
    Loads CLIP base model and a debiased wrapper from a single checkpoint file.
    Returns: debiased_model (CompensatedDebiasedSigLIP if checkpoint exists else base model),
             preprocess, tokenizer

    Expects checkpoint saved using:
      torch.save({'state_dict': debiased_model.state_dict(),
                  'compensation_vector': optional_numpy_array,
                  'extra': optional_dict}, os.path.join(save_path, 'debiased_model.pth'))
    """
    device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
    device = torch.device(device)

    base_model, preprocess = clip.load("ViT-B/16", device=device)

    base_model.float().eval()
    tokenizer = open_clip.get_tokenizer(CONFIG["model_name"])

    
    # projection_module = NullspaceProjection(dim=512)
    # projection_module.load_state_dict(torch.load('/home/aarish/VLM-superstition-analysis/models_final_debias/projection_module_aarish.pth', map_location=device))
    # load_nullspace_projection('/home/aarish/VLM-superstition-analysis/models_final_debias/projection_module_aarish.pth', device=device)
    # projection_module = load_nullspace_projection('/home/aarish/VLM-superstition-analysis/models_final_debias/projection_module_aarish_new.pth', device=device)
    projection_module_path = os.environ.get("PROJECTION_MODULE_PATH")
    projection_module = load_nullspace_projection(projection_module_path, device=device)

    # projection_module = torch.load('/home/aarish/VLM-superstition-analysis/models_final_debias/projection_module_aarish.pth')

    debiased_model = CompensatedDebiasedSigLIP(
        base_model,
        projection_module=projection_module,
        debias_strength=CONFIG["DEBIAS_STRENGTH"],
        compensate_similarity=CONFIG["SIMILARITY_COMPENSATION"]
    ).to(device)
    
    debiased_model_path = os.environ.get("DEBIASED_MODEL_PATH")
    # debiased_model.load_state_dict(torch.load('/home/aarish/VLM-superstition-analysis/models_final_debias/debiased_model_aarish.pth', map_location=device))
    debiased_model.load_state_dict(torch.load(debiased_model_path, map_location=device))

    return debiased_model, preprocess, tokenizer

    