import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional, Tuple, Dict, Any

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer inputs"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiModalAttention(nn.Module):
    """Multi-head attention with support for cross-modal interactions"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None, modal_mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, value.size(1), self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        if modal_mask is not None:
            scores = scores.masked_fill(modal_mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + query), attention_weights

class CrossModalTransformerLayer(nn.Module):
    """Transformer layer with both self-attention and cross-attention"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiModalAttention(d_model, num_heads, dropout)
        
        # Cross-attention
        self.cross_attention = MultiModalAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cross_input=None, self_mask=None, cross_mask=None):
        # Self-attention
        attended_x, self_attn_weights = self.self_attention(x, x, x, self_mask)
        
        # Cross-attention (if cross_input is provided)
        if cross_input is not None:
            cross_attended_x, cross_attn_weights = self.cross_attention(
                attended_x, cross_input, cross_input, cross_mask)
            attended_x = cross_attended_x
        else:
            cross_attn_weights = None
        
        # Feed-forward
        ffn_output = self.ffn(attended_x)
        output = self.layer_norm(ffn_output + attended_x)
        
        return output, self_attn_weights, cross_attn_weights

class DrugTargetTransformer(nn.Module):
    """Main model for drug-target interaction prediction with multi-task learning"""
    
    def __init__(
        self,
        chem_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        protein_model_name: str = "Rostlab/prot_bert",
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_ligand_len: int = 512,
        max_protein_len: int = 1024,
        freeze_encoders: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_ligand_len = max_ligand_len
        self.max_protein_len = max_protein_len
        
        # Load pre-trained encoders
        self.chem_encoder = AutoModel.from_pretrained(chem_model_name)
        self.protein_encoder = AutoModel.from_pretrained(protein_model_name)
        
        # Load tokenizers
        self.chem_tokenizer = AutoTokenizer.from_pretrained(chem_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
        
        # Freeze encoders if specified
        if freeze_encoders:
            for param in self.chem_encoder.parameters():
                param.requires_grad = False
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
        
        # Dimension alignment layers
        chem_dim = self.chem_encoder.config.hidden_size
        protein_dim = self.protein_encoder.config.hidden_size
        
        self.chem_projection = nn.Linear(chem_dim, d_model)
        self.protein_projection = nn.Linear(protein_dim, d_model)
        
        # Modal embeddings (0: CLS, 1: ligand, 2: protein)
        self.modal_embedding = nn.Embedding(3, d_model)
        
        # Positional encodings
        self.pos_encoding = PositionalEncoding(d_model, max_ligand_len + max_protein_len + 2)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Cross-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Binding affinity regression head
        self.binding_affinity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Binding level classification head (will be set dynamically)
        self.binding_level_head = None
        self.dropout = nn.Dropout(dropout)
    
    def set_num_binding_levels(self, num_levels):
        """Set the number of binding levels for classification head"""
        device = next(self.parameters()).device
        self.binding_level_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 4, num_levels)
        ).to(device)  # Ensure the new head is on the correct device
        
    def encode_molecules(self, smiles_list):
        """Encode SMILES strings using ChemBERTa"""
        device = next(self.parameters()).device
        
        # Tokenize SMILES
        encoded = self.chem_tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_ligand_len,
            return_tensors="pt"
        )
        
        # Move tokenized inputs to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.chem_encoder(**encoded)
            embeddings = outputs.last_hidden_state
        
        # Project to common dimension
        embeddings = self.chem_projection(embeddings)
        
        return embeddings, encoded['attention_mask']
    
    def encode_proteins(self, protein_seqs):
        """Encode protein sequences using ProtBERT"""
        device = next(self.parameters()).device
        
        # Add spaces between amino acids for ProtBERT
        spaced_seqs = [" ".join(seq) for seq in protein_seqs]
        
        # Tokenize sequences
        encoded = self.protein_tokenizer(
            spaced_seqs,
            padding=True,
            truncation=True,
            max_length=self.max_protein_len,
            return_tensors="pt"
        )
        
        # Move tokenized inputs to device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.protein_encoder(**encoded)
            embeddings = outputs.last_hidden_state
        
        # Project to common dimension
        embeddings = self.protein_projection(embeddings)
        
        return embeddings, encoded['attention_mask']
    
    def create_modal_masks(self, ligand_mask, protein_mask, batch_size):
        """Create masks for different modalities"""
        # Total sequence: [CLS] + ligand + [SEP] + protein + [SEP]
        total_len = 1 + ligand_mask.size(1) + 1 + protein_mask.size(1) + 1
        
        # Attention mask (1 for valid tokens, 0 for padding)
        attention_mask = torch.zeros(batch_size, total_len, device=ligand_mask.device)
        attention_mask[:, 0] = 1  # CLS token
        attention_mask[:, 1:1+ligand_mask.size(1)] = ligand_mask
        attention_mask[:, 1+ligand_mask.size(1)] = 1  # SEP token
        attention_mask[:, 2+ligand_mask.size(1):2+ligand_mask.size(1)+protein_mask.size(1)] = protein_mask
        attention_mask[:, 2+ligand_mask.size(1)+protein_mask.size(1)] = 1  # Final SEP token
        
        return attention_mask
    
    def forward(self, smiles_list, protein_seqs):
        batch_size = len(smiles_list)
        device = next(self.parameters()).device
        
        # Encode molecules and proteins (now returns tensors on correct device)
        ligand_emb, ligand_mask = self.encode_molecules(smiles_list)
        protein_emb, protein_mask = self.encode_proteins(protein_seqs)
        
        # Ensure everything is on the same device (should already be, but double-check)
        ligand_emb = ligand_emb.to(device)
        protein_emb = protein_emb.to(device)
        ligand_mask = ligand_mask.to(device)
        protein_mask = protein_mask.to(device)
        
        # Create sequence with special tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
        # Concatenate: [CLS] + ligand + [SEP] + protein + [SEP]
        sequence = torch.cat([
            cls_tokens,
            ligand_emb,
            sep_tokens,
            protein_emb,
            sep_tokens
        ], dim=1)
        
        # Add modal embeddings
        modal_ids = torch.zeros(batch_size, sequence.size(1), dtype=torch.long, device=device)
        modal_ids[:, 0] = 0  # CLS
        modal_ids[:, 1:1+ligand_emb.size(1)] = 1  # Ligand
        modal_ids[:, 1+ligand_emb.size(1)] = 0  # SEP
        modal_ids[:, 2+ligand_emb.size(1):2+ligand_emb.size(1)+protein_emb.size(1)] = 2  # Protein
        modal_ids[:, 2+ligand_emb.size(1)+protein_emb.size(1)] = 0  # SEP
        
        modal_emb = self.modal_embedding(modal_ids)
        sequence = sequence + modal_emb
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence.transpose(0, 1)).transpose(0, 1)
        
        # Create attention mask
        attention_mask = self.create_modal_masks(ligand_mask, protein_mask, batch_size)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            sequence, self_attn, cross_attn = layer(
                sequence, 
                cross_input=None, 
                self_mask=attention_mask.unsqueeze(1).unsqueeze(1), 
                cross_mask=None
            )
            attention_weights.append((self_attn, cross_attn))
        
        # Extract CLS token representation for prediction
        cls_representation = sequence[:, 0]  # [batch_size, d_model]
        
        # Predictions
        binding_affinity = self.binding_affinity_head(cls_representation)
        
        outputs = {
            'binding_affinity': binding_affinity.squeeze(-1),
            'attention_weights': attention_weights,
            'cls_representation': cls_representation
        }
        
        # Add binding level prediction if head is available
        if self.binding_level_head is not None:
            binding_level = self.binding_level_head(cls_representation)
            outputs['binding_level'] = binding_level
        
        return outputs

class DTIDataset(Dataset):
    """Dataset class for Drug-Target Interaction data with multi-task support"""
    
    def __init__(self, csv_file=None, dataframe=None, sequence_col='sequence', 
                 smiles_col='smiles', affinity_col='binding_affinity_nM', 
                 binding_level_col='binding_level', apply_log=True):
        """
        Initialize dataset from CSV file or DataFrame
        
        Args:
            csv_file: Path to CSV file
            dataframe: Pre-loaded pandas DataFrame
            sequence_col: Column name for protein sequences
            smiles_col: Column name for SMILES strings
            affinity_col: Column name for binding affinity values (in nM)
            binding_level_col: Column name for binding level (high/medium/low)
            apply_log: Whether to apply log transformation to binding affinity
        """
        if csv_file is not None:
            self.df = pd.read_csv(csv_file)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Either csv_file or dataframe must be provided")
        
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.affinity_col = affinity_col
        self.binding_level_col = binding_level_col
        self.apply_log = apply_log
        
        # Clean data
        self._clean_data()
        
        # Process binding levels
        self._process_binding_levels()
        
        # Apply log transformation if requested
        if self.apply_log:
            self._apply_log_transformation()
        
        # Store column mappings
        self.sequences = self.df[sequence_col].tolist()
        self.smiles = self.df[smiles_col].tolist()
        self.affinities = self.df[affinity_col].values.astype(np.float32)
        self.binding_levels = self.df[f'{binding_level_col}_encoded'].values.astype(np.int64)
        
        print(f"Loaded {len(self.df)} drug-target pairs")
        if self.apply_log:
            print(f"Log-transformed affinity range: {self.affinities.min():.2f} to {self.affinities.max():.2f}")
        else:
            print(f"Affinity range: {self.affinities.min():.2f} to {self.affinities.max():.2f} nM")
        print(f"Binding level distribution: {dict(self.df[binding_level_col].value_counts())}")
    
    def _clean_data(self):
        """Clean and validate the dataset"""
        initial_len = len(self.df)
        
        # Remove rows with missing values
        required_cols = [self.sequence_col, self.smiles_col, self.affinity_col, self.binding_level_col]
        self.df = self.df.dropna(subset=required_cols)
        
        # Remove empty sequences or SMILES
        self.df = self.df[self.df[self.sequence_col].str.len() > 0]
        self.df = self.df[self.df[self.smiles_col].str.len() > 0]
        
        # Convert affinity to numeric
        self.df[self.affinity_col] = pd.to_numeric(self.df[self.affinity_col], errors='coerce')
        self.df = self.df.dropna(subset=[self.affinity_col])
        
        # Remove non-positive binding affinities (required for log transformation)
        if self.apply_log:
            initial_positive = len(self.df)
            self.df = self.df[self.df[self.affinity_col] > 0]
            if len(self.df) < initial_positive:
                print(f"Removed {initial_positive - len(self.df)} samples with non-positive binding affinities")
        
        # Remove sequences that are too long (for memory efficiency)
        max_seq_len = 1000  # Adjust based on your needs
        self.df = self.df[self.df[self.sequence_col].str.len() <= max_seq_len]
        
        print(f"Data cleaning: {initial_len} -> {len(self.df)} samples")
    
    def _process_binding_levels(self):
        """Process and encode binding level categories"""
        # Normalize binding level strings (handle case variations)
        self.df[self.binding_level_col] = self.df[self.binding_level_col].str.lower().str.strip()
        
        # Create label mapping
        unique_levels = sorted(self.df[self.binding_level_col].unique())
        print(f"Found binding levels: {unique_levels}")
        
        # Standard mapping (adjust if your categories are different)
        level_mapping = {}
        for i, level in enumerate(unique_levels):
            level_mapping[level] = i
        
        # Store mapping for later use
        self.level_mapping = level_mapping
        self.level_names = unique_levels
        
        # Encode levels
        encoded_col = f'{self.binding_level_col}_encoded'
        self.df[encoded_col] = self.df[self.binding_level_col].map(level_mapping)
        
        print(f"Binding level mapping: {level_mapping}")
    
    def _apply_log_transformation(self):
        """Apply log transformation to binding affinity values"""
        original_col = f"{self.affinity_col}_original"
        
        # Store original values
        self.df[original_col] = self.df[self.affinity_col].copy()
        
        # Apply log transformation (using natural log)
        self.df[self.affinity_col] = np.log(self.df[self.affinity_col])
        
        print(f"Applied log transformation to {self.affinity_col}")
        print(f"Original range: {self.df[original_col].min():.2f} to {self.df[original_col].max():.2f} nM")
        print(f"Log-transformed range: {self.df[self.affinity_col].min():.2f} to {self.df[self.affinity_col].max():.2f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'smiles': self.smiles[idx],
            'binding_affinity': self.affinities[idx],
            'binding_level': self.binding_levels[idx]
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    sequences = [item['sequence'] for item in batch]
    smiles = [item['smiles'] for item in batch]
    affinities = torch.tensor([item['binding_affinity'] for item in batch], dtype=torch.float32)
    binding_levels = torch.tensor([item['binding_level'] for item in batch], dtype=torch.long)
    
    return {
        'sequences': sequences,
        'smiles': smiles,
        'binding_affinity': affinities,
        'binding_level': binding_levels
    }

def create_data_loaders(csv_file, batch_size=16, test_size=0.2, val_size=0.1, 
                       sequence_col='sequence', smiles_col='smiles', 
                       affinity_col='binding_affinity_nM', binding_level_col='binding_level',
                       apply_log=True, random_state=42):
    """
    Create train, validation, and test data loaders from CSV file
    
    Args:
        csv_file: Path to CSV file
        batch_size: Batch size for training
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        sequence_col: Column name for protein sequences
        smiles_col: Column name for SMILES strings
        affinity_col: Column name for binding affinity values (in nM, will be log-transformed)
        binding_level_col: Column name for binding level (high/medium/low)
        apply_log: Whether to apply log transformation to binding affinity
        random_state: Random seed for reproducibility
    """
    # Load full dataset
    full_dataset = DTIDataset(csv_file, sequence_col=sequence_col, 
                             smiles_col=smiles_col, affinity_col=affinity_col,
                             binding_level_col=binding_level_col, apply_log=apply_log)
    
    # Split data with stratification on binding levels for balanced splits
    train_val_df, test_df = train_test_split(
        full_dataset.df, test_size=test_size, random_state=random_state,
        stratify=full_dataset.df[binding_level_col]
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), random_state=random_state,
        stratify=train_val_df[binding_level_col]
    )
    
    # Create datasets
    train_dataset = DTIDataset(dataframe=train_df, sequence_col=sequence_col,
                              smiles_col=smiles_col, affinity_col=affinity_col,
                              binding_level_col=binding_level_col, apply_log=False)  # Already transformed
    val_dataset = DTIDataset(dataframe=val_df, sequence_col=sequence_col,
                            smiles_col=smiles_col, affinity_col=affinity_col,
                            binding_level_col=binding_level_col, apply_log=False)  # Already transformed
    test_dataset = DTIDataset(dataframe=test_df, sequence_col=sequence_col,
                             smiles_col=smiles_col, affinity_col=affinity_col,
                             binding_level_col=binding_level_col, apply_log=False)  # Already transformed
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset.affinities, full_dataset.level_mapping

class DTILoss(nn.Module):
    """Multi-task loss for DTI prediction with both regression and classification"""
    
    def __init__(self, affinity_weight=1.0, level_weight=0.5):
        super().__init__()
        self.affinity_weight = affinity_weight
        self.level_weight = level_weight
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        # Binding affinity losses
        mse_loss = self.mse_loss(predictions['binding_affinity'], targets['binding_affinity'])
        mae_loss = self.mae_loss(predictions['binding_affinity'], targets['binding_affinity'])
        
        # Binding level loss (if available)
        level_loss = torch.tensor(0.0, device=predictions['binding_affinity'].device)
        if 'binding_level' in predictions and 'binding_level' in targets:
            level_loss = self.ce_loss(predictions['binding_level'], targets['binding_level'])
        
        # Total loss
        total_loss = (self.affinity_weight * mse_loss + 
                     self.level_weight * level_loss)
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'level_loss': level_loss
        }

def create_model():
    """Create and initialize the DTI model"""
    model = DrugTargetTransformer(
        d_model=768,
        num_heads=12,
        num_layers=6,
        dropout=0.1,
        freeze_encoders=True  # Start with frozen encoders
    )
    return model

def train_step(model, batch, optimizer, loss_fn, device):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    smiles = batch['smiles']
    proteins = batch['sequences']
    targets = {
        'binding_affinity': batch['binding_affinity'].to(device),
        'binding_level': batch['binding_level'].to(device)
    }
    
    predictions = model(smiles, proteins)
    losses = loss_fn(predictions, targets)
    
    losses['total_loss'].backward()
    optimizer.step()
    
    return losses

def evaluate(model, data_loader, loss_fn, device, level_names=None):
    """Evaluate model on validation/test set with both regression and classification metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_level_loss = 0
    num_batches = 0
    
    # For regression metrics
    affinity_predictions = []
    affinity_targets = []
    
    # For classification metrics
    level_predictions = []
    level_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            smiles = batch['smiles']
            proteins = batch['sequences']
            targets = {
                'binding_affinity': batch['binding_affinity'].to(device),
                'binding_level': batch['binding_level'].to(device)
            }
            
            predictions = model(smiles, proteins)
            losses = loss_fn(predictions, targets)
            
            total_loss += losses['total_loss'].item()
            total_mse += losses['mse_loss'].item()
            total_mae += losses['mae_loss'].item()
            total_level_loss += losses['level_loss'].item()
            num_batches += 1
            
            # Store regression predictions
            affinity_predictions.extend(predictions['binding_affinity'].cpu().numpy())
            affinity_targets.extend(targets['binding_affinity'].cpu().numpy())
            
            # Store classification predictions
            if 'binding_level' in predictions:
                level_pred = torch.argmax(predictions['binding_level'], dim=1)
                level_predictions.extend(level_pred.cpu().numpy())
                level_targets.extend(targets['binding_level'].cpu().numpy())
    
    # Calculate regression metrics
    affinity_correlation = np.corrcoef(affinity_predictions, affinity_targets)[0, 1]
    
    # Calculate classification metrics
    level_accuracy = accuracy_score(level_targets, level_predictions)
    level_precision, level_recall, level_f1, _ = precision_recall_fscore_support(
        level_targets, level_predictions, average='weighted'
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(level_targets, level_predictions)
    
    results = {
        'avg_loss': total_loss / num_batches,
        'avg_mse': total_mse / num_batches,
        'avg_mae': total_mae / num_batches,
        'avg_level_loss': total_level_loss / num_batches,
        'affinity_correlation': affinity_correlation,
        'level_accuracy': level_accuracy,
        'level_precision': level_precision,
        'level_recall': level_recall,
        'level_f1': level_f1,
        'confusion_matrix': conf_matrix,
        'affinity_predictions': affinity_predictions,
        'affinity_targets': affinity_targets,
        'level_predictions': level_predictions,
        'level_targets': level_targets
    }
    
    return results

def train_model(csv_file, num_epochs=50, batch_size=16, learning_rate=1e-4, 
                sequence_col='sequence', smiles_col='smiles', 
                affinity_col='binding_affinity_nM', binding_level_col='binding_level',
                apply_log=True, save_path='best_model.pt'):
    """
    Complete training pipeline with multi-task learning
    
    Args:
        csv_file: Path to CSV file with training data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        sequence_col: Column name for protein sequences
        smiles_col: Column name for SMILES strings
        affinity_col: Column name for binding affinity values (in nM, will be log-transformed)
        binding_level_col: Column name for binding level (high/medium/low)
        apply_log: Whether to apply log transformation to binding affinity
        save_path: Path to save the best model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_affinities, level_mapping = create_data_loaders(
        csv_file, batch_size=batch_size, sequence_col=sequence_col,
        smiles_col=smiles_col, affinity_col=affinity_col, 
        binding_level_col=binding_level_col, apply_log=apply_log
    )
    
    # Create model and set up classification head
    model = create_model().to(device)
    model.set_num_binding_levels(len(level_mapping))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Binding level mapping: {level_mapping}")
    
    # Loss function and optimizer
    loss_fn = DTILoss(affinity_weight=1.0, level_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    level_names = list(level_mapping.keys())
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0
        
        for batch in train_loader:
            losses = train_step(model, batch, optimizer, loss_fn, device)
            epoch_train_loss += losses['total_loss'].item()
            num_train_batches += 1
        
        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        val_results = evaluate(model, val_loader, loss_fn, device, level_names)
        val_losses.append(val_results['avg_loss'])
        
        # Learning rate scheduling
        scheduler.step(val_results['avg_loss'])
        
        # Save best model
        if val_results['avg_loss'] < best_val_loss:
            best_val_loss = val_results['avg_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'level_mapping': level_mapping,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, save_path)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_results['avg_loss']:.4f}")
            print(f"  Val MAE: {val_results['avg_mae']:.4f}")
            print(f"  Val Affinity Correlation: {val_results['affinity_correlation']:.4f}")
            print(f"  Val Level Accuracy: {val_results['level_accuracy']:.4f}")
            print(f"  Val Level F1: {val_results['level_f1']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Final evaluation on test set
    print("\nFinal Test Evaluation:")
    model.load_state_dict(torch.load(save_path)['model_state_dict'])
    test_results = evaluate(model, test_loader, loss_fn, device, level_names)
    
    print("=== REGRESSION METRICS ===")
    print(f"Test MSE: {test_results['avg_mse']:.4f}")
    print(f"Test MAE: {test_results['avg_mae']:.4f}")
    print(f"Test Correlation: {test_results['affinity_correlation']:.4f}")
    
    print("\n=== CLASSIFICATION METRICS ===")
    print(f"Test Accuracy: {test_results['level_accuracy']:.4f}")
    print(f"Test Precision: {test_results['level_precision']:.4f}")
    print(f"Test Recall: {test_results['level_recall']:.4f}")
    print(f"Test F1-Score: {test_results['level_f1']:.4f}")
    
    print("\n=== CONFUSION MATRIX ===")
    print("Binding Level Confusion Matrix:")
    conf_matrix = test_results['confusion_matrix']
    print(f"Labels: {level_names}")
    for i, row in enumerate(conf_matrix):
        print(f"{level_names[i]:>8}: {row}")
    
    return model, train_losses, val_losses, test_results, level_mapping

# Example usage and training setup
if __name__ == "__main__":
    # Set environment variable to suppress tokenizer warnings
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Train model from CSV file
    csv_file = "final_w-bad-r.csv"  # Your actual CSV file
    
    # Expected CSV columns:
    # - Sequence: protein sequence
    # - SMILES: SMILES string  
    # - Binding_Value_nM: binding affinity in nM (will be log-transformed automatically)
    # - Affinity_Category: categorical binding level (high/medium/low)
    
    model, train_losses, val_losses, test_results, level_mapping = train_model(
        csv_file=csv_file,
        num_epochs=100,  # Increased from 50
        batch_size=8,  # Adjust based on GPU memory
        learning_rate=5e-5,  # Reduced from 1e-4
        sequence_col='Sequence',           # Match your CSV column
        smiles_col='SMILES',              # Match your CSV column
        affinity_col='Binding_Value_nM',  # Match your CSV column
        binding_level_col='Affinity_Category',  # Match your CSV column
        apply_log=True  # Will automatically apply log transformation
    )
    
    # Enhanced plotting with both tasks
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 10))
    
    # Training curves
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    # Regression results
    plt.subplot(2, 3, 2)
    plt.scatter(test_results['affinity_targets'], test_results['affinity_predictions'], alpha=0.6)
    plt.plot([min(test_results['affinity_targets']), max(test_results['affinity_targets'])], 
             [min(test_results['affinity_targets']), max(test_results['affinity_targets'])], 'r--')
    plt.xlabel('True Binding Affinity (log nM)')
    plt.ylabel('Predicted Binding Affinity (log nM)')
    plt.title(f'Regression: r={test_results["affinity_correlation"]:.3f}')
    
    # Classification confusion matrix
    plt.subplot(2, 3, 3)
    level_names = list(level_mapping.keys())
    sns.heatmap(test_results['confusion_matrix'], 
                xticklabels=level_names, yticklabels=level_names,
                annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Level')
    plt.ylabel('True Level')
    plt.title(f'Classification Acc: {test_results["level_accuracy"]:.3f}')
    
    # Classification metrics by class
    from sklearn.metrics import classification_report
    plt.subplot(2, 3, 4)
    class_report = classification_report(test_results['level_targets'], 
                                       test_results['level_predictions'], 
                                       target_names=level_names, output_dict=True)
    
    metrics_df = pd.DataFrame(class_report).iloc[:-1, :-2].T  # Remove support and averages
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlBu_r')
    plt.title('Per-Class Classification Metrics')
    
    # Residuals plot
    plt.subplot(2, 3, 5)
    residuals = np.array(test_results['affinity_predictions']) - np.array(test_results['affinity_targets'])
    plt.scatter(test_results['affinity_targets'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Binding Affinity (log nM)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # Performance summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    summary_text = f"""
    REGRESSION METRICS:
    • Correlation: {test_results['affinity_correlation']:.4f}
    • MAE: {test_results['avg_mae']:.4f}
    • MSE: {test_results['avg_mse']:.4f}
    
    CLASSIFICATION METRICS:
    • Accuracy: {test_results['level_accuracy']:.4f}
    • Precision: {test_results['level_precision']:.4f}
    • Recall: {test_results['level_recall']:.4f}
    • F1-Score: {test_results['level_f1']:.4f}
    
    BINDING LEVELS:
    {dict(zip(level_names, range(len(level_names))))}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.show()
    
    # Example prediction on new data
    model.eval()
    with torch.no_grad():
        sample_smiles = ["CCO"]  # Ethanol
        sample_sequence = ["MKWVTFISLLFLFSSAYS"]  # Sample protein sequence
        
        prediction = model(sample_smiles, sample_sequence)
        predicted_log_affinity = prediction['binding_affinity'].item()
        predicted_affinity_nM = np.exp(predicted_log_affinity)
        
        # Get predicted binding level
        level_probs = torch.softmax(prediction['binding_level'], dim=1)
        predicted_level_idx = torch.argmax(level_probs, dim=1).item()
        predicted_level = level_names[predicted_level_idx]
        confidence = level_probs[0][predicted_level_idx].item()
        
        print(f"\nSample prediction:")
        print(f"  Log binding affinity: {predicted_log_affinity:.2f}")
        print(f"  Binding affinity: {predicted_affinity_nM:.2f} nM")
        print(f"  Predicted binding level: {predicted_level} (confidence: {confidence:.3f})")
        print(f"  Level probabilities: {dict(zip(level_names, level_probs[0].cpu().numpy()))}")
        