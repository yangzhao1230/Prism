import torch
import torch.nn as nn
import torch.nn.functional as F

from caduceus.modeling_caduceus import CaduceusPreTrainedModel, Caduceus
from caduceus.configuration_caduceus import CaduceusConfig


def entropy(mask, eps=1e-10):
    entropy = -mask * torch.log(mask + eps) - (1 - mask) * torch.log(1 - mask + eps)
    average_entropy = entropy.mean()
    return average_entropy


# Paper Section 3.2: Signal Encoder — 3-layer 1D-CNN that maps multimodal epigenomic signals
# (H3K27ac, DNase-seq, Hi-C) to n_context context-aware weight vectors for sequence weighting.
class SignalWeightGenerator(nn.Module):
    def __init__(self, signal_size, d_model, n_context, cnn_dim, weight_act="sigmoid"):
        super().__init__()
        self.signal_size = signal_size
        self.d_model = d_model
        self.n_context = n_context
        self.weight_act = weight_act

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(signal_size, cnn_dim, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(cnn_dim, cnn_dim*2, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(cnn_dim*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(cnn_dim*2, cnn_dim*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(cnn_dim*4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.weight_projector = nn.Sequential(
            nn.Linear(cnn_dim*4, d_model * n_context),
        )

    def forward(self, signals):
        batch_size = signals.shape[0]

        encoded_signals = self.cnn_encoder(signals)

        flat_weights = self.weight_projector(encoded_signals)

        raw_weights = flat_weights.view(batch_size, self.d_model, self.n_context)

        if self.weight_act == "sigmoid":
            activated_weights = torch.sigmoid(raw_weights)
        elif self.weight_act == "softmax":
            activated_weights = F.softmax(raw_weights, dim=1)
        else:
            raise ValueError(f"Don't support the activation mode: {self.weight_act}")

        return activated_weights


# Paper Section 3: Prism model — Caduceus (bidirectional Mamba) backbone with multimodal signal
# integration via SignalWeightGenerator. n_context=2 weight vectors are learned from epigenomic
# signals and used to re-weight sequence positions before expression prediction.
class Prism(CaduceusPreTrainedModel):
    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        self.caduceus = Caduceus(config, **{'ignore_embed_layer': True})
        self.seq_input_layer = nn.Linear(config.base_size, config.d_model)

        self.signal_input_layer = nn.Linear(config.signal_size, config.d_model)
        self.n_context = config.n_context if hasattr(config, 'n_context') else 4
        self.gaussian_kernel_t = config.gaussian_kernel_t if hasattr(config, 'gaussian_kernel_t') else 2
        self.weight_act = config.weight_act if hasattr(config, 'weight_act') else "sigmoid"

        self.intervention_loss_weight = config.intervention_loss_weight
        if self.intervention_loss_weight != 0:
            self.weight_generator = SignalWeightGenerator(
                signal_size=config.signal_size,
                d_model=config.d_model,
                n_context=self.n_context,
                cnn_dim=config.cnn_dim,
                weight_act=self.weight_act
            )

        self.pToExpr = nn.Sequential(
            nn.Linear(config.d_model + config.rna_feat_dim if config.useRNAFeat else config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        self.post_init()

    def compute_uniform_loss(self, context_weights):
        # context_weights: (bs, d_model, n_context)
        norm = torch.norm(context_weights, p=2, dim=1, keepdim=True)
        A_s_norm = context_weights / (norm + 1e-8)

        A_s_norm_t = A_s_norm.transpose(1, 2)
        dot_products = torch.bmm(A_s_norm_t, A_s_norm)

        t = self.gaussian_kernel_t
        G_t = torch.exp(2 * t * dot_products - 2 * t)

        sample_losses = torch.log(torch.sum(G_t, dim=(1, 2)))

        uniform_loss = torch.mean(sample_losses)
        return uniform_loss

    def forward(
        self,
        seqs,
        signals,
        rna_feat=None,
        bio_mask=None,
        mask_regions=None,
        peak_mask=None,
        output_hidden_states=False,
        return_dict=False,
    ):
        seq_input_embeds = self.seq_input_layer(seqs)

        aux_infor = {}

        signals = signals[..., :self.config.signal_size]
        signal_input_embeds = self.signal_input_layer(signals)


        if self.intervention_loss_weight != 0:
            batch_size, seq_len, signal_size = signals.shape
            signals_transposed = signals.permute(0, 2, 1)
            # (bs, d_model, n_context)
            context_weights = self.weight_generator(signals_transposed)
            uniform_loss = self.compute_uniform_loss(context_weights)
            aux_infor['uniform_loss'] = uniform_loss

            # (bs, len, d_model, 1 + n_context)
            context_weights_with_one = torch.cat([torch.ones_like(context_weights[:, :, :1]), context_weights], dim=-1).unsqueeze(1)

            # (bs, len, d_model, 1 + n_context)
            signal_input_embeds_weighted = signal_input_embeds.unsqueeze(-1) * context_weights_with_one
            seq_input_embeds_expanded = seq_input_embeds.unsqueeze(-1).expand(-1, -1, -1, self.n_context + 1)
            inputs_embeds = seq_input_embeds_expanded + signal_input_embeds_weighted

            inputs_embeds = inputs_embeds.permute(0, 3, 1, 2).reshape(batch_size * (self.n_context + 1), -1, self.config.d_model)

            outputs = self.caduceus(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ) # (bs, len, d_model, 1 + n_context)

            hidden_states = outputs.reshape(batch_size, (self.n_context + 1), -1, self.config.d_model) # (bs, 1 + n_context, len, d_model)
            original_hidden_states = hidden_states[:, 0, :, :] # (bs, len, d_model)
            intervention_hidden_states = hidden_states[:, 1:, :, :].mean(dim=1) # (bs, len, d_model)
        else:
            inputs_embeds = seq_input_embeds + signal_input_embeds
            outputs = self.caduceus( # (bs, len, d_model)
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            original_hidden_states = outputs

        if self.config.center_len:
            start_index = (original_hidden_states.shape[1] - self.config.center_len) // 2
            end_index = start_index + self.config.center_len
            original_hidden_states = original_hidden_states[:, start_index: end_index, :]
            if self.intervention_loss_weight != 0:
                intervention_hidden_states = intervention_hidden_states[:, start_index: end_index, :]

        original_hidden_states = torch.mean(original_hidden_states, dim=1)  # [bs, d_model]
        if self.intervention_loss_weight != 0:
            intervention_hidden_states = torch.mean(intervention_hidden_states, dim=1)  # [bs, d_model]

        original_p_embed = torch.cat([original_hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else original_hidden_states
        logits = self.pToExpr(original_p_embed)
        logits = logits.float()

        if self.intervention_loss_weight != 0:
            intervention_p_embed = torch.cat([intervention_hidden_states, rna_feat], dim=-1) if self.config.useRNAFeat else intervention_hidden_states
            intervention_logits = self.pToExpr(intervention_p_embed)
            intervention_logits = intervention_logits.float()

            aux_infor['intervention_logits'] = intervention_logits

        return logits, aux_infor, None
