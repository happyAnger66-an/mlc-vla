import torch
from torch import nn

from model.network import LayerNorm, RMSNorm  


class TrajHead(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=2085,
        traj_size=2048,
        enable_critic_head=False,
        critic_input_scale=1.0,
        use_rmsnorm=False,
        use_bias=False,
    ):
        super().__init__()


        self.ln = LayerNorm(input_dim, bias=use_bias) if not use_rmsnorm else RMSNorm(input_dim)

        self.head = nn.Linear(input_dim, output_dim, bias=False)
        self.critic_head = None
        self.critic_input_scale = critic_input_scale
        if enable_critic_head:
            self.critic_head = nn.Linear(input_dim, 1)
        self.traj_size = traj_size


    def forward(self, hidden_states):
        hidden_states = self.ln(hidden_states)
        planner_logits = self.head(hidden_states)
        planner_logits = planner_logits.float()
        pred_next_wp_ids = torch.argmax(planner_logits[:, :-1], dim=-1) # B, N
        wp_probs = nn.functional.softmax(planner_logits[:, :-1], dim=-1)
        pred_main_action_ids = pred_next_wp_ids[:, -15:]
        
        output = {
            "logits": planner_logits,
            "pred_next_wp_ids": pred_next_wp_ids,
            "probs": wp_probs,
            "pred_main_action_ids": pred_main_action_ids,
        }
        if self.critic_head:
            planner_critic_value = self.critic_head(hidden_states * self.critic_input_scale)
            planner_critic_value = planner_critic_value.float()
            output.update(
                {
                    "critic_value": planner_critic_value[:, :-1]
                }
            )

        return output