"""DRQN Core.

This module implements a sequence-aware TD update core for recurrent DQN.

It extends :class:`DQNCore` and supports:

- Transition batches ``(B, ...)`` via inherited DQN update logic.
- Sequence batches ``(B, T, ...)`` with optional ``sequence_mask``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch as th
import torch.nn.functional as F

from rllib.model_free.baselines.value_based.dqn.core import DQNCore
from rllib.model_free.common.utils.common_utils import _to_column, _to_scalar
from rllib.model_free.common.utils.policy_utils import _get_per_weights


class DRQNCore(DQNCore):
    """DRQN core.

    Notes
    -----
    Supports both:
    - transition batches ``(B, ...)`` (same as DQN), and
    - sequence batches ``(B, T, ...)`` with optional ``batch.sequence_mask``.
    """

    def update_from_batch(self, batch: Any) -> Dict[str, float]:
        """Run one DRQN update from a transition or sequence batch.

        Parameters
        ----------
        batch : Any
            Replay batch containing either:
            - transition tensors with shape ``(B, ...)``, or
            - sequence tensors with shape ``(B, T, ...)``.
            Optional ``sequence_mask`` of shape ``(B, T, 1)`` or broadcastable
            equivalent marks valid timesteps.

        Returns
        -------
        Dict[str, float]
            Logging metrics and PER TD-errors array.
        """
        self._bump()

        obs = batch.observations.to(self.device)
        act = batch.actions.to(self.device).long()
        rew = batch.rewards.to(self.device)
        next_obs = batch.next_observations.to(self.device)
        done = batch.dones.to(self.device)

        if obs.dim() <= 2:
            return super().update_from_batch(batch)

        if obs.dim() < 3:
            raise ValueError(f"Expected obs dim >= 3 for sequence batch, got {tuple(obs.shape)}")

        bsz, tlen = int(obs.shape[0]), int(obs.shape[1])
        rew = rew if rew.dim() == 3 else rew.view(bsz, tlen, 1)
        done = done if done.dim() == 3 else done.view(bsz, tlen, 1)
        act = act if act.dim() == 2 else act.view(bsz, tlen)

        w = _get_per_weights(batch, bsz, device=self.device)  # None or (B,1)

        mask = getattr(batch, "sequence_mask", None)
        if isinstance(mask, th.Tensor):
            seq_mask = mask.to(self.device).float()
        else:
            seq_mask = th.ones((bsz, tlen, 1), device=self.device)
        valid_steps = seq_mask.sum().clamp_min(1.0)

        q_all = self.head.q(obs)                                            # (B,T,A)
        q_sa = q_all.gather(-1, act.unsqueeze(-1))                          # (B,T,1)

        with th.no_grad():
            q_next_target_all = self.head.q_target(next_obs)                # (B,T,A)
            if self.double_dqn:
                a_star = th.argmax(self.head.q(next_obs), dim=-1, keepdim=True)
                q_next = q_next_target_all.gather(-1, a_star)              # (B,T,1)
            else:
                q_next = q_next_target_all.max(dim=-1, keepdim=True).values
            target = rew + self.gamma * (1.0 - done) * q_next

        if self.huber:
            per_step = F.smooth_l1_loss(q_sa, target, reduction="none")
        else:
            per_step = F.mse_loss(q_sa, target, reduction="none")
        per_step = per_step * seq_mask

        if w is None:
            loss = per_step.sum() / valid_steps
        else:
            w_col = _to_column(w).view(bsz, 1, 1)
            loss = (per_step * w_col).sum() / (seq_mask * w_col).sum().clamp_min(1e-6)

        self.opt.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self._clip_params(
                self._q_params,
                max_grad_norm=self.max_grad_norm,
                optimizer=self.opt,
            )
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_params(self._q_params, max_grad_norm=self.max_grad_norm)
            self.opt.step()

        if self.sched is not None:
            self.sched.step()

        with th.no_grad():
            td_step = (target - q_sa).abs() * seq_mask                      # (B,T,1)
            td_error = td_step.sum(dim=1).view(bsz) / seq_mask.sum(dim=1).view(bsz).clamp_min(1.0)
            q_mean = (q_sa * seq_mask).sum() / valid_steps
            tgt_mean = (target * seq_mask).sum() / valid_steps

        self._maybe_update_target(
            target=getattr(self.head, "q_target", None),
            source=self.head.q,
            interval=self.target_update_interval,
            tau=self.tau,
        )

        return {
            "loss/q": float(_to_scalar(loss)),
            "q/mean": float(_to_scalar(q_mean)),
            "target/mean": float(_to_scalar(tgt_mean)),
            "lr": float(self._get_optimizer_lr(self.opt)),
            "per/td_errors": td_error.detach().cpu().numpy(),
        }
