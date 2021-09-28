from typing import Any, Union

import torch
from pytorch_lightning import LightningModule
from transformers import (
    TransfoXLConfig, TransfoXLLMHeadModel,
    GPT2Config, GPT2LMHeadModel
)
from .performance_encoder import PerformanceEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR


class TransformerModel(LightningModule):

    transformer: Union[TransfoXLLMHeadModel, GPT2LMHeadModel]

    def __init__(
        self,
        lr: float,
        num_velocity_bins: int,
        steps_per_second: int,
        n_positions: int,
        n_layer: int,
        d_embed: int,
        n_head: int,
        d_inner: int,
        mem_len: int,
        clamp_len: int,
        architecture: str = 'gpt2'
    ):
        super().__init__()

        self.save_hyperparameters({
            'lr': lr,
            'num_velocity_bins': num_velocity_bins,
            'steps_per_second': steps_per_second,
            'n_layer': n_layer,
            'd_embed': d_embed,
            'n_head': n_head,
            'd_inner': d_inner,
            'mem_len': mem_len,
            'clamp_len': clamp_len,
            'n_positions': n_positions,
            'architecture': architecture
        })

        self.encoder = PerformanceEncoder(
            num_velocity_bins=num_velocity_bins,
            steps_per_second=steps_per_second
        )

        if architecture == 'transfoxl':
            configuration = TransfoXLConfig(
                vocab_size=self.encoder.vocab_size,
                d_model=d_embed,
                n_layer=n_layer,
                d_embed=d_embed,
                n_head=n_head,
                d_inner=d_inner,
                mem_len=mem_len,
                clamp_len=clamp_len,
                adaptive=False,
                cutoffs=[0]
            )
            self.transformer = TransfoXLLMHeadModel(configuration)
        elif architecture == 'gpt2':
            configuration = GPT2Config(
                vocab_size=self.encoder.vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_layer=n_layer,
                n_embd=d_embed,
                n_head=n_head,
                n_inner=d_inner,
            )
            self.transformer = GPT2LMHeadModel(configuration)

    def forward(self, x: torch.Tensor):
        return self.transformer(x, labels=x) # type: ignore

    def training_step(self, batch: Any, batch_idx: int): # type: ignore
        output = self.forward(batch)
        if self.hparams['architecture'] == 'transfoxl':
            loss = output.losses.mean()
        else:
            loss = output.loss
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int): # type: ignore
        output = self.forward(batch)
        if self.hparams['architecture'] == 'transfoxl':
            loss = output.losses.mean()
        else:
            loss = output.loss
        self.log("val/loss", loss, prog_bar=False)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        scheduler = {
            "scheduler": CyclicLR(optimizer,
                step_size_up=10000,
                base_lr=self.hparams['lr'] / 100,
                max_lr=self.hparams['lr'],
                cycle_momentum=False
            ),
            'name': 'learning_rate',
            'interval':'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
