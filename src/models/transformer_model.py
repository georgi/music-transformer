from typing import Any, Union, List, Tuple

import torch
from pytorch_lightning import LightningModule
from transformers import (
    TransfoXLConfig, TransfoXLLMHeadModel,
    GPT2Config, GPT2LMHeadModel,
    GPTNeoConfig, GPTNeoForCausalLM
)
from .performance_encoder import PerformanceEncoder
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR

class TransformerModel(LightningModule):

    transformer: Union[TransfoXLLMHeadModel, GPT2LMHeadModel, GPTNeoForCausalLM]

    def __init__(
        self,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        num_velocity_bins: int,
        steps_per_second: int,
        n_positions: int,
        n_layer: int,
        n_head: int,
        n_embed: int,
        attention_types: List[Any],
        architecture: str,
        gradient_checkpointing: bool = False
    ):
        super().__init__()

        self.lr = lr

        self.save_hyperparameters({
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'num_velocity_bins': num_velocity_bins,
            'steps_per_second': steps_per_second,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_positions': n_positions,
            'n_embed': n_embed,
            'attention_types': attention_types,
            'architecture': architecture
        })

        self.encoder = PerformanceEncoder(
            num_velocity_bins=num_velocity_bins,
            steps_per_second=steps_per_second
        )

        if architecture == 'transfoxl':
            configuration = TransfoXLConfig(
                vocab_size=self.encoder.vocab_size,
                n_layer=n_layer,
                n_head=n_head,
                mem_len=200,
                clamp_len=1000,
                adaptive=False,
                cutoffs=[0]
            )
            self.transformer = TransfoXLLMHeadModel(configuration)
        elif architecture == 'gpt2':
            configuration = GPT2Config(
                vocab_size=self.encoder.vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_embd=n_embed,
                n_layer=n_layer,
                n_head=n_head,
            )
            self.transformer = GPT2LMHeadModel(configuration)
        elif architecture == 'gptneo':
            configuration = GPTNeoConfig(
                vocab_size=self.encoder.vocab_size,
                max_position_embeddings=n_positions,
                num_layers=n_layer,
                attention_types=attention_types,
                hidden_size=n_embed,
                num_heads=n_head,
                use_cache=not gradient_checkpointing,
                gradient_checkpointing=gradient_checkpointing
            )
            self.transformer = GPTNeoForCausalLM(configuration)
        else:
            raise RuntimeError('unknown architecture: ' + architecture)

    def forward(self, x: torch.Tensor):
        output = self.transformer(x, labels=x) # type: ignore
        if self.hparams['architecture'] == 'transfoxl':
            loss = output.losses.mean()
        else:
            loss = output.loss
        return loss

    def training_step(self, batch: Any, _: int):
        loss = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, _: int):
        loss = self.forward(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
            lr=self.hparams['lr'],
            # betas=self.hparams['betas'],
            # weight_decay=self.hparams['weight_decay']
        )
        scheduler = {
            "scheduler": CyclicLR(
                optimizer=optimizer,
                # mode="exp_range",
                step_size_up=100000,
                base_lr=self.hparams['lr'] / 100,
                max_lr=self.hparams['lr'],
                cycle_momentum=False
            ),
            'name': 'learning_rate',
            'interval':'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
