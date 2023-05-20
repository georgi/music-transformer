from typing import Any, Union, Tuple, Optional
import torch
from pytorch_lightning import LightningModule
from transformers import (
    TransfoXLConfig,
    TransfoXLLMHeadModel,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR


class TransformerModel(LightningModule):
    """
    This is a PyTorch Lightning module that defines a model using the Transformer architecture.
    The specific variant of the Transformer (GPT-2, GPTNeo, or TransfoXL) can be specified during
    the model's initialization.

    1. The `TransformerModel` class extends the `LightningModule`, which is PyTorch Lightning's
      base module class. It contains the logic for a full training/evaluation loop. The class contains
      a transformer model from the Hugging Face's `transformers` library, which can be one of three
      types: `TransfoXLLMHeadModel`, `GPT2LMHeadModel`, or `GPTNeoForCausalLM`.

    2. The `__init__` method initializes the model. It takes a number of hyperparameters, including
      the learning rate, beta values for the Adam optimizer, epsilon value for numerical stability in Adam,
      weight decay for L2 regularization, the number of velocity bins and steps per second for the
      Performance Encoder, and transformer model-specific hyperparameters like the number of positions,
      layers, heads, and embeddings, attention types, and dropout rates. The architecture type is also specified here.

    3. The `forward` method defines the forward pass of the model. It takes an input tensor `x`,
      passes it through the transformer model, and calculates the loss. The calculation of the loss differs
      depending on the transformer architecture used.

    4. The `training_step` and `validation_step` methods define what happens during one training or
      validation step, respectively. They both calculate and return the loss, and log it for progress tracking.

    5. The `configure_optimizers` method sets up the optimizer and learning rate scheduler to be used
      during training. It uses the Adam optimizer with parameters defined during initialization and
      a cyclic learning rate scheduler.

    """

    transformer: Union[TransfoXLLMHeadModel, GPT2LMHeadModel, GPTNeoXForCausalLM]

    def __init__(
        self,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        n_positions: int,
        n_layer: int,
        n_head: int,
        n_embed: int,
        architecture: str = "gptneo",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.lr = lr
        self.vocab_size = 500

        self.save_hyperparameters(
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_positions": n_positions,
                "n_embed": n_embed,
                "architecture": architecture,
            }
        )

        if architecture == "transfoxl":
            configuration = TransfoXLConfig(
                vocab_size=self.vocab_size,
                n_layer=n_layer,
                n_head=n_head,
                mem_len=200,
                clamp_len=1000,
                adaptive=False,
                cutoffs=[0],
            )
            self.transformer = TransfoXLLMHeadModel(configuration)
        elif architecture == "gpt2":
            configuration = GPT2Config(
                vocab_size=self.vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_embd=n_embed,
                n_layer=n_layer,
                n_head=n_head,
            )
            self.transformer = GPT2LMHeadModel(configuration)
        elif architecture == "gptneo":
            configuration = GPTNeoXConfig(
                vocab_size=self.vocab_size,
                max_position_embeddings=n_positions,
                num_hidden_layers=n_layer,
                hidden_size=n_embed,
                num_attention_heads=n_head,
                use_cache=not gradient_checkpointing,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.transformer = GPTNeoXForCausalLM(configuration)
        else:
            raise RuntimeError("unknown architecture: " + architecture)

    def forward(self, x: torch.Tensor):
        output = self.transformer(x, labels=x)  # type: ignore
        if self.hparams["architecture"] == "transfoxl":
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
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            # betas=self.hparams['betas'],
            # weight_decay=self.hparams['weight_decay']
        )
        scheduler = {
            "scheduler": CyclicLR(
                optimizer=optimizer,
                # mode="exp_range",
                step_size_up=25000,
                step_size_down=100000,
                base_lr=self.hparams["lr"] / 100,
                max_lr=self.hparams["lr"],
                cycle_momentum=False,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
