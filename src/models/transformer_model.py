from typing import Any, Union, Tuple, Optional
from lightning.pytorch import LightningModule
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup


class TransformerModel(LightningModule):
    """
    This is a PyTorch Lightning module that defines a model using the Transformer architecture.
    The specific variant of the Transformer (GPT-2, GPTNeo, or TransfoXL) can be specified during
    the model's initialization.
    """

    transformer: Union[GPT2LMHeadModel, GPTNeoXForCausalLM]
    total_steps: int = 0

    def __init__(
        self,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        warmup_steps: int,
        n_positions: int,
        n_layer: int,
        n_head: int,
        n_embed: int,
        vocab_size: int,
        architecture: str = "gpt",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.lr = lr

        self.save_hyperparameters(
            {
                "lr": lr,
                "betas": betas,
                "vocab_size": vocab_size,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_positions": n_positions,
                "n_embed": n_embed,
                "architecture": architecture,
            }
        )

        if architecture == "gpt2":
            configuration = GPT2Config(
                vocab_size=vocab_size,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_embd=n_embed,
                n_layer=n_layer,
                n_head=n_head,
            )
            self.transformer = GPT2LMHeadModel(configuration)
        elif architecture == "gptneo":
            configuration = GPTNeoXConfig(
                vocab_size=vocab_size,
                hidden_size=n_embed,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
                intermediate_size=n_embed * 4,
                max_position_embeddings=n_positions,
                use_cache=not gradient_checkpointing,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.transformer = GPTNeoXForCausalLM(configuration)
        else:
            raise RuntimeError("unknown architecture: " + architecture)

    def forward(self, inputs: dict):
        output = self.transformer(**inputs, labels=inputs["input_ids"])
        return output.loss

    def training_step(self, batch: Any, _: int):
        loss = self.forward(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, _: int):
        loss = self.forward(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Any, _: int):
        loss = self.forward(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,  # type: ignore
            lr=self.hparams["lr"],
            betas=self.hparams["betas"],
        )

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams["warmup_steps"],
            num_training_steps=self.total_steps,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
