from abc import abstractmethod
from pytorch_lightning import LightningModule
from torch import Tensor


class BaseLearner(LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.test_result_path: str = "test"


    @abstractmethod
    def unfreezing_handler(
        self, 
        val_loss_name: str = "val_loss", 
        log_blocks: bool = True,
        plateau_threshold: float = 1e-4,
        min_epoch: int | None = None
    ) -> None:
        pass


    @abstractmethod
    def multi_task_forward(
        self,
        batch: Tensor, 
        attention_mask: Tensor = None, 
        aggregate: bool = True,
        norm: bool = True,
        dropout: bool = True,
        return_embedding: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        pass