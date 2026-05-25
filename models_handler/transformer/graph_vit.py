import pytorch_lightning as pl
from models_handler.transformer.vit import VitClassifier

class GraphVit(VitClassifier):
    

    def __init__(
        self,
        backbone_type: str,
        head_type: str,
        gnn_type: str,
        bpt_percentage: float = 0.9,
        cosine_threshold: float = 0.7,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        min_epoch: int = 5,
        k_classes: int = 11,
        num_head_mha: int = 12, 
        use_weighted_loss: bool = False,
        contrastive_loss: bool = False, 
        masked_attention: bool = False, 
        full_dataset: bool = True, 
        db_path: str = ""         
    ) -> None:
        
        super().__init__(
            backbone_type=backbone_type,
            head_type=head_type,
            lr=lr,
            weight_decay=weight_decay,
            min_epochs_head=min_epoch,
            k_classes=k_classes,
            num_head_mha=num_head_mha,
            use_weighted_loss=use_weighted_loss,
            contrastive_loss=contrastive_loss,
            masked_attention=masked_attention,
            full_dataset = full_dataset, 
            db_path= db_path
        )

        self.save_hyperparameters(
            {
                "gnn_type": gnn_type, 
                "bpt_percentage": bpt_percentage,
                "cosine_threshold": cosine_threshold
            }
        )

        self.build_model()


    def build_model(self) -> None:







    