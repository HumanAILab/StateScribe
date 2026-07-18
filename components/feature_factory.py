import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from config import DINO_FEATURE_MODEL_NAME, TEXT_EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class FeatureFactory:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="feature_factory")
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_FEATURE_MODEL_NAME)
        self.dino_model = AutoModel.from_pretrained(DINO_FEATURE_MODEL_NAME).to(self.device)
        self.dino_model.eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_EMBEDDING_MODEL_NAME)
        self.text_model = AutoModel.from_pretrained(TEXT_EMBEDDING_MODEL_NAME).to(self.device)
        self.text_model.eval()
        logger.debug("FeatureFactory initialized on %s", self.device)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def compute_dino_features(self, images_rgb: List[np.ndarray]) -> List[np.ndarray]:
        if not images_rgb:
            return []
        return self.executor.submit(self._compute_dino_features_sync, images_rgb).result()

    def compute_text_features(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        return self.executor.submit(self._compute_text_features_sync, texts).result()

    def compute_dino_feature(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        feats = self.compute_dino_features([image_rgb]) if image_rgb is not None else []
        return feats[0] if feats else None

    def compute_text_feature(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        feats = self.compute_text_features([text])
        return feats[0] if feats else None

    def _compute_dino_features_sync(self, images_rgb: List[np.ndarray]) -> List[np.ndarray]:
        pil_images = [self._prepare_dino_image(img) for img in images_rgb]
        inputs = self.dino_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.dino_model(**inputs)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state[:, 0, :]
        pooled = F.normalize(pooled, p=2, dim=1)
        feats = pooled.detach().cpu().numpy().astype(np.float32)
        return [feat for feat in feats]

    def _compute_text_features_sync(self, texts: List[str]) -> List[np.ndarray]:
        encoded = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            model_output = self.text_model(**encoded)
        pooled = self._mean_pooling(model_output, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        feats = pooled.detach().cpu().numpy().astype(np.float32)
        return [feat for feat in feats]

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    @staticmethod
    def _prepare_dino_image(image_rgb: np.ndarray) -> Image.Image:
        img = image_rgb
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
