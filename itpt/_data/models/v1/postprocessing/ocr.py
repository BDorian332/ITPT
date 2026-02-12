import numpy as np
from typing import Dict, List
from doctr.models import ocr_predictor

def get_textsDetector_model():
    return ocr_predictor(pretrained=True)

def detect_texts(images: List[np.ndarray], predictor) -> List[List[dict]]:
    """
    return : list of texts found by image :
    [
        [
            {"bbox": [x1,y1,x2,y2], "text": str, "score": float},
            ...
        ]
        ...
    ]
    """

    result = predictor(images)

    outputs = []
    for page in result.pages:
        entries = []
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x1, y1), (x2, y2) = word.geometry
                    bbox = [x1, y1, x2, y2]
                    entries.append({
                        "bbox": bbox,
                        "text": word.value,
                        "score": float(word.confidence),
                    })
        outputs.append(entries)

    return outputs
