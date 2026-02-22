import numpy as np
from typing import Dict, List
from doctr.models import ocr_predictor

def get_textsDetector_model():
    return ocr_predictor(pretrained=True, resolve_lines=True)

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
            image_results = []
            for block in page.blocks:
                for line in block.lines:
                    full_text = " ".join([word.value for word in line.words])

                    xmin = min(w.geometry[0][0] for w in line.words)
                    ymin = min(w.geometry[0][1] for w in line.words)
                    xmax = max(w.geometry[1][0] for w in line.words)
                    ymax = max(w.geometry[1][1] for w in line.words)

                    image_results.append({
                        "bbox": [xmin, ymin, xmax, ymax],
                        "text": full_text,
                        "score": sum(w.confidence for w in line.words) / len(line.words)
                    })
            outputs.append(image_results)

    return outputs
