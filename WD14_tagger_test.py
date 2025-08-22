import numpy as np
import pandas as pd
import cv2
from PIL import Image
from onnxruntime import InferenceSession
import argparse

class WD14Tagger:
    def __init__(self, model_path: str, tags_path: str):
        self.model = InferenceSession(model_path)
        self.tags = pd.read_csv(tags_path)

    def _preprocess(self, image: Image.Image, size: int) -> np.ndarray:
        image = image.convert('RGB')
        image = np.asarray(image)[:, :, ::-1]  # RGB to BGR
        h, w = image.shape[:2]
        s = max(h, w)
        delta_w = s - w
        delta_h = s - h
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        return image

    def tag(self, image: Image.Image, threshold: float = 0.35) -> dict:
        size = self.model.get_inputs()[0].shape[2]
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name

        image_data = self._preprocess(image, size)
        conf = self.model.run([output_name], {input_name: image_data})[0][0]

        result = dict()
        for name, score in zip(self.tags['name'], conf):
            if score >= threshold:
                result[name] = float(score)
        return result

if __name__ == "__main__":
    tagger = WD14Tagger(
        model_path=r"D:\dev\TripoSGServer\wd14\model.onnx",
        tags_path=r"D:\dev\TripoSGServer\wd14\selected_tags.csv"
    )

    # img = Image.open(r"D:\dev\TripoSGServer\assets\example_scribble_data\cat_with_wings.png")
    # img = Image.open(r"D:\dev\Comfy3D_WinPortable\ComfyUI\input\Paint_14.png")
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    
    tags = tagger.tag(Image.open(args.input_file))
    prompt = ",".join([p for p in tags.keys()])
    
    print(f"Detected tags:{prompt}")
