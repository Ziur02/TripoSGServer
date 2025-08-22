import socket
import json
import time
import io
import os
import base64
import random

from PIL import Image
from openai import OpenAI
from openai.types.chat import ChatCompletion

import numpy as np
import torch
import trimesh
from triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline

# import numpy as np
# import pandas as pd
# import cv2
# from onnxruntime import InferenceSession

# class WD14Tagger:
#     def __init__(self, model_path: str, tags_path: str):
#         self.model = InferenceSession(model_path)
#         self.tags = pd.read_csv(tags_path)

#     def _preprocess(self, image: Image.Image, size: int) -> np.ndarray:
#         image = image.convert('RGB')
#         image = np.asarray(image)[:, :, ::-1]  # RGB to BGR
#         h, w = image.shape[:2]
#         s = max(h, w)
#         delta_w = s - w
#         delta_h = s - h
#         top, bottom = delta_h // 2, delta_h - delta_h // 2
#         left, right = delta_w // 2, delta_w - delta_w // 2
#         image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
#         image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
#         image = image.astype(np.float32)
#         image = np.expand_dims(image, 0)
#         return image

#     def tag(self, image: Image.Image, threshold: float = 0.35) -> dict:
#         size = self.model.get_inputs()[0].shape[2]
#         input_name = self.model.get_inputs()[0].name
#         output_name = self.model.get_outputs()[0].name

#         image_data = self._preprocess(image, size)
#         conf = self.model.run([output_name], {input_name: image_data})[0][0]

#         result = dict()
#         for name, score in zip(self.tags['name'], conf):
#             if score >= threshold:
#                 result[name] = float(score)
#         return result

# tagger = WD14Tagger(
#     model_path=r"D:\dev\TripoSGServer\wd14\model.onnx",
#     tags_path=r"D:\dev\TripoSGServer\wd14\selected_tags.csv"
# )

triposg_scribble_weights_dir = "pretrained_weights/TripoSG-scribble"
pipe: TripoSGScribblePipeline = TripoSGScribblePipeline.from_pretrained(triposg_scribble_weights_dir).to("cuda", torch.float16)


client = OpenAI(
    api_key="sk-690ae890ec6c4b5ab544013be229cde2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def create_completion(image_base64_str: str) -> ChatCompletion:
    return client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "system",
                "content": [{"type":"text",
                             "text": (
                                "You are a professional prompt writer for diffusion-based generative models. "
                                "Your job is to analyze the content of the sketch image and generate a short, precise prompt."
                                "suitable for 3D model generation. Only describe the object itself, focusing on its key features. "
                                "The skectch is more like a child's drawing, with simple lines, but maybe not easy to recognize."
                                "The 3D model we will generate is more like a common low-poly 3d model, like car, tree, toy, animal, monster, building...... "
                                "Do not mention background, sketch lines, drawing quality, outline, or anything related with the drawing itself. "
                                "Do not mention something like 'only two wheels of the car is visible', which is not a feature of the object itself. "
                                "Output should be a list of concise keywords separated by commas, describing the object and its attributes only."
                             )
                }]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64_str}"},
                    },
                    {"type": "text", 
                     "text": ("Generate a concise diffusion model prompt describing the object in the sketch." 
                              "Focus on the object's identity and distinct features."
                              "Return only keywords separated by commas."
                              )
                    },
                ],
            }
        ],
    )

def start_server(host='0.0.0.0', port=50008):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"Server listening on {host}:{port}")

    while True:
        conn, addr = server.accept()
        print(f"Connected by {addr}")

        # 包装成类文件对象
        file_obj = conn.makefile('rwb')
        try:
            while True:
                # 阻塞直到收到一行（\n结尾），不会卡死不会粘包
                line = file_obj.readline()
                if not line:
                    print("Client disconnected.")
                    break
                # 注意：Python3下line是bytes，需要decode
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                # 处理数据
                imageData = json.loads(line_str)
                base64Str = imageData["imageBase64"]
                completion = create_completion(base64Str)
                prompt = completion.choices[0].message.content
                if "leaves" in prompt:
                    prompt = prompt.replace("leaves,", "")
                print(prompt)
                
                image = Image.open(io.BytesIO(base64.b64decode(base64Str))).convert("RGB")
                voxel_bools: torch.Tensor = pipe(
                    image=image,
                    prompt=prompt + ", low poly 3d model",
                    generator=torch.Generator(device=pipe.device).manual_seed(random.randint(0, 2**32 - 1)),
                    num_inference_steps=16,
                    guidance_scale=0,
                    attention_kwargs={"cross_attention_scale": 1.0, "cross_attention_2_scale": 0.3},
                    use_flash_decoder=False,
                    dense_octree_depth=6, hierarchical_octree_depth=7,
                    return_voxel=True,
                    voxel_resolution=25,
                ).samples

                # 保证是list
                if isinstance(voxel_bools, torch.Tensor):
                    voxel_bools = voxel_bools.cpu().numpy().flatten().tolist()
                elif isinstance(voxel_bools, np.ndarray):
                    voxel_bools = voxel_bools.flatten().tolist()

                result = {
                    "status": "success",
                    "resolution": 25,
                    "voxelBools": voxel_bools,
                }
                print("sending result")
                # 发回时一定加\\n分隔，且必须是str
                file_obj.write((json.dumps(result) + "\n").encode("utf-8"))
                file_obj.flush()
        except Exception as e:
            print(f"Fatal error: {e}")
        finally:
            conn.close()
            print("Connection closed")

if __name__ == "__main__":
    start_server()
