# CoGMA + InfantAnimator
General movement assessment (GMA) is a non-invasive method used to evaluate neuromotor behavior in infants under six months of age and is considered a reliable tool for the early detection of cerebral palsy (CP). However, traditional GMA relies on the subjective judgment of multiple internationally certified physicians, making it time-consuming and limiting its accessibility for widespread use. Furthermore, artificial intelligence (AI) approaches may overcome these limitations but are usually based on motion skeletons and lack the ability to capture detailed body information. Here, we propose CoGMA (Collaborative General Movements Assessment), a novel multi-modality co-learning framework for GMA. By integrating multimodal large language models as auxiliary networks during training, CoGMA enables efficient learning while utilizing only skeletal data during inference. Experimental results demonstrate that CoGMA performs robustly not only in assessing writhing movements but also in zero-shot evaluation of fidgety movements. Additionally, to address the need for anonymized data sharing, we introduce InfantAnimator produces non-identifiable, anonymized video datasets that preserve critical motion features, supporting broader data sharing and research initiatives. This framework significantly enhances the GMA methodology and lays the groundwork for future advancements in early detection and research on infant neuromotor behavior.

## InfantAnimator Quickstart
For the released model checkpoint, it supports generating videos at a 576x1024 resolution.
### Environment setup
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch==2.5.1+cu124 xformers --index-url https://download.pytorch.org/whl/cu124
cd InfantAnimator
pip install -r requirements.txt
```
### Download weights
Please download weights manually from [FrancisRing/StableAnimator](https://huggingface.co/FrancisRing/StableAnimator) and [wwYinYin/InfantAnimator](https://huggingface.co/wwYinYin/InfantAnimator)
All the weights should be organized in models as follows The overall file structure of this project should be organized as follows:
```
InfantAnimator/
├── DWPose
├── animation
├── checkpoints
│   ├── DWPose
│   │   ├── dw-ll_ucoco_384.onnx
│   │   └── yolox_l.onnx
│   ├── Animation
│   │   └── checkpoint-26500
│   │       ├── pose_net-26500.pth
│   │       ├── face_encoder-26500.pth
│   │       └── unet-26500.pth
│   ├── SVD
│   │   ├── feature_extractor
│   │   ├── image_encoder
│   │   ├── scheduler
│   │   ├── unet
│   │   ├── vae
│   │   ├── model_index.json
│   │   ├── svd_xt.safetensors
│   │   └── svd_xt_image_decoder.safetensors
│   └── inference.zip
├── models
│   │   └── antelopev2
│   │       ├── 1k3d68.onnx
│   │       ├── 2d106det.onnx
│   │       ├── genderage.onnx
│   │       ├── glintr100.onnx
│   │       └── scrfd_10g_bnkps.onnx
├── command_basic_infer.sh
├── face_mask_extraction.py
├── face_mask_extraction_multi.py
├── inference_basic.py
├── requirement.txt
```
### Model inference
A sample configuration for testing is provided as command_basic_infer.sh. You can also easily modify the various configurations according to your needs.
```
bash command_basic_infer.sh
```
