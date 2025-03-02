## Generating Realistic Human Mobility Data with Hybrid Large Language Model Agent


![system](pics/system.png)

**:heart: Implementation of the "Generating Realistic Human Mobility Data with Hybrid Large Language Model Agent".**


### :exclamation: Requirements:

```python
matplotlib==3.7.4
numpy==1.24.4
openai==0.28.1
pycitysim==1.21.0
scipy==1.12.0
setproctitle==1.3.3
tqdm==4.66.5
accelerate==0.23.0
einops==0.8.1
ema_pytorch==0.2.3
packaging==21.3
pandas==1.4.4
Pillow==11.1.0
pytorch_fid==0.3.0
scikit_learn==1.1.2
setuptools==65.3.0
torch==2.4.1
torchvision==0.21.0
```

For generating intention sequences with LLMs, you should first set your openai key in function: 'setOpenAi'  (in CoPB/utils.py) 


### :star2: How to Use Our Data Generator to Generate Intention-aware Trajectories:

**Step1**: Gen the intention templates:

```shell
python genDayIntention_TPB.py  --TemplateRes_PATH TemplateRes
```

**Step2**: Gen the traces and evaluation:

With Diffusion model: 
```shell
python CUDA_VISIBLE_DEVICES=0 python main.py --epochs 100001 --expIndex 0 --denoise queryonce --dataset foursquare
python CUDA_VISIBLE_DEVICES=1 python main.py --epochs 100001 --expIndex 1 --denoise queryonce --dataset tencent
python CUDA_VISIBLE_DEVICES=2 python main.py --epochs 100001 --expIndex 2 --denoise queryonce --dataset chinamobile
cd DLModel/Eval
python eval_foursquare.py
python eval_tencent.py
python eval_mobile.py
```

With lightweight Gravity model: 
```shell
python genTraceGravity.py --Template_PATH TemplateRes --Trajectory_PATH TrajectoryRes
python eval.py --trajPath TrajectoryRes
```


### :clipboard: Mobility Datasets: 
We are making the Tencent and Chinamobile dataset we use open source [ link: https://pan.baidu.com/s/1wth1pbybpEG5dsHIj9kuQw?pwd=g6gp with password: g6gp ] to facilitate reproduction and further research.

### :sparkles: Finetuned Params of Lightweight LLaMA 3-8B:
The dataset for finetuning is in the path *'LLaMA_Finetune\Dataset'*.
The finetuned params are in the path *'LLaMA_Finetune\training_save'*. (There seems to be some problem with the display in the anonymous repository, but it is normal after downloading. There are 13 files in the folder.)
You need to merge the params with initial LLaMA3-8B version following https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md

The specific training parameters for LLaMA3-8B are in 'CoPB\LLaMA_Finetune\llama3_lora_sft.yaml'. You can use them directly after modifying the path.

Thanks for the help from the great work [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).