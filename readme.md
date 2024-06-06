## Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs


![system](pics/system.png)

**:heart: Implementation of the "Chain-of-Planned-Behaviour Workflow Elicits Few-Shot Mobility Generation in LLMs".**

>The powerful reasoning capabilities of large language models (LLMs) have brought revolutionary changes to many fields, but their performance in human behaviour generation has not yet been extensively explored. This gap likely emerges because the internal processes governing behavioral intentions cannot be solely explained by abstract reasoning. Instead, they are also influenced by a multitude of factors, including social norms and personal preference. Inspired by the Theory of Planned Behaviour (TPB), we develop a LLM workflow named Chain-of-Planned Behaviour (CoPB) for mobility behaviour generation, which reflects the important spatio-temporal dynamics of human activities. Through exploiting the cognitive structures of attitude, subjective norms, and perceived behaviour control in TPB, CoPB significantly enhance the ability of LLMs to reason the intention of next movement. Specifically, CoPB substantially reduces the error rate of mobility intention generation from 57.8% to 19.4%. To improve the scalability of the proposed CoPB workflow, we further explore the synergy between LLMs and mechanistic models. We find mechanistic mobility models, such as gravity model, can effec15 tively map mobility intentions to physical mobility behaviours. The strategy of integrating CoPB with gravity model can reduce the token cost by 97.7% and achieve better performance simultaneously. Besides, the proposed CoPB workflow can facilitate GPT-4-turbo to automatically generate high quality labels for mobility behavior reasoning. We show such labels can be leveraged to fine-tune the smaller-scale, open source LLaMA 3-8B, which significantly reduces usage costs without sacrificing the quality of the generated behaviours.




### :exclamation: Requirements:

```python
matplotlib==3.7.4
numpy==1.24.4
openai==0.28.1
pycitysim==1.15.1
scipy==1.12.0
setproctitle==1.3.3
tqdm==4.66.1
```

And you should first set your openai key in function: 'setOpenAi'  (in utils.py) 



### :star2: Generate:

**Step1**: Gen the intention templates:

```shell
python genDayIntention_TPB.py  --TemplateRes_PATH TemplateRes
```

**Step2**: Gen the traces:

```shell
python genTraceGravity.py --Template_PATH TemplateRes --Trajectory_PATH TrajectoryRes
```

**Eval**: Statistical Evaluation:

```shell
python eval.py --trajPath TrajectoryRes
```


### :clipboard: Mobility Datasets: 
We are making the Tencent dataset we use open source [ link: https://pan.baidu.com/s/1wth1pbybpEG5dsHIj9kuQw?pwd=g6gp with password: g6gp ] to facilitate reproduction and further research.

### :sparkles: Finetuned Params of LLaMA 3-8B:
The dataset for finetuning is in the path *'LLaMA_Finetune\Dataset'*.
The finetuned params are in the path *'LLaMA_Finetune\0515_v5_intent&time_2_epoch'*. (There seems to be some problem with the display in the anonymous repository, but it is normal after downloading. There are 13 files in the folder.)
You need to merge the params with initial LLaMA3-8B version following https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README.md

Thanks for the help from the great work [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).