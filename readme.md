## MobiGeaR (Mobility Generation as Reasoning)  

#### Framework:

![system](pics/system.png)



#### Requirements:

```python
matplotlib==3.7.4
numpy==1.24.4
openai==0.28.1
pywolong==2.5.0
scipy==1.12.0
setproctitle==1.3.3
tqdm==4.66.1
```

And you should first set your openai key in function: 'setOpenAi'  (in utils.py) 



#### Generate:

Step1: Gen the templates:

```shell
python genDayIntention.py  --TemplateRes_PATH TemplateRes
```

Step2: Gen the traces:

```shell
python genTraceGravity.py --Template_PATH TemplateRes --Trajectory_PATH TrajectoryRes
```

Eval: Statistical Evaluation:

```shell
python eval.py --trajPath TrajectoryRes
```
