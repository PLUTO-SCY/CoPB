import pickle

import numpy

from utils import *


def hm2float(hmstr):
    h,m = hmstr.split(':')
    h = int(h)
    m = int(m)
    return h+m/60

def getBasicDataLocal(personBasicInfo):
    '''
    适配一问到底的token设计, 并使用本地模型
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    education, gender, consumption, occupation = personBasicInfo.split('-')

    # ageDescription = "" if age == '0' else "Age: {}".format(age)
    genderDescription = "" if gender == 'uncertain' else "Gender: {}".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption: {}".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}".format(occupation)
    
#     personDescription = """You are a person and your basic information is as follows:
# {}; {}; {}; {}; {}""".format(ageDescription, genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription = """You are a person and your basic information is as follows:
{}; {}; {}; {}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    # globalInfo is what needs to be added every question.
    # Input the task Settings & basic character information & Individual principle in the role of "system"
    globalInfo = [
        # 首先交待任务, 并给出选择POI的示例。
        {"role": "system","content":
"""
{}
Your task is to make subjective decisions and choices with my guidance. Please note: your choices need to align with the information about your character.
""".format(personDescription)},]
    # print('Basic Data perpare done!')

    return personBasicInfo, globalInfo

def getBasicData2(personBasicInfo):
    '''
    适配一问到底的token设计,没有age信息了
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

#     personDescription = """The person's basic information is as follows:
# {}{}{}{}{}""".format(ageDescription, genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription = """The person's basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.
Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""},
                  {"role": "user","content":
"""{}
Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone."""
                    .format(personDescription)}]

    anchors = askChatGPT(AutoAnchor, model="gpt-4", temperature=1.3)
    # print('anchors now')
    # print(anchors)
    # sys.exit(0)

#     personDescription2 = """You are a person and your basic information is as follows:
# {}{}{}{}{}""".format(ageDescription, genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription2 = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription3 = """{}
You have some fixed principles or habits that you generally will not violate: {}""".format(personDescription2, anchors)

    # print(personDescription)
    # sys.exit(0)


    # globalInfo is what needs to be added every question.
    # Input the task Settings & basic character information & Individual principle in the role of "system"
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""
{}
Your task is to generate your own schedule throughout the day from 0:00 to 24:00. You need to consider how your character attributes relate to your behavior.
I would also need you to answer the reasons for each small shedule.

Example:
Now I provide you with some examples of time schedules and explain them. You can use them as references for planning.

Example1:
This is a 30-year-old investment advisor with a relatively low consumption level and a bachelor's degree. On October 10, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '06:18:29', 'sleep']
['08:15:57', '08:47:30', 'work']
['09:18:51', '13:18:45', 'work']
['13:52:59', '15:31:54', 'handle the trivialities']
['15:46:48', '22:49:26', 'work']
['23:50:42', '23:59:59', 'sleep']
Explanation:
For a significant number of people, working hours are not in the conventional sense of 9 to 5. For this person, work starts around 8 in the morning and continues until around 23:00. There is a significant occurrence of overtime in his work. For most of his workdays, there may not be many activities aside from work and rest.

Example2:
This is a 26-year-old male IT engineer with a moderate level of consumption. He holds a bachelor's degree. On October 14, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '08:01:30', 'sleep']
['09:12:30', '12:15:31', 'work']
['13:01:04', '15:17:04', 'excursion']
['18:35:30', '19:10:42', 'work']
['20:26:45', '21:41:46', 'at home']
['22:55:59', '23:24:48', 'handle the trivialities']
['23:30:50', '23:59:59', 'sleep']
Explanation:
His working hours are relatively normal. He starts work a little after 9 in the morning and finishes around 19:00 in the evening. There is also time for relaxation and exercise during the lunch break. After returning home, he can attend to some personal matters.

Example3:
This is a 40-year-old male online salesperson with a high school education and a moderate level of consumption. On December 1, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '11:37:58', 'sleep']
['12:04:41', '12:29:12', 'handle the trivialities']
['12:46:07', '12:59:12', 'home']
['13:16:04', '19:15:19', 'work']
['19:49:47', '23:59:59', 'home']
Explanation:
His work hours are quite flexible, starting work in the afternoon. Most of the other time is spent at home.

Summary:
From the examples above, we can see that, in reality, many people's lives are not as structured as we might imagine. In the lives of most people, it's either work or rest, with occasional activities like exercise or shopping – that's the norm. I hope in your generation process, you can also take into account these real-life factors.

Note that:
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24.00.
3. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].

Answer format: [["Event", "(begin_time, end_time)", "reason"], ["Event", "(begin_time, end_time)", "reason"]...]
""".format(personDescription3)},
    ]
    # print('Basic Data perpare done!')

    return personBasicInfo, anchors, globalInfo

def getBasicData3(personBasicInfo):
    '''
    适配一问到底的token设计,没有age信息了
    现在其实和2没区别
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

#     personDescription = """The person's basic information is as follows:
# {}{}{}{}{}""".format(ageDescription, genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription = """The person's basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.
Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""},
                  {"role": "user","content":
"""{}
Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone."""
                    .format(personDescription)}]

    anchors = askChatGPT(AutoAnchor, model="gpt-4", temperature=1.3)
    # print('anchors now')
    # print(anchors)
    # sys.exit(0)

#     personDescription2 = """You are a person and your basic information is as follows:
# {}{}{}{}{}""".format(ageDescription, genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription2 = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    personDescription3 = """{}
You have some fixed principles or habits that you generally will not violate: {}""".format(personDescription2, anchors)

    # print(personDescription)
    # sys.exit(0)


    # globalInfo is what needs to be added every question.
    # Input the task Settings & basic character information & Individual principle in the role of "system"
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""
{}
Your task is to generate your own schedule throughout the day from 0:00 to 24:00. You need to consider how your character attributes relate to your behavior.
I would also need you to answer the reasons for each small shedule.

Example:
Now I provide you with some examples of time schedules and explain them. You can use them as references for planning.

Example1:
This is a 30-year-old investment advisor with a relatively low consumption level and a bachelor's degree. On October 10, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '06:18:29', 'sleep']
['08:15:57', '08:47:30', 'work']
['09:18:51', '13:18:45', 'work']
['13:52:59', '15:31:54', 'handle the trivialities']
['15:46:48', '22:49:26', 'work']
['23:50:42', '23:59:59', 'sleep']
Explanation:
For a significant number of people, working hours are not in the conventional sense of 9 to 5. For this person, work starts around 8 in the morning and continues until around 23:00. There is a significant occurrence of overtime in his work. For most of his workdays, there may not be many activities aside from work and rest.

Example2:
This is a 26-year-old male IT engineer with a moderate level of consumption. He holds a bachelor's degree. On October 14, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '08:01:30', 'sleep']
['09:12:30', '12:15:31', 'work']
['13:01:04', '15:17:04', 'excursion']
['18:35:30', '19:10:42', 'work']
['20:26:45', '21:41:46', 'at home']
['22:55:59', '23:24:48', 'handle the trivialities']
['23:30:50', '23:59:59', 'sleep']
Explanation:
His working hours are relatively normal. He starts work a little after 9 in the morning and finishes around 19:00 in the evening. There is also time for relaxation and exercise during the lunch break. After returning home, he can attend to some personal matters.

Example3:
This is a 40-year-old male online salesperson with a high school education and a moderate level of consumption. On December 1, 2019, his schedule is as follows:
The format is [start time, end time, event type].
['00:00:00', '11:37:58', 'sleep']
['12:04:41', '12:29:12', 'handle the trivialities']
['12:46:07', '12:59:12', 'home']
['13:16:04', '19:15:19', 'work']
['19:49:47', '23:59:59', 'home']
Explanation:
His work hours are quite flexible, starting work in the afternoon. Most of the other time is spent at home.

Summary:
From the examples above, we can see that, in reality, many people's lives are not as structured as we might imagine. In the lives of most people, it's either work or rest, with occasional activities like exercise or shopping – that's the norm. I hope in your generation process, you can also take into account these real-life factors.

Note that:
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24.00.
3. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].
""".format(personDescription3)},
    ]
    # print('Basic Data perpare done!')

    return personBasicInfo, anchors, globalInfo


def getBasicData4(personBasicInfo, day):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

The specific requirements of your assignment are as follows:
1. Generate your own schedule for {}.
2. You need to consider how your character attributes relate to your behavior.
3. I would place a limit on the number of events you can have per day. Your final result cannot exceed this limit.
4. I want you to answer the reasons for each event, including why you want to do it and how to arrange the time.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].
4. Answer format: [["Event1", "(begin_time1, end_time1)", "reason1"], ["Event2", "(begin_time2, end_time2)", "reason2"]...]
""".format(personDescription, day)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo

def getBasicData4_0518(personBasicInfo, day):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

The specific requirements of your assignment are as follows:
1. Generate your own schedule for {}.
2. You need to consider how your character attributes relate to your behavior.
3. I'll tell you your home and workplace, and give you the POIs and distance information around you.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].
""".format(personDescription, day)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo

def getBasicData4_0510(personBasicInfo, day, N):
    '''
    一次性生成且无profile出现
    '''
    personDescription = """You are a human behavior generator."""

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}
I need you to give me a schedule for a person's day in {}.

Note that: 
1. All times are in 24-hour format.
2. I want to limit the total number of events in the day to {}. It means you have to schedule {} number of things, no more and no less. I hope you can make every decision based on this limit.
3. The generated schedule must start at 0:00 and end at 24:00. Don't let the schedule spill over into the next day.
4. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].
5. Answer format: [["Event1", "(begin_time1, end_time1)", "reason1"], ["Event2", "(begin_time2, end_time2)", "reason2"]...]
""".format(personDescription, day, N, N)},
    ]

    return globalInfo



def getBasicData4_2(personBasicInfo, day, N):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    为防止
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior.
2. I want to limit your total number of events in a day to {}. Your final result cannot exceed this limit.
3. I want you to answer the reasons for each event, including why you want to do it and how to arrange the time.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples of scheduling throughout the day for reference.
Example 1:
[
[go to sleep, (00:00, 11:11)], 
[go to work, (12:08, 12:24)], 
[eat, (12:35, 13:01)], 
[go to work, (13:15, 20:07)], 
[go to sleep, (21:03, 23:59)]
]

Example 2:
[
[go to sleep, (00:00, 06:04)], 
[eat, (08:11, 10:28)], 
[go home, (12:26, 13:06)], 
[excursion, (13:34, 13:53)], 
[go to work, (14:46, 16:19)]
]

Example 3:
[
[go to sleep, (00:00, 06:20)], 
[handle the trivialities of life, (07:18, 07:32)], 
[leisure or entertainment, (07:38, 17:27)], 
[handle the trivialities of life, (18:22, 19:11)],
 [go to sleep, (20:51, 23:59)]
]

Example 4:
[
[go to work, (9:21, 16:56)], 
[go home, (20:00, 23:59)]
]

Example 5:
[
[go to sleep, (00:00, 08:25)], 
[go to work, (09:01, 19:18)], 
[go home, (20:54, 23:59)]
]

Example 6:
[
[handle the trivialities of life, (07:18, 08:32)], 
[go to work, (11:21, 20:56)], 
[go home, (23:10, 23:59)]
]

Example 7:
[
[eat, (06:11, 7:28)], 
[handle the trivialities of life, (07:48, 08:32)],  
[go home, (9:00, 11:00)],
[medical treatment, (13:44, 17:03)],
[go home, (19:00, 23:59)]
]

Example 8:
[
[go to sleep, (00:00, 09:36)], 
[medical treatment, (11:44, 12:03)],
[go to work, (12:27, 14:56)], 
[go to sleep, (17:05, 23:59)]
]

As shown in the example, the day's planning should start with "go to sleep" and end with "go to sleep" or "go home".
""".format(personDescription, day, N)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo


def getBasicData5(personBasicInfo, day):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0123: 去除了事件数量的限制,来评估意图的生成结果
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

    personDescription = """The person's basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

The specific requirements of your assignment are as follows:
1. Generate your own schedule for {}.
2. You need to consider how your character attributes relate to your behavior.
3. I want you to answer the reasons for each event, including why you want to do it and how to arrange the time.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, have breakfast, have lunch, have dinner, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, Cultural Institutions and Events].
4. Answer format: [["Event1", "(begin_time1, end_time1)", "reason1"], ["Event2", "(begin_time2, end_time2)", "reason2"]...]
""".format(personDescription, day)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo

def getRatingPrompts():
    systemInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""You are now a discriminator of human behavior. I will provide 10 real people's behavior arrangements in a day, and another 3 behavior arrangements which could be real or could be the result of virtual generation. I hope you can rate these 3 uncertain behavioral arrangement to measure how similar they are to the real behavior arrangement.

10 real people's behavior arrangements are as follows:

First:
[
["go to sleep", "(00:00, 09:00)"], 
["go home", "(09:00, 17:50)"], 
["handle the trivialities of life", "(18:03, 18:14)"], 
["go to work", "(20:12, 22:31)"], 
["go to sleep", "(23:28, 23:59)"]
]

Second:
[
["go to sleep", "(00:00, 09:54)"], 
["go to work", "(10:19, 11:25)"], 
["go home", "(11:36, 11:48)"], 
["go to work", "(11:55, 20:41)"], 
["go home", "(21:01, 23:04)"]
]

Third:
[
["go to sleep", "(00:00, 07:03)"], 
["go to work", "(07:49, 12:29)"], 
["handle the trivialities of life", "(12:54, 13:57)"], 
["go to work", "(16:03, 21:23)"]
]

Fourth:
[
["go to sleep", "(00:00, 12:14)"], 
["eat", "(13:59, 15:29)"], 
["go home", "(16:28, 17:15)"], 
["go to work", "(17:20, 22:05)"], 
["go to sleep", "(22:23, 23:59)"]
]

Fifth:
[
["go to sleep", "(00:00, 07:00)"], 
["go to work", "(08:11, 16:31)"], 
["go home", "(17:47, 18:50)"], 
["do sports", "(19:54, 20:17)"], 
["go home", "(20:18, 20:41)"], 
["do sports", "(22:06, 22:18)"], 
["go to sleep", "(22:40, 23:59)"]
]

Sixth:
[
["go to sleep", "(00:00, 08:31)"], 
["go to work", "(10:59, 18:29)"], 
["go to sleep", "(20:47, 23:59)"]
]

Seventh:
[["go home", "(00:00, 23:59)"]]

Eighth:
[
["go home", "(00:00, 00:05)"], 
["leisure or entertainment", "(00:12, 00:42)"], 
["handle the trivialities of life", "(10:16, 19:36)"]
]

Ninth:
[
["go to sleep", "(00:00, 07:56)"], 
["go to work", "(09:28, 18:02)"], 
["go home", "(19:18, 22:54)"]
]

Tenth:
[
["go to sleep", "(00:00, 07:00)"],
["go home", "(07:00, 11:30)"], 
["handle the trivialities of life", "(12:50, 16:44)"], 
["go home", "(17:01, 17:13)"], 
["eat", "(17:26, 18:02)"], 
["go home", "(18:10, 23:59)"]
]
"""},]
    return systemInfo


def getBasicData6(personBasicInfo, day, N):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    '''
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}
e
Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes rlate to your behavior.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, day, N)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo


def getBasicData_0509(personBasicInfo, day, N, GPT_MODEL):
    '''
    0122: 去除了anchor机制,使得生成更加自由.
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    0509: 重构prompts,三块之间需要划分清楚,如何重构
    '''
    
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    # 需要在这里生成锚点.
    
    # 这是一个单独于逐次生成对话之外的锚点生成对话
    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.
Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""},
                  {"role": "user","content":
"""The person's basic information is as follows: 
{}{}{}{}
Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone and organize your answers in 1.xxx 2.xxx 3.xxx format."""
                    .format(genderDescription, educationDescription, consumptionDescription, occupationDescription)}]
    
    # print('personBasicInfo' + personBasicInfo)
    

    anchors = askChatGPT(AutoAnchor, model = GPT_MODEL, temperature = 1)  # 现在的效果好像要好一些
    # print('generate anchors:\n')

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

You have some habits or tendencies as follows:
{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes, routines or habits relate to your behavior decisions.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, anchors, day, N)},
    ]
    
    # print('Basic Data perpare done!')
    return globalInfo


def getBasicData_0509_CM(personBasicInfo, day, N, GPT_MODEL):
    '''
    0122: 去除了anchor机制,使得生成更加自由.
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    0509: 重构prompts,三块之间需要划分清楚,如何重构
    '''
    
    #'bachelor degree-man-medium'
    education, gender, consumption= personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}""".format(genderDescription, educationDescription, consumptionDescription)
    # 需要在这里生成锚点.
    
    # 这是一个单独于逐次生成对话之外的锚点生成对话
    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.
Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""},
                  {"role": "user","content":
"""The person's basic information is as follows: 
{}{}{}{}
Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone and organize your answers in 1.xxx 2.xxx 3.xxx format."""
                    .format(genderDescription, educationDescription, consumptionDescription, occupationDescription)}]
    
    # print('personBasicInfo' + personBasicInfo)
    

    anchors = askChatGPT(AutoAnchor, model = GPT_MODEL, temperature = 1)  # 现在的效果好像要好一些
    # print('generate anchors:\n')

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

You have some habits or tendencies as follows:
{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes, routines or habits relate to your behavior decisions.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, anchors, day, N)},
    ]
    
    # print('Basic Data perpare done!')
    return globalInfo


def getBasicData_0515_ablation_subject(personBasicInfo, day, N, GPT_MODEL):
    '''
    0122: 去除了anchor机制,使得生成更加自由.
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    0509: 重构prompts,三块之间需要划分清楚,如何重构
    '''
    
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior decisions.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, day, N)},
    ]
    
    # print('Basic Data perpare done!')
    return globalInfo


def getBasicData_0515_ablation_attitude(personBasicInfo, day, N, GPT_MODEL):
    '''
    0122: 去除了anchor机制,使得生成更加自由.
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    0509: 重构prompts,三块之间需要划分清楚,如何重构
    '''

    # 需要在这里生成锚点.
    
    # 这是一个单独于逐次生成对话之外的锚点生成对话
    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now you are a human, and I hope you can generate 3 habits or tendencies that you may have."""},
                  {"role": "user","content":
"""Please generate 3 anchor points. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone and organize your answers in 1.xxx 2.xxx 3.xxx format."""}]
    

    anchors = askChatGPT(AutoAnchor, model = GPT_MODEL, temperature = 1)  # 现在的效果好像要好一些
    # print('generate anchors:\n')

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""You have some habits or tendencies as follows:
{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes, routines or habits relate to your behavior decisions.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.
""".format(anchors, day, N)},
    ]
    
    # print('Basic Data perpare done!')
    return globalInfo



def getBasicData6_CM(personBasicInfo, day, N):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.
    '''
    #'bachelor degree-man-medium'
    education, gender, consumption= personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}""".format(genderDescription, educationDescription, consumptionDescription)

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the reasons for each event.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours, so this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He decided to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[handle the trivialities of life, (07:18, 08:32)], (Reason: When she gets up in the morning, she takes care of the trivial things in her life.)
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:59)], (Reason: After finishing the evening's work, she went home to rest.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]
""".format(personDescription, day, N)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo

# As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".

def getBasicData7(personBasicInfo, day, N):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.事实证明确实是有效果vc 的.
    0128: 移除了人物profile.做消融实验看看效果.就是不做role play了.
    '''

    personDescription = """You are a human behavior generator."""

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}
Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior.
2. I want to limit your total number of events in a day to {}. Your final result cannot exceed this limit.
3. I want you to answer the reasons for each event, including why you want to do it and how to arrange the time.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.
""".format(personDescription, day, N, N)},
    ]
    # print('Basic Data perpare done!')
  
# 先删掉再说吧  
# Here are some intent sequence examples for reference.

# Example 1:
# [
# ["go to sleep", "(00:00, 09:00)"], 
# ["go home", "(09:00, 17:50)"], 
# ["handle the trivialities of life", "(18:03, 18:14)"], 
# ["go to work", "(20:12, 22:31)"], 
# ["go to sleep", "(23:28, 23:59)"]
# ]

# Example 2:
# [
# ["go to sleep", "(00:00, 09:54)"], 
# ["go to work", "(10:19, 11:25)"], 
# ["go home", "(11:36, 11:48)"], 
# ["go to work", "(11:55, 20:41)"], 
# ["go home", "(21:01, 23:04)"]
# ]

# Example 3:
# [
# ["go to sleep", "(00:00, 07:03)"], 
# ["go to work", "(07:49, 12:29)"], 
# ["handle the trivialities of life", "(12:54, 13:57)"], 
# ["go to work", "(16:03, 21:23)"]
# ]

# Example 4:
# [
# ["go to sleep", "(00:00, 12:14)"], 
# ["eat", "(13:59, 15:29)"], 
# ["go home", "(16:28, 17:15)"], 
# ["go to work", "(17:20, 22:05)"], 
# ["go to sleep", "(22:23, 23:59)"]
# ]

# Example 5:
# [
# ["go to sleep", "(00:00, 07:00)"], 
# ["go to work", "(08:11, 16:31)"], 
# ["go home", "(17:47, 18:50)"], 
# ["do sports", "(19:54, 20:17)"], 
# ["go home", "(20:18, 20:41)"], 
# ["do sports", "(22:06, 22:18)"], 
# ["go to sleep", "(22:40, 23:59)"]
# ]

# Example 6:
# [
# ["go to sleep", "(00:00, 08:31)"], 
# ["go to work", "(10:59, 18:29)"], 
# ["go to sleep", "(20:47, 23:59)"]
# ]

# Example 7:
# [
# ["go home", "(00:00, 00:05)"], 
# ["leisure or entertainment", "(00:12, 00:42)"], 
# ["handle the trivialities of life", "(10:16, 19:36)"]
# ]

# Example 8:
# [
# ["go to sleep", "(00:00, 07:56)"], 
# ["go to work", "(09:28, 18:02)"], 
# ["go home", "(19:18, 22:54)"]
# ]

    return globalInfo

def getBasicData7_0510(personBasicInfo, day, N):
    '''
    简单COT+profile
    '''
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)
   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

The specific requirements of your assignment are as follows:
1. Generate your own schedule for {}.
2. You need to consider how your character attributes relate to your behavior.
3. I want you to answer the reasons for each event, including why you want to do it and how to arrange the time.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.
""".format(personDescription, day)},
    ]

    return globalInfo


def getBasicData8(personBasicInfo, day, N):
    '''
    0122: 去除了anchor机制,使得生成更加自由
    0124: 加入example强调回家的作用.
    0125: 恢复原始的轮询式提问方式,在global中提问中注意改变全局的提问.
    0128: 加入COT机制,试图增强性能. But I doubt that.事实证明确实是有效果vc 的.
    0128: 移除了人物profile.做消融实验看看效果.就是不做role play了.
    '''

    personDescription = """You are a human behavior generator."""

   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}
I need you to give me a schedule for a person's day in {}.

Note that: 
1. All times are in 24-hour format.
2. I want to limit the total number of events in the day to {}. It means you have to schedule {} number of things, no more and no less. I hope you can make every decision based on this limit.
3. The generated schedule must start at 0:00 and end at 24:00. Don't let the schedule spill over into the next day.
4. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
5. I'll ask you step by step what to do, and you just have to decide what to do next each time.
""".format(personDescription, day, N, N)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo


'''
0408 7天生成,需要灵活加入时间信息
'''
def getBasicData9(personBasicInfo, day, N):
     # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
   
    globalInfo = [
        {"role": "system","content":
"""{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior.
2. I want you to answer the reasons for each event.
3. I hope you can take into account the diversity of human life in your answer and avoid mechanical repetition.
4. I want to limit your total number of events today to {}. I hope you can make every decision based on this limit.
5. I will provide you with your schedule for the previous few days of this week for your reference.

Note that: 
1. All times are in 24-hour format.
2. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples of a single day for reference. For each example I will give a portrait of the corresponding character and the reasons for each arrangement.

Example 1:
This is the schedule of a day for a coder who works at an Internet company.
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
This is the schedule of a day for a salesperson at a shopping mall.
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
This is the schedule of a day for a manager who is about to retire.
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
This is the schedule of a day for a lawyer who suffers a medical emergency in the morning.
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
This is an architect's schedule on a Sunday.
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
This is the schedule of a day for a customer service specialist.
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
This is the schedule of a day for a wedding event planner.
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
This is the schedule of a day for a high school teacher in Saturday.
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, day, N)},
    ]

    return globalInfo



'''
0408去除了用户的画像做实验检测效果
'''
def getBasicData10(personBasicInfo, day, N):
    # age, education, gender, consumption, occupation = personBasicInfo.split('-')
    # ageDescription = "" if age == '0' else "Age: {}; ".format(age)
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {};".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
   
    globalInfo = [
        # 首先交待任务,申明轨迹点的一般形式.
        {"role": "system","content":
"""{}

Now I want you to generate your own schedule for today.(today is {}).
The specific requirements of the task are as follows:
1. You need to consider how your character attributes relate to your behavior.
2. I want to limit your total number of events in a day to {}. I hope you can make every decision based on this limit.
3. I want you to answer the reasons for each event.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time.

Here are some examples for reference. For each example I will give the reasons for each arrangement.

Example 1:
[
["go to sleep", "(00:00, 11:11)"], (Reason: Sleep is the first thing every day.)
["go to work", "(12:08, 12:24)"], (Reason: Work for a while after sleep. This person's working hours are relatively free, there is no fixed start and finish time.) 
["eat", "(12:35, 13:01)"], (Reason: It's noon after work. Go get something to eat.)
["go to work", "(13:15, 20:07)"],   (Reason: After lunch, the afternoon and evening are the main working hours. And he works so little in the morning that he need to work more in the afternoon and evening. So this period of work can be very long.)
["go to sleep", "(21:03, 23:59)"]  (Reason: It was already 9pm when he got off work, and it is time to go home and rest.)
]

Example 2:
[
["go to sleep", "(00:00, 08:25)"], (Reason: Of course the first thing of the day is to go to bed.)
["go to work", "(09:01, 19:18)"], (Reason: Generally, the business hours of shopping malls are from 9 am to 7 pm, so she works in the mall during this time and will not go anywhere else.)
["go home", "(20:54, 23:59)"], (Reason: It's almost 9pm after getting off work. Just go home and rest at home.)
]

Example 3:
[
["go to sleep", "(00:00, 06:04)"], (Reason: He is used to getting up early, so he got up around 6 o'clock in the morning.)
["eat", "(08:11, 10:28)"], (Reason: He has the habit of having morning tea after getting up and enjoys the time of slowly enjoying delicious food in the morning.)
["go home", "(12:26, 13:06)"], (Reason: After breakfast outside, take a walk for a while, and then go home at noon.)
["excursion", "(13:34, 13:53)"], (Reason: He stays at home most of the morning, so he decides to go out for a while in the afternoon.)
["go to work", "(14:46, 16:19)"], (Reason: Although life is already relatively leisurely, he still has work to do, so he has to go to the company to work for a while in the afternoon.)
]

Example 4:
[
["go to sleep", "(00:00, 09:36)"], (Reason: Sleep until 9:30 in the morning. Lawyers' working hours are generally around 10 o'clock.)
["medical treatment", "(11:44, 12:03)"], (Reason: He suddenly felt unwell at noon, so he went to the hospital for treatment.)
["go to work", "(12:27, 14:56)"], (Reason: After seeing the doctor, the doctor said there was nothing serious, so he continued to return to the company to work for a while.)
["go to sleep", "(17:05, 23:59)"], (Reason: Since he was not feeling well, he got off work relatively early and went home to rest at 5 p.m.)
]

Example 5:
[
["go to sleep", "(00:00, 06:20)"], (Reason: The first thing is of course to sleep.)
["handle the trivialities of life", "(07:18, 07:32)"], (Reason: After getting up, he first dealt with the trivial matters in life that had not been resolved during the week.)
["leisure or entertainment", "(07:38, 17:27)"], (Reason: Since today was Sunday, he didn't have to work, so he decided to go out and have fun.)
["handle the trivialities of life", "(18:22, 19:11)"], (Reason: After coming back in the evening, he would take care of some chores again.)
 ["go to sleep", "(20:51, 23:59)"] (Reason: Since tomorrow is Monday, go to bed early to better prepare for the new week.)
]

Example 6:
[
[go to work, (9:21, 16:56)], (Reason: Work dominated the day and was the main event of the day.)
[go home, (20:00, 23:59)], (Reason: After a day's work and some proper relaxation, he finally got home at 8 o 'clock.)
]

Example 7:
[
[go to work, (11:21, 20:56)], (Reason: As a wedding planner, her main working hours are from noon to evening.)
[go home, (23:10, 23:30)], (Reason: After finishing the evening's work, she went home to rest.)
[handle the trivialities of life, (23:30, 23:59)], (Reason: Before she goes to bed, she takes care of the trivial things in her life.)
]

Example 8:
[
[eat, (06:11, 7:28)], (Reason: He has a good habit: have breakfast first after getting up in the morning.)
[handle the trivialities of life, (07:48, 08:32)],  (Reason: After breakfast, he usually goes out to deal with some life matters.)
[go home, (9:00, 11:00)], (Reason: After finishing all the things, go home.)
[medical treatment, (13:44, 17:03)], (Reason: Today is Saturday and he doesn't have to work, so he decides to go to the hospital to check on his body and some recent ailments.)
[go home, (19:00, 23:59)], (Reason: After seeing the doctor in the afternoon, he goes home in the evening.)
]

As shown in the example, a day's planning always starts with "go to sleep" and ends with "go to sleep" or "go home".
""".format(personDescription, day, N)},
    ]
    # print('Basic Data perpare done!')

    return globalInfo



if __name__ == '__main__':

    # print(hm2float('3:59'))
    setOpenAi()  # set api key and proxy
    pass








"""

Your task is to generate a day's worth of trajectory information.
I will provide you with previous historical trajectory, tell you what to do next, and give you some candidate locations to choose from.
In addition to the names of the locations, I will also provide distance information.
You need to choose one from these candidate locations.

Example:
An example of a single selection is as follows:

Question:
Up until now, your trajectory is as follows:
['sleep', ('金台北街小区', 700039704), '(0:00, 7:30)']
['have breakfast', ('人民日报食堂', 700866819), '(7:37, 8:07)']
Your next agenda is to go exercise. And nearby venues related to this activity are as follows:
('金石健身房', 500, "121215151"), ('碧波游泳馆', 1000, "24242552454"), ('第三中学足球场', 2000, "45452452454")

In each tuple, the first element is the name of the POI,
the second element is the distance from the current location (in meters),
and the third element is the ID of the POI.
When making a selection, you need to choose a complete tuple.

Answer:
('金石健身房', 500, "121215151")

"""