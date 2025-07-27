import pickle
import numpy
from utils import *


def hm2float(hmstr):
    h,m = hmstr.split(':')
    h = int(h)
    m = int(m)
    return h+m/60

def getRatingPrompts():
    systemInfo = [
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


def getGlobalPrompts(personBasicInfo, day, N, GPT_MODEL):
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)

    AutoAttitude = [{"role": "user", "content":"""What activities does a person with a character profile of {}{}{}{} tend to prefer and engage in?""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)}]
    attitude = askChatGPT(AutoAttitude, model = GPT_MODEL, temperature = 1)  # ask LLM to generate the preferences

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
    

    anchors = askChatGPT(AutoAnchor, model = GPT_MODEL, temperature = 1) 
    # print('generate anchors:\n')

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
        {"role": "system","content":
"""{}

Your preferences are as follows:
{}

You have some habits or routines as follows:
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
""".format(personDescription, attitude, anchors, day, N)},
    ]
    
    return globalInfo


def getBasicData_0509_CM(personBasicInfo, day, N, GPT_MODEL):

    education, gender, consumption= personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}""".format(genderDescription, educationDescription, consumptionDescription)
 
    AutoAnchor = [{"role": "system","content":
"""Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.
Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""},
                  {"role": "user","content":
"""The person's basic information is as follows: 
{}{}{}
Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone and organize your answers in 1.xxx 2.xxx 3.xxx format."""
                    .format(genderDescription, educationDescription, consumptionDescription)}]
   
    

    anchors = askChatGPT(AutoAnchor, model = GPT_MODEL, temperature = 1)  

    day = day if day != 'Saturday' and day != 'Sunday' else day + '. It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time.'
    
    globalInfo = [
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
""".format(personDescription, anchors, day, N)},
    ]

    return globalInfo


def basicGlobalInfo(personBasicInfo, day, N):
    education, gender, consumption, occupation = personBasicInfo.split('-')
    genderDescription = "" if gender == 'uncertain' else "Gender: {}; ".format(gender)
    educationDescription = "" if education == 'uncertain' else "Education: {}; ".format(education)
    consumptionDescription = "" if consumption == 'uncertain' else "Consumption level: {}; ".format(consumption)
    occupationDescription = "" if occupation == 'uncertain' else "Occupation or Job Content: {}; ".format(occupation)

    personDescription = """You are a person and your basic information is as follows:
{}{}{}{}""".format(genderDescription, educationDescription, consumptionDescription, occupationDescription)
   
    globalInfo = [
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
]""".format(personDescription, day, N, N)},
    ]
    return globalInfo

