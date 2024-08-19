import os
import openai

openai.api_key = ''
def get_struct(caption):
    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {
                        'role': 'user',
                        'content':f' I want to predict that the duration of speech based on the content and you need to give me the results in the following format:\
                        Question: Three members of this shift separately took this opportunity to visit the Cellar Coffee House.\
                        Answer: It includes 15 words, it may cost 5 to 6 seconds.\
                        Question: This Allotment Division will consider all of the recommendations submitted to it. \
                        Answer: It includes 12 words, it may cost 4 to 5 seconds. \
                        Question: has been far less than in any previous, comparable period. \
                        Answer: It includes 10 words, it may cost 3 to 4 seconds. \
                        Question: They were glancing about with eager eyes. \
                        Answer: It includes 7 words, it may cost 2 to 3 seconds. \
                        Question: I do not think so. \
                        Answer: It includes 5 words, it may cost 1 to 2 seconds. \
                        You should consider the pronunciation of each word. Some words may need more time to pronunciate. In general, if a word includes more letter, it costs more time to read.  \
                        In summary, you should know how many words in the sentence, then consider how long it will cost to read it.\
                        Question: {caption} \
                        Answer:',
                    },
                ]
            )
    return completion.choices[0].message.content


print(get_struct('Generate a poem'))
