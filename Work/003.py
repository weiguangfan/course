from transformers import pipeline


generator = pipeline('text-generation')
# print(generator('In this course, we will teach you how to'))
# print(generator('In this course, we will teach you how to', max_length=30, num_return_sequences=2, ))
generator2 = pipeline('text-generation', model='distilgpt2')
print(generator2('In this course, we will teach you how to', max_length=100, num_return_sequences=3))