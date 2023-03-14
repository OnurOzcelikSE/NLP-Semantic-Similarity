import spacy

# loading md language model
nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print("\n---md language model---\n")

print(f"Cat/Monkey:    {word1.similarity(word2)}")
print(f"Banana/Monkey: {word3.similarity(word2)}")
print(f"Banana/Cat:    {word3.similarity(word1)}")

print("\n")

# my own example to compare 
word4 = nlp("guitar")
word5 = nlp("trumpet")
word6 = nlp("electricity")

print(f"Guitar/Trumpet:       {word4.similarity(word5)}")
print(f"Trumpet/Electricty:   {word6.similarity(word5)}")
print(f"Electricity/Guitar:   {word6.similarity(word4)}")

print("\n")

# Working With Vectors
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Working With Sentences
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ",  similarity)

"""
Note:
Cat/Monkey:    0.5929930274321619
Banana/Monkey: 0.40415016164997786
Banana/Cat:    0.22358825939615987

The similarities between these three words are in the descending order,
Cat/Monkey, has the highest similarity because both of them are animals
Banana/Monkey has the second highest similarity because banana is monkey's favourite food
Banana/Cat has the lowest similarity, because banana is not the favourite food of cat

Guitar/Trumpet:       0.6845285039266278
Trumpet/Electricty:  -0.06952536972482753
Electricity/Guitar:   0.04514051615805639

Guitar/Trumpet has the most similar words, because they both are instruments
Electricty/Guitar is more similar than Trumpet/Electricity because guitar can be used 
with electric as electric guitar by using amplifiers, 
but there is no relation between trumpet and electricity

cat cat        1.0
cat apple      0.2036806046962738   
cat monkey     0.5929930210113525  
cat banana     0.2235882580280304  
apple cat      0.2036806046962738   

apple apple    1.0
apple monkey   0.2342509925365448
apple banana   0.6646699905395508
monkey cat     0.5929930210113525  
monkey apple   0.2342509925365448

monkey monkey  1.0
monkey banana  0.4041501581668854
banana cat     0.2235882580280304
banana apple   0.6646699905395508
banana monkey  0.4041501581668854

banana banana  1.0

where did my dog go -         0.630065230699739
Hello, there is my car -      0.8033180111627156
I've lost my car in my car -  0.6787541571030323
I'd like my boat back -       0.5624940517078084
I will name my dog Diana -    0.6491444739190607
"""
print("\n---sm languange model---\n")

# loading sm language model
nlp = spacy.load("en_core_web_sm")

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Working With Vectors
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

# Working With Sentences
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ",  similarity)

"""
Cat/Monkey:    0.6770565478895127
Banana/Monkey: 0.7276309976205778
Banana/Cat:    0.6806929391210822

Due to this library: 
Banana/Monkey is the most similar comparison,
Banana/Cat has the second highest similarity,
Cat/Monkey has the lowest similarity,

md model prioritizes type of items more than sm.
because sm don't ship with word vectors and only use context-sensitive tensors.

cat cat        1.0
cat apple      0.7018378973007202
cat monkey     0.6455236077308655
cat banana     0.2214718759059906
apple cat      0.7018378973007202

apple apple    1.0
apple monkey   0.7389943599700928
apple banana   0.36197030544281006
monkey cat     0.6455236077308655
monkey apple   0.7389943599700928

monkey monkey  1.0
monkey banana  0.4232020080089569
banana cat     0.2214718759059906
banana apple   0.36197030544281006
banana monkey  0.4232020080089569

banana banana 1.0

where did my dog go -         0.4043351553824302
Hello, there is my car -      0.5648939507997681
I've lost my car in my car -  0.548028403302901
I'd like my boat back -       0.3007499696891998
I will name my dog Diana -    0.3904074310483232
"""