#import pandas as pd
with open('/Users/sriramreddy/Downloads/ML/2/w_normalize_1.csv') as f:
    lines = f.read().splitlines()
with open('/Users/sriramreddy/Downloads/ML/2/features_type.txt') as f:
    features = f.read().splitlines()
dict ={}
print(len(lines))
for i in range(len(lines)):
    dict[lines[i]]=i
lines.sort()
print("Bottom features, value")
for i in range(10):
    print(features[dict[lines[i]]],",",lines[i])
print("Top features,value")
for i in range(len(lines)-10,len(lines),1):
    print(features[dict[lines[i]]],",",lines[i])
