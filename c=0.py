c=0

for k in data.diagnosis:
    if k == 'M':
        data.diagnosis[c]=1
    else:
        data.diagnosis[c]=0
    c=c+1