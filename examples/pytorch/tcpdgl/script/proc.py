import os


f = open('/home/chijj/tcpdgl/examples/pytorch/tcpdgl/script/raw.txt', 'r')
lline = f.readlines()
f.close()
p=0
hits = {}
epoch = set()
while p < len(lline):
    line = lline[p]
    p += 1
    if (not 'Hits' in line):
        continue
    k = int(line.split('@')[-1])
    line = lline[p]
    p += 1
    if 'Epoch' in line:
        epo = int(line.split('Epoch: ')[1].split(',')[0])
        train = float(line.split('Train: ')[1].split('%')[0])
        valid = float(line.split('Valid: ')[1].split('%')[0])
        test = float(line.split('Test: ')[1].split('%')[0])
        
        if k not in hits:
            hits[k] = {}
        if epo not in hits[k]:
            hits[k][epo] = {'train': [], 'valid': [], 'test': []}
        hits[k][epo]['train'].append(train)
        hits[k][epo]['valid'].append(valid)
        hits[k][epo]['test'].append(test)
        epoch.add(epo)


with open('proc.txt', 'w+') as f:
    for k in hits:
        for epo in epoch:
            train = sum(hits[k][epo]['train']) / len(hits[k][epo]['train'])
            valid = sum(hits[k][epo]['valid']) / len(hits[k][epo]['valid'])
            test = sum(hits[k][epo]['test']) / len(hits[k][epo]['test'])
            if epo % 20 == 0:
                f.write('Hits@%d Epoch %d:\t %.2f , %.2f , %.2f\n' % (k, epo, train, valid, test))
        f.write('\n')

    f.close()