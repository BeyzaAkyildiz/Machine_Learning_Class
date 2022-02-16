import random

isaretli=random.randint(0, 2)
secilmis=random.randint(0, 2)
temp=0
sonuc=0
for x in range(1000):
    isaretli = random.randint(0, 2)
    secilmis = random.randint(0, 2)
    if(isaretli==secilmis):
        temp+=1

sonuc=(temp*100)/1000
print(sonuc)

tempp=0
hesap=0
for y in range(10000):

    yeni=0
    digeri=0
    isaretlii = random.randint(0, 2)
    secilmiss = random.randint(0, 2)
    for z in range(3):

        if((z != secilmiss) and (z != isaretlii)):
            yeni=z
    for m in range(3):
        if ((m != secilmiss) and (m != yeni)):
            digeri=m




    if(isaretlii==digeri):
        tempp+=1
hesap=(tempp*100)/10000
print(hesap)



