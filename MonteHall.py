import random
def main(n):
    wins=0
    for i in range (n):
        prizedoor=random.randint(1,3)
        chosendoor=random.randint(1,3)
        revealeddoor=random.randint(1,3)
        while(revealeddoor!=chosendoor and revealeddoor!=prizedoor):
            revealeddoor=random.randint(1,3)
        while(chosendoor!=revealeddoor):
            chosendoor=random.randint(1,3)
        if chosendoor==prizedoor:
            wins+=1
    print (wins/n)
main(100000)