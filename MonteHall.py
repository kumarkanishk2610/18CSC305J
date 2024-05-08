import random
def monty_hall_simulations(n):
    wins = 0
    for _ in range(n):
        prize_door = random.randint(1, 3)
        chosen_door = random.randint(1, 3)
        revealed_door = random.choice([door for door in range(1, 4) if door != chosen_door and door != prize_door])
        chosen_door = [door for door in range(1, 4) if door != chosen_door and door != revealed_door][0]
        if chosen_door == prize_door:
            wins += 1
    return wins / n
n = int(input("Enter number of simulations: "))
average_probability = monty_hall_simulations(n)
print("Average probability of winning if switching doors:", average_probability)