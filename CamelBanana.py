dp = [[-1 for i in range(3001)] for j in range(1001)]
def main(A, B, C):
	if B <= A:
		return 0
	if B <= C:
		return B - A
	if dp[A][B] != -1:
		return dp[A][B]
	maxCount = 0
	x=2*B//C
	if B % C==0:
		x = x-1
	else:
		x = x+1
	for i in range(1,A+1):
		curCount = main(A - i, B - x * i, C)
		if curCount > maxCount:
			maxCount = curCount
			dp[A][B] = maxCount
	return maxCount
A = 1000
B = 3000
C = 1000
print(main(A, B, C))