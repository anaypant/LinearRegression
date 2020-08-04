import linearregression


x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
iterations = 1000

reg = linearregression.linearregression(x, y, iterations)
reg.train()