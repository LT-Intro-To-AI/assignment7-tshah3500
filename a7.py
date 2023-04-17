from neural import NeuralNet


print("------Training X or model------")

x_or_trainingdata = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], [0])
]

xorn = NeuralNet(2, 1, 1)
xorn.train(x_or_trainingdata)
print("")
print(xorn.test_with_expected(x_or_trainingdata))


print("\n------Training Voter Opinion------")

voting_training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0])
]

von = NeuralNet(5, 25, 1)
von.train(voting_training_data)

test_data = [[1, 1, 1, .1, .1], [.5, .2, .1, .7, .7], [.8, .3, .3, .3, .8], [.8, .3, .3, .8, .3], [.9, .8, .8, .3, .6]]
print("\n------Testing Voter Data------")
print(von.test(test_data))


