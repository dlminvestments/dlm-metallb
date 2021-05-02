from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.HyperbolicTangent import HyperbolicTanget
from neuralnetwork.Backpropagation import Backpropagation

hyperbolicTangent = HyperbolicTangent()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, hyperbolicTangent)

backpropagation = Backpropagation(feedForward,0.7,0.3,0.001)

trainingSet = [
                    [-1,-1,-1],
                    [-1,1,1],
                    [1,-1,1],
                    [1,1,-1]
                ];

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([-1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([-1,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
