package com.neuralnetwork.Layer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dense implements ILayer {
	private INDArray weights;
	private INDArray bias;
	private INDArray input;
	private INDArray output;

	public Dense(int inputSize, int outputSize) {
		super();
		this.weights = Nd4j.randn(new int[] { outputSize, inputSize });
		this.bias = Nd4j.randn(new int[] { outputSize, 1 });
		;
	}

	@Override
	public INDArray forward(INDArray input) {
		this.input = input;
		this.output = Nd4j.matmul(this.weights, this.input).addi(this.bias);
		return this.output;
	}

	@Override
	public INDArray backward(INDArray outputGradient, double learningRate) {
		INDArray weightsGradient = Nd4j.matmul(outputGradient, this.input.transpose());
		INDArray inputGradient = Nd4j.matmul(this.weights.transpose(), outputGradient);
		this.weights = this.weights.sub(weightsGradient.mul(learningRate));
		this.bias = this.bias.sub(outputGradient.mul(learningRate));
		return inputGradient;
	}

}
