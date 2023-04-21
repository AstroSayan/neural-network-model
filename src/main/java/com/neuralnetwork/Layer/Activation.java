package com.neuralnetwork.Layer;

import java.util.function.Function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Activation implements ILayer {
	private Function<INDArray, INDArray> activationFunction;
	private Function<INDArray, INDArray> activationFunctionDerivative;
	private INDArray input;
	private INDArray output;

	public Activation(Function<INDArray, INDArray> activationFunction,
			Function<INDArray, INDArray> activationFunctionDerivative) {
		super();
		this.activationFunction = activationFunction;
		this.activationFunctionDerivative = activationFunctionDerivative;
	}

	@Override
	public INDArray forward(INDArray input) {
		this.input = input;
		this.output = this.activationFunction.apply(input);
		return this.output;
	}

	@Override
	public INDArray backward(INDArray outputGradient, double learningRate) {
		return Nd4j.matmul(outputGradient, this.activationFunctionDerivative.apply(this.input));
	}

}
