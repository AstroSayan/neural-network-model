package com.neuralnetwork.Layer;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ILayer {
	INDArray forward(INDArray input);

	INDArray backward(INDArray outputGradient, double learningRate);
}
