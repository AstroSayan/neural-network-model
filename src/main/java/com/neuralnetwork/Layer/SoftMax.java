package com.neuralnetwork.Layer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SoftMax implements ILayer {
	private INDArray output;

	@Override
	public INDArray forward(INDArray input) {
		INDArray mat = Transforms.exp(input);
		this.output = mat.div(mat.sumNumber());
		return this.output;
	}

	@Override
	public INDArray backward(INDArray outputGradient, double learningRate) {
		long outputSize = this.output.length();
		INDArray mat = Nd4j.eye(outputSize).sub(this.output.transpose());
		mat = mat.muli(this.output);
		return Nd4j.matmul(mat, outputGradient);
	}
	
}
