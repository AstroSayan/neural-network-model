package com.neuralnetwork.ActivationFuncs;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ActivationFuncs {

	public static INDArray tanh(INDArray x) {
		return Transforms.tanh(x);
	}

	public static INDArray tanhDerivative(INDArray x) {
		INDArray mat = tanh(x);
		mat = mat.muli(mat);
		return mat.subi(1).muli(-1);
	}

	public static INDArray sigmoid(INDArray x) {
		return Transforms.sigmoid(x);
	}

	public static INDArray sigmoidDerivative(INDArray x) {
		INDArray mat1 = sigmoid(x);
		INDArray mat2 = mat1.subi(1).muli(-1);
		return mat1.muli(mat2);
	}
}
