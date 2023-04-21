package com.neuralnetwork.LossFuncs;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class LossFuncs {
	public static Number mse(INDArray yTrue, INDArray yPred) {
		INDArray err = yTrue.sub(yPred);
		INDArray squaredErr = err.mul(err);
		return squaredErr.meanNumber();
	}
	
	public static INDArray mseDerivative(INDArray yTrue, INDArray yPred) {
		INDArray err = yPred.sub(yTrue);
		err = err.mul(2);
		return err.div(yTrue.length());
	}
	
	public static Number binaryCrossEntropy(INDArray yTrue, INDArray yPred) {
		INDArray mat1 = Transforms.log(yPred.sub(1).mul(-1));
		INDArray mat2 = yTrue.sub(1).mul(-1);
		mat1 = mat2.mul(mat1);
		INDArray mat3 = yTrue.mul(-1);
		mat3 = mat3.mul(Transforms.log(yPred));
		INDArray err = mat3.sub(mat1);
		return err.meanNumber();
	}
	
	public static INDArray binaryCrossEntropyDerivative(INDArray yTrue, INDArray yPred) {
		INDArray mat1 = yPred.sub(1).mul(-1);
		INDArray mat2 = yTrue.sub(1).mul(-1);
		mat1 = mat2.div(mat1);
		mat1 = mat1.sub(yTrue.div(yPred));
		return mat1.div(yTrue.length());
	}
}
