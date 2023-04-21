package com.neuralnetwork;

import java.util.function.BiFunction;
import java.util.function.Function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.neuralnetwork.ActivationFuncs.ActivationFuncs;
import com.neuralnetwork.Layer.Activation;
import com.neuralnetwork.Layer.Dense;
import com.neuralnetwork.LossFuncs.LossFuncs;
import com.neuralnetwork.Network.NNetwork;

public class NNTest {

	public static void main(String[] args) {
		INDArray X = Nd4j.create(new float[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
				//.reshape(new int[] { 4, 2, 1 });
		INDArray Y = Nd4j.create(new float[][] { { 0 }, { 1 }, { 1 }, { 0 } }); //.reshape(new int[] { 4, 1, 1 });

		Function<INDArray, INDArray> activationFunction = (x) -> ActivationFuncs.tanh(x);
		Function<INDArray, INDArray> activationFunctionDerivative = (x) -> ActivationFuncs.tanhDerivative(x);
		Activation Tanh = new Activation(activationFunction, activationFunctionDerivative);

		BiFunction<INDArray, INDArray, Number> lossFunc = (yTrue, yPred) -> LossFuncs.mse(yTrue, yPred);
		BiFunction<INDArray, INDArray, INDArray> lossFuncDerivative = (yTrue, yPred) -> LossFuncs.mseDerivative(yTrue,
				yPred);

		NNetwork.addLayer(new Dense(2, 5));
		NNetwork.addLayer(Tanh);
		NNetwork.addLayer(new Dense(5, 1));
		NNetwork.addLayer(Tanh);

		NNetwork.train(lossFunc, lossFuncDerivative, X, Y, 1000, 0.01, true);

	}

}
