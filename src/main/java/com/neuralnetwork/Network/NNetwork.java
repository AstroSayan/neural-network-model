package com.neuralnetwork.Network;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.neuralnetwork.Layer.ILayer;

public class NNetwork {
	private static List<ILayer> layers = new ArrayList<ILayer>();
	private static INDArray output;

	public static void addLayer(ILayer layer) {
		layers.add(layer);
	}
	
	private static List<ILayer> reversedLayers() {
		List<ILayer> reverseLayers = new ArrayList<ILayer>(layers);
		Collections.reverse(reverseLayers);
		return reverseLayers;
	}

	public static INDArray predict(INDArray input) {
		output = input;
		layers.forEach(layer -> {
			output = layer.forward(output);
		});
		return output;
	}

	public static void train(BiFunction<INDArray, INDArray, Number> lossFunc,
			BiFunction<INDArray, INDArray, INDArray> lossDerivativeFunc, INDArray xTrain, INDArray yTrain, long epochs,
			double learningRate, boolean verbose) {
		for (long e = 0; e < epochs; e++) {
			double error = 0;
			int xColSize = xTrain.columns();
			int yColSize = yTrain.columns();
			for (long i = 0; i < xTrain.rows(); i++) {
				INDArray x_Train = xTrain.getRow(i).reshape(xColSize, 1);
				INDArray y_Train = yTrain.getRow(i).reshape(yColSize, 1);

				output = predict(x_Train);
				error += (double) lossFunc.apply(y_Train, output);

				INDArray grad = lossDerivativeFunc.apply(y_Train, output);
				for (ILayer layer : reversedLayers()) {
					grad = layer.backward(grad, learningRate);
				}
			}
			error /= xTrain.length();
			if (verbose) {
				System.out.println(e + 1 + "/" + epochs + ", error= " + error);
			}
		}
	}

}
