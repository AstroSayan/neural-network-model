package com.neuralnetwork.matrixops;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class MatrixOps {

	public static void main(String[] args) {
		INDArray arr1 = Nd4j.create(new double[][] { { 1, 4 }, { 5, 6 } });
		INDArray arr2 = Nd4j.create(new double[][] { { 2, 4 }, { 5, 2 } });
		// get row
		INDArray get = arr2.get(NDArrayIndex.indices(1));
		System.out.println(get);
		System.out.println();
		// dot product
		INDArray arr = Nd4j.matmul(arr1, arr2);
		System.out.println(arr);
		// scalar multiplication
		INDArray scalMulArr = arr2.muli(2);
		System.out.println(scalMulArr);
		// transpose
		INDArray transArr = arr1.transpose();
		System.out.println(transArr);
		// generate random matrix (std. normal dist.)
		INDArray randMat = Nd4j.randn(new int[] { 3, 4 });
		System.out.println(randMat);
		// tanh, exp matrix
		INDArray ops = Transforms.tanh(arr1);
		System.out.println(ops);
		ops = Transforms.exp(arr1);
		System.out.println(ops);
		// identity matrix
		ops = Nd4j.eye(3);
		System.out.println(ops);
		// sum of each col keeping dimension
		ops = arr1.sum(true, 0);
		System.out.println(ops);
		// sum of each row keeping dimension
		ops = arr1.sum(true, 1);
		System.out.println(ops);
		// sum of all elems in matrix
		Number sumAns = arr1.sumNumber();
		System.out.println(sumAns);
		// cumulative ops (1 - tanh(X) ** 2)
		ops = Transforms.tanh(arr1);
		ops = ops.muli(ops); // squaring the matrix
		ops = ops.subi(1).muli(-1);
		System.out.println(ops);
		
		INDArray X = Nd4j.create(new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });//.reshape(new int[] {4, 2, 1});
		INDArray Y = Nd4j.create(new double[][] { { 0 }, { 1 }, { 1 }, { 0 } }); //.reshape(new int[] {4, 1, 1});
		System.out.println(X.getRow(0).reshape(2, 1));
		System.out.println(Y.getRow(0).reshape(1, 1));
	}

}
