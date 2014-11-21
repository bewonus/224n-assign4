package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, Wout, U, b1, b2; // word-vector matrix, weight matrix, weight matrix out

	public int windowSize, wordSize, hiddenSize; // C, n, H
  public double learningRate; // alpha
  public int numLabels = 5;

  private static final String START = "<s>";
  private static final String END = "</s>";
  private static final String UNK = "UUUNKKK";

  /**
   *
   * @param _windowSize = C
   * @param _hiddenSize = H
   * @param _lr = the learning rate (alpha)
   */
	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		windowSize = _windowSize;
    hiddenSize = _hiddenSize;
    learningRate = _lr;
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){

    // TODO: randomly initialize these guys later...

    wordSize = FeatureFactory.allVecs.numRows(); // == 50

    // word-vector matrix (i.e. the x's)
    L = new SimpleMatrix(FeatureFactory.allVecs);

    // initialize with bias inside as the last column (plus 1)
    W = new SimpleMatrix(hiddenSize, windowSize * wordSize + 1);
    U = new SimpleMatrix(numLabels, hiddenSize + 1);
	}

  /**
   * Create the concatenated vector x
   * @param indices into the matrix L
   * @return
   */
  private SimpleMatrix makeX(List<Integer> indices) {
    SimpleMatrix x = new SimpleMatrix(wordSize * windowSize + 1, 1);
    int counter = 0;
    for (int col : indices) {
      System.out.println(indices);
      for (int row = 0; row < wordSize; row++) {
        x.set(counter * wordSize + row, 0, L.get(row, col));
      }
      counter++;
    }
    x.set(counter * wordSize, 0, 1);
    return x;
  }


  private List<Integer> createWindow(List<Datum> _trainData, int index) {

    List<Integer> indices = new ArrayList<Integer>(windowSize);
    Boolean startSeen = false;
    Boolean endSeen = false;

    for (int i = index; i >= index - (windowSize - 1)/2; i--) {
      if (startSeen || _trainData.get(i).word.equals(START)) {
        startSeen = true;
        indices.add(0, FeatureFactory.wordToNum.get(START));
      }else {
        Integer nextIndex = FeatureFactory.wordToNum.get(_trainData.get(i).word.toLowerCase());
        if (nextIndex == null) {
          indices.add(0, FeatureFactory.wordToNum.get(UNK));
        } else {
          indices.add(0, nextIndex);
        }
      }
    }

    for (int i = index+1; i <= index + (windowSize - 1)/2; i++) {
      if (endSeen || _trainData.get(i).word.equals(END)) {
        endSeen = true;
        indices.add(FeatureFactory.wordToNum.get(END));
      }else {
        Integer nextIndex = FeatureFactory.wordToNum.get(_trainData.get(i).word.toLowerCase());
        if (nextIndex == null) {
          indices.add(FeatureFactory.wordToNum.get(UNK));
        } else {
          indices.add(nextIndex);
        }
      }
    }

    return indices;
  }

  private SimpleMatrix fTransform(SimpleMatrix x) {
    SimpleMatrix toReturn = new SimpleMatrix(x.numRows() +1 , x.numCols());
    for(int i = 0 ; i < x.numRows() ; i++){
      double val = Math.tanh(x.get(i,0));
      toReturn.set(i, 0, val);
    }
    toReturn.set(toReturn.numRows()-1,0,1);
    return toReturn;
  }

  private SimpleMatrix gTransform(SimpleMatrix x) {
    double norm = 0;

    for(int i = 0 ; i < x.numRows() ; i++){
      double exp = Math.exp(x.get(i,0));
      norm += exp;
      x.set(i,0,exp);
    }
    x = x.divide(norm);
    return x;
  }

  private SimpleMatrix makeWout(SimpleMatrix x) {
      SimpleMatrix afterF = fTransform(W.mult(x));
      return gTransform(U.mult(afterF));
  }

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> _trainData ){

    for (int i = 0; i < _trainData.size(); i++) {

      if (_trainData.get(i).word.equals(START) || _trainData.get(i).word.equals(END)) {
        continue;
      }

      List<Integer> windowIndices = createWindow(_trainData, i);

      SimpleMatrix x = makeX(windowIndices);

      Wout = makeWout(x);
      System.out.println(Wout);
    }

	}

	
	public void test(List<Datum> testData){
		// TODO
		}
	
}
