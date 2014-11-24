package cs224n.deep;

import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

  protected SimpleMatrix L, W, Wout, U, b1, b2; // word-vector matrix, weight matrix, weight matrix out

  private List<String> labels = new ArrayList<String>(Arrays.asList("O", "LOC", "MISC", "ORG", "PER"));

  public int windowSize, wordSize, hiddenSize; // C, n, H
  public double learningRate; // alpha
  public int numLabels = labels.size();
  private double lambda;

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
  public void initWeights() {
    wordSize = FeatureFactory.allVecs.numRows(); // == 50

    // word-vector matrix (i.e. the x's)
    L = new SimpleMatrix(FeatureFactory.allVecs);

    // initialize with bias inside as the last column (plus 1)
//    W = new SimpleMatrix(hiddenSize, windowSize * wordSize + 1);
//    U = new SimpleMatrix(numLabels, hiddenSize + 1);

    // set params for random initialization
    double eW = Math.sqrt(6.0) / Math.sqrt(hiddenSize + windowSize * wordSize);
    double eU = Math.sqrt(6.0) / Math.sqrt(numLabels + hiddenSize);

    // randomly initialize W and U
    W = SimpleMatrix.random(hiddenSize, windowSize * wordSize + 1, -eW, eW, new java.util.Random());
    U = SimpleMatrix.random(numLabels, hiddenSize + 1, -eU, eU, new java.util.Random());

    // initialize bias terms to 0
    for (int i = 0; i < W.numRows(); i++) {
      W.set(i, W.numCols() - 1, 0);
    }
    for (int i = 0; i < U.numRows(); i++) {
      U.set(i, U.numCols() - 1, 0);
    }

    // TODO: initialize lambda better...?
    lambda = 1.0;
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
      for (int row = 0; row < wordSize; row++) {
        x.set(counter * wordSize + row, 0, L.get(row, col));
      }
      counter++;
    }
    // add an intercept term at the end of x
    x.set(counter * wordSize, 0, 1);
    return x;
  }


  /**
   * Create a window of word indices centered at index
   * @param _trainData list of training data
   * @param index center index of window
   * @return list of indices
   */
  private List<Integer> createWindow(List<Datum> _trainData, int index) {

    List<Integer> indices = new ArrayList<Integer>(windowSize);
    Boolean startSeen = false;
    Boolean endSeen = false;

    // start of the window
    for (int i = index; i >= index - (windowSize - 1) / 2; i--) {
      if (startSeen || _trainData.get(i).word.equals(START)) { // handle start tokens
        startSeen = true;
        indices.add(0, FeatureFactory.wordToNum.get(START));
      } else {
        Integer nextIndex = FeatureFactory.wordToNum.get(_trainData.get(i).word.toLowerCase());
        if (nextIndex == null) { // handle words not contained in the vocabulary
          indices.add(0, FeatureFactory.wordToNum.get(UNK));
        } else {
          indices.add(0, nextIndex);
        }
      }
    }

    // end of the window
    for (int i = index + 1; i <= index + (windowSize - 1) / 2; i++) {
      if (endSeen || _trainData.get(i).word.equals(END)) { // handle end tokens
        endSeen = true;
        indices.add(FeatureFactory.wordToNum.get(END));
      } else {
        Integer nextIndex = FeatureFactory.wordToNum.get(_trainData.get(i).word.toLowerCase());
        if (nextIndex == null) { // handle words not contained in the vocabulary
          indices.add(FeatureFactory.wordToNum.get(UNK));
        } else {
          indices.add(nextIndex);
        }
      }
    }

    return indices;
  }

  /**
   * Apply the function f (hyperbolic tangent) to every element of x
   * @param x
   * @return
   */
  private SimpleMatrix fTransform(SimpleMatrix x) {
    SimpleMatrix toReturn = new SimpleMatrix(x.numRows() + 1, x.numCols());
    for (int i = 0; i < x.numRows(); i++) {
      double val = Math.tanh(x.get(i, 0));
      toReturn.set(i, 0, val);
    }
    // add an intercept entry at the end of x
    toReturn.set(toReturn.numRows() - 1, 0, 1);
    return toReturn;
  }

  /**
   * Apply the function g (softmax) to every element of x
   * @param x
   * @return
   */
  private SimpleMatrix gTransform(SimpleMatrix x) {
    double norm = 0;
    for (int i = 0; i < x.numRows(); i++) {
      double exp = Math.exp(x.get(i, 0));
      norm += exp;
      x.set(i, 0, exp);
    }
    // normalize x
    x = x.divide(norm);
    return x;
  }


  /**
   * Apply the feed-forward function to the input window x
   * @param x
   * @return
   */
  private SimpleMatrix feedForward(SimpleMatrix x) {
    SimpleMatrix afterF = fTransform(W.mult(x));
    return gTransform(U.mult(afterF));
  }


  /**
   * Compute the cost function using the log-likelihood
   * @param p
   * @param label
   * @return
   */
  private double costFunction(SimpleMatrix p, String label) {
    int y = labels.indexOf(label);
    return Math.log(p.get(y,0));
  }


  /**
   * Compute the regularization term for the cost function
   * @param m
   * @return
   */
  private double regularize(int m) {
    // subtract because don't want to penalize bias terms
    double R = Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2) - (hiddenSize + numLabels);
    System.out.println(R * lambda / (2 * m));
    return R * lambda / (2 * m);
  }

  /**
   * Simplest SGD training
   */
  public void train(List<Datum> _trainData) {

    int m = _trainData.size();
    double J = 0;
    for (int i = 0; i < m; i++) {

      // ignore sentence start and end tokens
      if (_trainData.get(i).word.equals(START) || _trainData.get(i).word.equals(END)) {
        continue;
      }

      // create input matrix x
      List<Integer> windowIndices = createWindow(_trainData, i);
      SimpleMatrix x = makeX(windowIndices);

      // apply feed-forward network function
      SimpleMatrix p = feedForward(x);

      // increment cost function
      J += costFunction(p, _trainData.get(i).label);
    }
    J /= -m;
    J += regularize(m);

    System.out.println(J);





  }


  public void test(List<Datum> testData) {
    // TODO
  }

}
