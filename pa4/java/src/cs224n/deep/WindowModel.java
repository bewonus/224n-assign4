package cs224n.deep;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import javax.swing.plaf.basic.BasicInternalFrameTitlePane;
import java.text.*;

public class WindowModel {

  protected SimpleMatrix L, W, U; // word-vector matrix, weight matrix, last layer weight matrix

  private List<String> labels = new ArrayList<String>(Arrays.asList("O", "LOC", "MISC", "ORG", "PER"));

  public int windowSize, wordSize, hiddenSize; // C, n, H
  public double learningRate; // alpha
  public int numLabels = labels.size();
  private double lambda;

  private static double epsilon = 0.0001;

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
    W = SimpleMatrix.random(hiddenSize, windowSize * wordSize + 1, -eW, eW, new java.util.Random()); // TODO: remove seed values later!
    U = SimpleMatrix.random(numLabels, hiddenSize + 1, -eU, eU, new java.util.Random());

    // initialize bias terms to 0
    for (int i = 0; i < W.numRows(); i++) {
      W.set(i, W.numCols() - 1, 0);
    }
    for (int i = 0; i < U.numRows(); i++) {
      U.set(i, U.numCols() - 1, 0);
    }

    // TODO: initialize lambda better...?
    lambda = 0.001;
//    lambda = 0;
  }

  /**
   * Create the concatenated vector x
   * @param indices into the matrix L
   * @return
   */
  private SimpleMatrix makeX(List<Integer> indices, SimpleMatrix L) {
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
   * @param trainData list of training data
   * @param index center index of window
   * @return list of indices
   */
  private List<Integer> createWindow(List<Datum> trainData, int index) {

    List<Integer> indices = new ArrayList<Integer>(windowSize);
    Boolean startSeen = false;
    Boolean endSeen = false;

    // start of the window
    for (int i = index; i >= index - (windowSize - 1) / 2; i--) {
      if (startSeen || trainData.get(i).word.equals(START)) { // handle start tokens
        startSeen = true;
        indices.add(0, FeatureFactory.wordToNum.get(START));
      } else {
        Integer nextIndex = FeatureFactory.wordToNum.get(trainData.get(i).word.toLowerCase());
        if (nextIndex == null) { // handle words not contained in the vocabulary
          indices.add(0, FeatureFactory.wordToNum.get(UNK));
        } else {
          indices.add(0, nextIndex);
        }
      }
    }

    // end of the window
    for (int i = index + 1; i <= index + (windowSize - 1) / 2; i++) {
      if (endSeen || trainData.get(i).word.equals(END)) { // handle end tokens
        endSeen = true;
        indices.add(FeatureFactory.wordToNum.get(END));
      } else {
        Integer nextIndex = FeatureFactory.wordToNum.get(trainData.get(i).word.toLowerCase());
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
   * @param k
   * @return
   */
  private double costFunction(SimpleMatrix p, int k) {
    return Math.log(p.get(k));
  }


  /**
   * Compute the regularization term for the cost function
   * @param m
   * @return
   */
  private double regularize(int m) {
    // subtract because don't want to penalize bias terms
    double R = Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2) - (hiddenSize + numLabels);
    return R * lambda / (2 * m);
  }

  /**
   * Compute delta2, which is used for computing the gradient dU.
   * @param p
   * @param k
   * @return
   */
  private SimpleMatrix delta2(SimpleMatrix p, int k) {
    SimpleMatrix e_k = new SimpleMatrix(p.numRows(), 1);
    e_k.set(k, 1.0);
    return p.minus(e_k);
  }


  private SimpleMatrix delta1(SimpleMatrix delta_2, SimpleMatrix h) {
    SimpleMatrix diag = SimpleMatrix.identity(h.numRows() - 1);
    for (int i = 0; i < diag.numRows(); i++) {
      diag.set(i, i, 1 - Math.pow(h.get(i), 2)); // TODO: not sure about intercept part here!!!
    }
    return diag.mult(U.extractMatrix(0, U.numRows(), 0, U.numCols() - 1).transpose()).mult(delta_2);
  }

  private double hiLoU(List<Datum> trainData, int m, int c, int j) {
    SimpleMatrix U2 = U.copy();
    U2.set(j, 0, U2.get(j, 0) + c * epsilon);
    double J = 0;
    SimpleMatrix z, h, p;
    for (int i = 0; i < m; i++) {

      // ignore sentence start and end tokens
      if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
        continue;
      }

      // create input matrix x
      List<Integer> windowIndices = createWindow(trainData, i);
      SimpleMatrix x = makeX(windowIndices, L);

      z = W.mult(x);
      h = fTransform(z);
      p = gTransform(U2.mult(h));
      int k = labels.indexOf(trainData.get(i).label);

      // increment cost function
      J += costFunction(p, k);
    }
    return -J / m;
  }

  private double hiLoW(List<Datum> trainData, int m, int c, int j) {
    SimpleMatrix W2 = W.copy();
    W2.set(j, 0, W2.get(j, 0) + c * epsilon);
    double J = 0;
    SimpleMatrix z, h, p;
    for (int i = 0; i < m; i++) {

      // ignore sentence start and end tokens
      if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
        continue;
      }

      // create input matrix x
      List<Integer> windowIndices = createWindow(trainData, i);
      SimpleMatrix x = makeX(windowIndices, L);

      z = W2.mult(x);
      h = fTransform(z);
      p = gTransform(U.mult(h));
      int k = labels.indexOf(trainData.get(i).label);

      // increment cost function
      J += costFunction(p, k);
    }
    return -J / m;
  }

  private double hiLoL(List<Datum> trainData, int m, int c, int j) {
    SimpleMatrix L2 = L.copy();
    L2.set(j, 0, L2.get(j, 0) + c * epsilon);
    double J = 0;
    SimpleMatrix z, h, p;
    for (int i = 0; i < m; i++) {

      // ignore sentence start and end tokens
      if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
        continue;
      }

      // create input matrix x
      List<Integer> windowIndices = createWindow(trainData, i);
      SimpleMatrix x = makeX(windowIndices, L2);

      z = W.mult(x);
      h = fTransform(z);
      p = gTransform(U.mult(h));
      int k = labels.indexOf(trainData.get(i).label);

      // increment cost function
      J += costFunction(p, k);
    }
    return -J / m;
  }


  /**
   * Simplest SGD training
   */
  public void train(List<Datum> trainData) {

    int m = trainData.size();
    m = 10000;
    System.out.println(m);
    double J;
    SimpleMatrix z, h, p, delta_2, delta_1, dx, dU, dW, dU2, dW2;
    SimpleMatrix dL = new SimpleMatrix(L.numRows(), L.numCols());
    SimpleMatrix dU_checking = new SimpleMatrix(U.numRows(), U.numCols());
    SimpleMatrix dW_checking = new SimpleMatrix(W.numRows(), W.numCols());
    SimpleMatrix dL_checking = new SimpleMatrix(L.numRows(), L.numCols());

    // Perform stochastic gradient steps
    int numGradSteps = 10;
    for (int j = 0; j < numGradSteps; j++) {
      J = 0.0;

      long t = System.currentTimeMillis();


      // Loop through training examples
      for (int i = 0; i < m; i++) {

//      System.out.println(t);

//        if (i % 1000 == 0) System.out.println("Example: " + i / 1000);


        // ignore sentence start and end tokens
        if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
          continue;
        }

        // create input matrix x
        List<Integer> windowIndices = createWindow(trainData, i);
        SimpleMatrix x = makeX(windowIndices, L);

        // apply feed-forward network function
//      SimpleMatrix p = feedForward(x);

        // get z, h, p, k
        z = W.mult(x);
        h = fTransform(z);
        p = gTransform(U.mult(h));
        int k = labels.indexOf(trainData.get(i).label);

        // increment cost function
        J += costFunction(p, k);

        // delta2
        delta_2 = delta2(p, k);

        // delta1
        delta_1 = delta1(delta_2, h);

        // increment dU
        dU = delta_2.mult(h.transpose());
//      dU_checking = dU_checking.plus(dU);

        dU2 = U.copy();
        for (int row = 0; row < dU2.numRows(); row++) {
          dU2.set(row, dU2.numCols() - 1, 0);
        }
        dU2.scale(lambda);

        // increment dW
        dW = delta_1.mult(x.transpose());
//      dW_checking = dW_checking.plus(dW);

        dW2 = W.copy();
        for (int row = 0; row < dW2.numRows(); row++) {
          dW2.set(row, dW2.numCols() - 1, 0);
        }
        dW2.scale(lambda);

        // increment dL (set dx first)
        dx = W.extractMatrix(0, W.numRows(), 0, W.numCols() - 1).transpose().mult(delta_1);
        int counter = 0;
        for (int col : windowIndices) {
          for (int row = 0; row < wordSize; row++) {
//          dL_checking.set(row, col, dL_checking.get(row, col) + dx.get(counter));
//          dL.set(row, col, dx.get(counter));
            L.set(row, col, L.get(row, col) - (dx.get(counter) * learningRate));
            counter++;
          }
        }

        U = U.minus(dU.scale(learningRate));
        U = U.minus(dU2.scale(learningRate/m));
        W = W.minus(dW.scale(learningRate));
        W = W.minus(dW2.scale(learningRate/m));
//      System.out.println(System.currentTimeMillis() - t);
      }
      J /= -m;
      J += regularize(m);
      System.out.println(J);
      System.out.println("time: " + (System.currentTimeMillis() - t));
    }

//    dU_checking = dU_checking.scale(1.0 / m);
//    dW_checking = dW_checking.scale(1.0 / m);
//    dL_checking = dL_checking.scale(1.0 / m);


    // TODO: have a for loop that chooses 10 random pairs of indices per gradient matrix (for gradient checking)

    // gradient checking for U and b2
//    for (int j = 0; j < 4; j++) {
//      double Jhi = hiLoU(trainData, m, 1, j);
//      double Jlo = hiLoU(trainData, m, -1, j);
//      System.out.println((Jhi - Jlo) / (2 * epsilon));
//      System.out.println(dU_checking.get(j, 0));
//    }
//
//    // gradient checking for W and b1
//    for (int j = 0; j < 4; j++) {
//      double Jhi = hiLoW(trainData, m, 1, j);
//      double Jlo = hiLoW(trainData, m, -1, j);
//      System.out.println((Jhi - Jlo) / (2 * epsilon));
//      System.out.println(dW_checking.get(j, 0));
//    }
//
//    // gradient checking for L
//    for (int j = 0; j < 4; j++) {
//      double Jhi = hiLoL(trainData, m, 1, j);
//      double Jlo = hiLoL(trainData, m, -1, j);
//      System.out.println((Jhi - Jlo) / (2 * epsilon));
//      System.out.println(dL_checking.get(j, 0));
//    }


//    J += regularize(m);


  }


  public void test(List<Datum> testData, Boolean isTest) {
    System.out.println("alpha: " + learningRate);
    System.out.println("lambda: " + lambda);
//    System.out.println(U);
//    System.out.println(W);
    try {
      File file;
      if (!isTest) {
        file = new File("windowTrainReg.txt");
      } else {
        file = new File("windowTestReg.txt");
      }

      BufferedWriter output = new BufferedWriter(new FileWriter(file));

      int m = testData.size();
      m = 10000;
      for (int i = 0; i < m; i++) {

        String predictLabel = "UNK"; // TODO: or "UUUNKKK"? (same for baseline?)
//        String predictLabel = "O";

        if (testData.get(i).word.equals(START) || testData.get(i).word.equals(END)) {
          continue;
        }

        List<Integer> windowIndices = createWindow(testData, i);
        SimpleMatrix x = makeX(windowIndices, L);

        // apply feed-forward network function
        SimpleMatrix p = gTransform(U.mult(fTransform(W.mult(x))));
        double max = 0.0;
//        System.out.println(p);
        for (int labelIndex = 0; labelIndex < p.numRows(); labelIndex++) {
          double next = p.get(labelIndex, 0);
          if (next > max) {
            max = next;
            predictLabel = labels.get(labelIndex);
          }
        }

        output.write(testData.get(i).word + "\t" + testData.get(i).label + "\t" + predictLabel + "\n");
      }

      output.close();

    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
