package cs224n.deep;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.simple.*;

public class WindowModel {

  // word-vector matrix, weight matrix, and last layer weight matrix
  private SimpleMatrix L, W, U;

  // list of possible entity labels
  private List<String> labels = new ArrayList<String>(Arrays.asList("O", "LOC", "MISC", "ORG", "PER"));

  // number of labels
  private int numLabels = labels.size();

  // special tokens
  private static final String START = "<s>";
  private static final String END = "</s>";
  private static final String UNK = "UUUNKKK";

  // --- Hyper Parameters --- //

  // c, n, and H in the handout
  private int windowSize, wordSize, hiddenSize;

  // learning rate
  private double alpha;

  // regularization parameter
  private double lambda;

  // epochs (number of iterations of SGD)
  private int epochs;

  /**
   * Construct a window model.
   */
  public WindowModel(int _windowSize, int _hiddenSize, double _alpha, double _lambda, int _epochs) {
    windowSize = _windowSize;
    hiddenSize = _hiddenSize;
    alpha = _alpha;
    lambda = _lambda;
    epochs = _epochs;
  }

  /**
   * Initializes the weight matrices L, W, and U.
   */
  public void initWeights() {
    wordSize = FeatureFactory.allVecs.numRows(); // == 50

    // word-vector matrix (i.e. the x's)
    L = new SimpleMatrix(FeatureFactory.allVecs);

    // set params for random initialization
    double eW = Math.sqrt(6.0) / Math.sqrt(hiddenSize + windowSize * wordSize);
    double eU = Math.sqrt(6.0) / Math.sqrt(numLabels + hiddenSize);
    // TODO: are eW and eU correct...?

    // randomly initialize W and U (with biases inside the last column (i.e. plus 1))
    W = SimpleMatrix.random(hiddenSize, windowSize * wordSize + 1, -eW, eW, new java.util.Random());
    U = SimpleMatrix.random(numLabels, hiddenSize + 1, -eU, eU, new java.util.Random());

    // initialize bias terms to 0
    for (int i = 0; i < W.numRows(); i++) {
      W.set(i, W.numCols() - 1, 0);
    }
    for (int i = 0; i < U.numRows(); i++) {
      U.set(i, U.numCols() - 1, 0);
    }
  }

  /**
   * Creates the concatenated vector x, using indices into L.
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
   * Create a window of word indices centered at the given index.
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
   * Apply the function f (hyperbolic tangent) to every element of x, then add an intercept term to x.
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
   * Apply the function g (softmax) to every element of x.
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
   * Compute the cost function from a single window, using log-likelihood.
   */
  private double costFunction(SimpleMatrix p, int k) {
    return Math.log(p.get(k));
  }

  /**
   * Compute the regularization term for the cost function.
   */
  private double regularize(int m) {
    // subtract because don't want to penalize bias terms
    double R = Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2) - (hiddenSize + numLabels);
    return R * lambda / (2 * m);
  }

  /**
   * Compute delta2, which is used in the gradient computations.
   */
  private SimpleMatrix delta2(SimpleMatrix p, int k) {
    SimpleMatrix e_k = new SimpleMatrix(p.numRows(), 1);
    e_k.set(k, 1.0);
    return p.minus(e_k);
  }

  /**
   * Compute delta1, which is used in the gradient computations.
   */
  private SimpleMatrix delta1(SimpleMatrix delta_2, SimpleMatrix h) {
    // ignore final row of h (corresponds to the intercept term, which is not used here)
    SimpleMatrix diag = SimpleMatrix.identity(h.numRows() - 1);
    for (int i = 0; i < diag.numRows(); i++) {
      diag.set(i, i, 1 - Math.pow(h.get(i), 2));
    }
    // ignore final column of U (corresponds to the intercept term, which is not used here)
    return diag.mult(U.extractMatrix(0, U.numRows(), 0, U.numCols() - 1).transpose()).mult(delta_2);
  }

  /**
   * Computes the cost function J (unregularized) using the given
   * input matrices and a number (m) of training samples to use.
   * Note: This method is only used for gradient checking.
   */
  private double getJ(SimpleMatrix U, SimpleMatrix W, SimpleMatrix L, List<Datum> trainData, int m) {
    double J = 0;
    SimpleMatrix p;
    for (int i = 0; i < m; i++) {
      // ignore sentence start and end tokens
      if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
        continue;
      }
      // create input matrix x
      List<Integer> windowIndices = createWindow(trainData, i);
      SimpleMatrix x = makeX(windowIndices, L);

      // compute p and k
      p = gTransform(U.mult(fTransform(W.mult(x))));
      int k = labels.indexOf(trainData.get(i).label);

      // increment cost function
      J += costFunction(p, k);
    }
    return -J / m;
  }

  /**
   * Performs gradient checking.
   */
  private void performGradCheck(List<Datum> trainData, int m) {
    SimpleMatrix z, h, p, delta_2, delta_1, dx;
    SimpleMatrix dU = new SimpleMatrix(U.numRows(), U.numCols());
    SimpleMatrix dW = new SimpleMatrix(W.numRows(), W.numCols());
    SimpleMatrix dL = new SimpleMatrix(L.numRows(), L.numCols());

    // 1. Compute Gradients

    // Loop through training examples
    for (int i = 0; i < m; i++) {
      // ignore sentence start and end tokens
      if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
        continue;
      }
      // create input matrix x
      List<Integer> windowIndices = createWindow(trainData, i);
      SimpleMatrix x = makeX(windowIndices, L);

      // get z, h, p, k (apply feed-forward network function)
      z = W.mult(x);
      h = fTransform(z);
      p = gTransform(U.mult(h));
      int k = labels.indexOf(trainData.get(i).label);

      // compute delta2 and delta1
      delta_2 = delta2(p, k);
      delta_1 = delta1(delta_2, h);

      // compute dU, dW, dx
      dU = dU.plus(delta_2.mult(h.transpose()));
      dW = dW.plus(delta_1.mult(x.transpose()));
      dx = W.extractMatrix(0, W.numRows(), 0, W.numCols() - 1).transpose().mult(delta_1);

      int counter = 0;
      for (int col : windowIndices) {
        for (int row = 0; row < wordSize; row++) {
          dL.set(row, col, dL.get(row, col) + dx.get(counter));
          counter++;
        }
      }
    }
    dU = dU.scale(1.0 / m);
    dW = dW.scale(1.0 / m);
    dL = dL.scale(1.0 / m);

    // 2. Compare theoretical gradients to approximate gradients based on the cost function.
    double Jhilo[] = new double[2];
    double epsilon = 0.0001;
    Ucheck(trainData, m, dU, Jhilo, epsilon);
    Wcheck(trainData, m, dW, Jhilo, epsilon);
    Lcheck(trainData, m, dL, Jhilo, epsilon);
  }

  /**
   * Perform gradient checking on U.
   */
  private void Ucheck(List<Datum> trainData, int m, SimpleMatrix dU, double[] Jhilo, double epsilon) {
    System.out.println("checking U...");
    SimpleMatrix U2 = U.copy();
    SimpleMatrix dUapprox = new SimpleMatrix(U.numRows(), U.numCols());
    for (int row = 0; row < U.numRows(); row++) {
      for (int col = 0; col < U.numCols(); col++) {
        for (int c = 0; c < 2; c++) {
          U2.set(row, col, U2.get(row, col) + (2 * c - 1) * epsilon);
          Jhilo[c] = getJ(U2, W, L, trainData, m);
          U2.set(row, col, U2.get(row, col) - (2 * c - 1) * epsilon);
        }
//        System.out.println((Jhilo[1] - Jhilo[0]) / (2 * epsilon));
//        System.out.println(dU.get(row, col));
        dUapprox.set(row, col, (Jhilo[1] - Jhilo[0]) / (2 * epsilon));
      }
    }
    System.out.println(dU.minus(dUapprox).normF());
  }

  /**
   * Perform gradient checking on W.
   */
  private void Wcheck(List<Datum> trainData, int m, SimpleMatrix dW, double[] Jhilo, double epsilon) {
    System.out.println("checking W...");
    SimpleMatrix W2 = W.copy();
    SimpleMatrix dWapprox = new SimpleMatrix(W.numRows(), W.numCols());
    for (int row = 0; row < W.numRows(); row++) {
      for (int col = 0; col < W.numCols(); col++) {
        for (int c = 0; c < 2; c++) {
          W2.set(row, col, W2.get(row, col) + (2 * c - 1) * epsilon);
          Jhilo[c] = getJ(U, W2, L, trainData, m);
          W2.set(row, col, W2.get(row, col) - (2 * c - 1) * epsilon);
        }
//        System.out.println((Jhilo[1] - Jhilo[0]) / (2 * epsilon));
//        System.out.println(dW.get(row, col));
        dWapprox.set(row, col, (Jhilo[1] - Jhilo[0]) / (2 * epsilon));
      }
    }
    System.out.println(dW.minus(dWapprox).normF());
  }

  /**
   * Perform gradient checking on L.
   */
  private void Lcheck(List<Datum> trainData, int m, SimpleMatrix dL, double[] Jhilo, double epsilon) {
    System.out.println("checking L...");
    SimpleMatrix L2 = L.copy();
    int numColsToCheck = 200;
    SimpleMatrix dLapprox = new SimpleMatrix(L.numRows(), numColsToCheck);
    for (int row = 0; row < L.numRows(); row++) {
      for (int col = 0; col < numColsToCheck; col++) {
        for (int c = 0; c < 2; c++) {
          L2.set(row, col, L2.get(row, col) + (2 * c - 1) * epsilon);
          Jhilo[c] = getJ(U, W, L2, trainData, m);
          L2.set(row, col, L2.get(row, col) - (2 * c - 1) * epsilon);
        }
//        System.out.println((Jhilo[1] - Jhilo[0]) / (2 * epsilon));
//        System.out.println(dL.get(row, col));
        dLapprox.set(row, col, (Jhilo[1] - Jhilo[0]) / (2 * epsilon));
      }
    }
    System.out.println(dL.extractMatrix(0, dL.numRows(), 0, numColsToCheck).minus(dLapprox).normF());
  }

  /**
   * Train a neural network using stochastic gradient descent.
   */
  public void train(List<Datum> trainData, Boolean gradChecking, String filename) {
    double J;
    SimpleMatrix z, h, p, delta_2, delta_1, dx, dU, dW, dUreg, dWreg;

    // *** GRADIENT CHECKING *** //
    if (gradChecking) {
      // number of examples to use for gradient checking
      int numExamples = 50;
      performGradCheck(trainData, numExamples);
    } else {
      // SGD and visualization
      try {
        File file = new File(filename);
        BufferedWriter output = new BufferedWriter(new FileWriter(file));

        output.write("iteration" + "\t" + "cost" + "\t" + "costReg" + "\t" + "C" + "\t" + "H" + "\t" + "alpha" + "\t" + "lambda" + "\t" + "time" + "\n");

        // Stochastic gradient descent
        int m = trainData.size();
//        m = 50000; // TODO: comment me out!
        System.out.println(m);

        // loop through epochs iterations of SGD
        for (int j = 0; j < epochs; j++) {
          J = 0;

          long t = System.currentTimeMillis();

          // loop through training examples
          for (int i = 0; i < m; i++) {
            if (i % 25000 == 0) System.out.print(i / 25000 + " ");

            // ignore sentence start and end tokens
            if (trainData.get(i).word.equals(START) || trainData.get(i).word.equals(END)) {
              continue;
            }
            // create input matrix x
            List<Integer> windowIndices = createWindow(trainData, i);
            SimpleMatrix x = makeX(windowIndices, L);

            // get z, h, p, k (apply feed-forward network function)
            z = W.mult(x);
            h = fTransform(z);
            p = gTransform(U.mult(h));
            int k = labels.indexOf(trainData.get(i).label);

            // increment cost function
            J += costFunction(p, k);

            // compute delta2 and delta1
            delta_2 = delta2(p, k);
            delta_1 = delta1(delta_2, h);

            // compute dU, dW, dx
            dU = delta_2.mult(h.transpose());
            dW = delta_1.mult(x.transpose());
            dx = W.extractMatrix(0, W.numRows(), 0, W.numCols() - 1).transpose().mult(delta_1);

            // compute dUreg (contribution to U from regularization)
            dUreg = U.copy();
            for (int row = 0; row < dUreg.numRows(); row++) {
              dUreg.set(row, dUreg.numCols() - 1, 0);
            }
            dUreg = dUreg.scale(lambda);

            // compute dWreg (contribution to W from regularization)
            dWreg = W.copy();
            for (int row = 0; row < dWreg.numRows(); row++) {
              dWreg.set(row, dWreg.numCols() - 1, 0);
            }
            dWreg = dWreg.scale(lambda);

            // SGD (update U, W, and L)
            U = U.minus(dU.scale(alpha));
            U = U.minus(dUreg.scale(alpha));
            W = W.minus(dW.scale(alpha));
            W = W.minus(dWreg.scale(alpha));
            int counter = 0;
            for (int col : windowIndices) {
              for (int row = 0; row < wordSize; row++) {
                L.set(row, col, L.get(row, col) - (dx.get(counter) * alpha));
                counter++;
              }
            }
          }

          // adjust and print cost function (with regularization)
          J /= -m;
          double Jreg = J + regularize(m);
//          J += regularize(m);
          System.out.println(j+1);
          System.out.println(J);
          System.out.println(Jreg);

          // write diagnostic information
          output.write(j + "\t" + J + "\t" + Jreg + "\t" + windowSize + "\t" + hiddenSize + "\t" + alpha + "\t" + lambda + "\t" + (System.currentTimeMillis() - t) + "\n");
        }
        output.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

  }

  /**
   * Predict named-entity labels (flag tells whether testData is actually the training data, if we want train error).
   */
  public void test(List<Datum> testData, String filename, boolean isTrain) {
    // print alpha and lambda
    System.out.println("C: " + windowSize);
    System.out.println("H: " + hiddenSize);
    System.out.println("alpha: " + alpha);
    System.out.println("lambda: " + lambda);

    try {
      File file = new File(filename);
      BufferedWriter output = new BufferedWriter(new FileWriter(file));

      // loop through testing examples
      int m = testData.size();
//      if (isTrain) m = 50000; // TODO: comment me out!

      for (int i = 0; i < m; i++) {
        String predictLabel = "UNK"; // TODO: or "UUUNKKK"? (same for baseline?)

        // ignore sentence start and end tokens
        if (testData.get(i).word.equals(START) || testData.get(i).word.equals(END)) {
          continue;
        }
        // create input matrix x
        List<Integer> windowIndices = createWindow(testData, i);
        SimpleMatrix x = makeX(windowIndices, L);

        // apply feed-forward network function
        SimpleMatrix p = gTransform(U.mult(fTransform(W.mult(x))));

        // find most probable label
        double max = 0.0;
        for (int labelIndex = 0; labelIndex < p.numRows(); labelIndex++) {
          double next = p.get(labelIndex, 0);
          if (next > max) {
            max = next;
            predictLabel = labels.get(labelIndex);
          }
        }

        // write word, gold standard, and prediction to file
        output.write(testData.get(i).word + "\t" + testData.get(i).label + "\t" + predictLabel + "\n");
      }
      output.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
