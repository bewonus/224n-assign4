package cs224n.deep;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.simple.*;

public class WindowModel {

  protected SimpleMatrix L, W, U; // word-vector matrix, weight matrix, last layer weight matrix

  private List<String> labels = new ArrayList<String>(Arrays.asList("O", "LOC", "MISC", "ORG", "PER"));

  public int windowSize, wordSize, hiddenSize; // C, n, H
  public double learningRate; // alpha
  public int numLabels = labels.size();
  private double lambda;

  private static final String START = "<s>";
  private static final String END = "</s>";
  private static final String UNK = "UUUNKKK";

  /**
   * Construct a window model.
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

    // TODO: initialize lambda better...?
    lambda = 0.0;
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
   * Apply the function f (hyperbolic tangent) to every element of x.
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
   * Apply the function g (softmax) to every element of x.
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
   * Apply the feed-forward function to the input window x.
   * @param x
   * @return
   */
  private SimpleMatrix feedForward(SimpleMatrix x) {
    return gTransform(U.mult(fTransform(W.mult(x))));
  }


  /**
   * Compute the cost function using the log-likelihood.
   * @param p
   * @param k
   * @return
   */
  private double costFunction(SimpleMatrix p, int k) {
    return Math.log(p.get(k));
  }


  /**
   * Compute the regularization term for the cost function.
   * @param m
   * @return
   */
  private double regularize(int m) {
    // subtract because don't want to penalize bias terms
    double R = Math.pow(W.normF(), 2) + Math.pow(U.normF(), 2) - (hiddenSize + numLabels);
    return R * lambda / (2 * m);
  }

  /**
   * Compute delta2, which is used in the gradient computations.
   * @param p
   * @param k
   * @return
   */
  private SimpleMatrix delta2(SimpleMatrix p, int k) {
    SimpleMatrix e_k = new SimpleMatrix(p.numRows(), 1);
    e_k.set(k, 1.0);
    return p.minus(e_k);
  }

  /**
   * Compute delta1, which is used in the gradient computations.
   * @param delta_2
   * @param h
   * @return
   */
  private SimpleMatrix delta1(SimpleMatrix delta_2, SimpleMatrix h) {
    SimpleMatrix diag = SimpleMatrix.identity(h.numRows() - 1);
    for (int i = 0; i < diag.numRows(); i++) {
      diag.set(i, i, 1 - Math.pow(h.get(i), 2)); // TODO: not sure about intercept part here!!!
    }
    return diag.mult(U.extractMatrix(0, U.numRows(), 0, U.numCols() - 1).transpose()).mult(delta_2);
  }


  private double performCheck(SimpleMatrix U, SimpleMatrix W, SimpleMatrix L, List<Datum> trainData, int m) {
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


  private void performGradCheck(List<Datum> trainData, int m) {
    SimpleMatrix z, h, p, delta_2, delta_1, dx;
    SimpleMatrix dU = new SimpleMatrix(U.numRows(), U.numCols());
    SimpleMatrix dW = new SimpleMatrix(W.numRows(), W.numCols());
    SimpleMatrix dL = new SimpleMatrix(L.numRows(), L.numCols());

    // Compute Gradients

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

    // Compute J high and J low, and compare to relevant gradient entry
    double Jhilo[] = new double[2];
    double epsilon = 0.0001;
    for (int UWL = 0; UWL < 3; UWL++) {
      switch (UWL) {
        case 0:
          System.out.println("checking U");
//          Ucheck(trainData, m, dU, Jhilo, epsilon);
          break;
        case 1:
          System.out.println("checking W");
//          Wcheck(trainData, m, dW, Jhilo, epsilon);
          break;
        default:
          System.out.println("checking L");
          Lcheck(trainData, m, dL, Jhilo, epsilon);
          break;
      }
    }
  }

  /**
   * Perform gradient checking on L.
   * @param trainData
   * @param m
   * @param dL
   * @param Jhilo
   * @param epsilon
   */
  private void Lcheck(List<Datum> trainData, int m, SimpleMatrix dL, double[] Jhilo, double epsilon) {
    SimpleMatrix L2 = L.copy();
    int numColsToCheck = 200;
    SimpleMatrix dLapprox = new SimpleMatrix(L.numRows(), numColsToCheck);
    for (int row = 0; row < L.numRows(); row++) {
      for (int col = 0; col < numColsToCheck; col++) {
        for (int c = 0; c < 2; c++) {
          L2.set(row, col, L2.get(row, col) + (2 * c - 1) * epsilon);
          Jhilo[c] = performCheck(U, W, L2, trainData, m);
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
   * Perform gradient checking on W.
   * @param trainData
   * @param m
   * @param dW
   * @param Jhilo
   * @param epsilon
   */
  private void Wcheck(List<Datum> trainData, int m, SimpleMatrix dW, double[] Jhilo, double epsilon) {
    SimpleMatrix W2 = W.copy();
    SimpleMatrix dWapprox = new SimpleMatrix(W.numRows(), W.numCols());
    for (int row = 0; row < W.numRows(); row++) {
      for (int col = 0; col < W.numCols(); col++) {
        for (int c = 0; c < 2; c++) {
          W2.set(row, col, W2.get(row, col) + (2 * c - 1) * epsilon);
          Jhilo[c] = performCheck(U, W2, L, trainData, m);
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
   * Perform gradient checking on U.
   * @param trainData
   * @param m
   * @param dU
   * @param jhilo
   * @param epsilon
   */
  private void Ucheck(List<Datum> trainData, int m, SimpleMatrix dU, double[] jhilo, double epsilon) {
    SimpleMatrix U2 = U.copy();
    SimpleMatrix dUapprox = new SimpleMatrix(U.numRows(), U.numCols());
    for (int row = 0; row < U.numRows(); row++) {
      for (int col = 0; col < U.numCols(); col++) {
        for (int c = 0; c < 2; c++) {
          U2.set(row, col, U2.get(row, col) + (2 * c - 1) * epsilon);
          jhilo[c] = performCheck(U2, W, L, trainData, m);
          U2.set(row, col, U2.get(row, col) - (2 * c - 1) * epsilon);
        }
//        System.out.println((jhilo[1] - jhilo[0]) / (2 * epsilon));
//        System.out.println(dU.get(row, col));
        dUapprox.set(row, col, (jhilo[1] - jhilo[0]) / (2 * epsilon));
      }
    }
    System.out.println(dU.minus(dUapprox).normF());
  }


  /**
   * Simplest SGD training
   */
  public void train(List<Datum> trainData) {

    Boolean gradChecking = true;
    double J;
    SimpleMatrix z, h, p, delta_2, delta_1, dx, dU, dW, dUreg, dWreg;

    if (gradChecking) { // Gradient checking
      int numExamples = 100;
      performGradCheck(trainData, numExamples);
    } else { // Stochastic gradient descent
      int m = trainData.size();
      int epochs = 10;
      for (int j = 0; j < epochs; j++) {
        J = 0.0;

        long t = System.currentTimeMillis();

        // Loop through training examples
        for (int i = 0; i < m; i++) {

          if (i % 10000 == 0) System.out.print(i / 10000 + " ");

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

          // compute dUreg and dWreg (contributions to U and W from regularization)
          dUreg = U.copy();
          for (int row = 0; row < dUreg.numRows(); row++) {
            dUreg.set(row, dUreg.numCols() - 1, 0);
          }
          dUreg = dUreg.scale(lambda);

          dWreg = W.copy();
          for (int row = 0; row < dWreg.numRows(); row++) {
            dWreg.set(row, dWreg.numCols() - 1, 0);
          }
          dWreg = dWreg.scale(lambda);

          // SGD (update U, W, and L)
          U = U.minus(dU.scale(learningRate));
          U = U.minus(dUreg.scale(learningRate));
          W = W.minus(dW.scale(learningRate));
          W = W.minus(dWreg.scale(learningRate));
          int counter = 0;
          for (int col : windowIndices) {
            for (int row = 0; row < wordSize; row++) {
              L.set(row, col, L.get(row, col) - (dx.get(counter) * learningRate));
              counter++;
            }
          }
        }
        J /= -m;
        J += regularize(m);
        System.out.println(J);
        System.out.println("time: " + (System.currentTimeMillis() - t));
        System.out.println(j);
      }
    }

  }

  /**
   * Predict labels using the testData.
   * @param testData
   * @param isTest
   */
  public void test(List<Datum> testData, Boolean isTest) {
    System.out.println("alpha: " + learningRate);
    System.out.println("lambda: " + lambda);
//    System.out.println(U);
//    System.out.println(W);
    try {
      File file;
      if (!isTest) {
        file = new File("windowTrain05H.txt");
      } else {
        file = new File("windowTest05H.txt");
      }

      BufferedWriter output = new BufferedWriter(new FileWriter(file));

      int m = testData.size();
//      if (!isTest) {
//        m = 25000;
//      }
      for (int i = 0; i < m; i++) {

        String predictLabel = "UNK"; // TODO: or "UUUNKKK"? (same for baseline?)

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
