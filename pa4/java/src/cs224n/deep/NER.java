package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {

  public static void main(String[] args) throws IOException {
    if (args.length < 2) {
//	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
//      System.out.println("USAGE: java -cp classes cs224n.deep.NER ../data/train ../data/dev");
      System.out.println("USAGE: java -cp classes:extlib cs224n.deep.NER ../data/train ../data/dev");
      return;
    }

    // read in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);

    // read in vocab and word vectors
    FeatureFactory.initializeVocab("../data/vocab.txt");
    SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

    // train and test baseline model
    BaselineModel baseline = new BaselineModel();
    baseline.train(trainData);
    baseline.test(testData);

    // initialize window model (with C, H, alpha, lambda, epochs)
//    WindowModel model = new WindowModel(5, 50, 0.01, 1.0, 5); //TODO: initially (5, 100, 0.001)
//    model.initWeights();


    WindowModel lambda1 = new WindowModel(5, 50, 0.01, 1.0, 20);
    WindowModel lambda2 = new WindowModel(5, 50, 0.01, 0.1, 20);
    WindowModel lambda3 = new WindowModel(5, 50, 0.01, 0.01, 20);
    WindowModel lambda4 = new WindowModel(5, 50, 0.01, 0.001, 20);
    WindowModel lambda5 = new WindowModel(5, 50, 0.01, 0.0001, 20);

    lambda1.initWeights();
    lambda2.initWeights();
    lambda3.initWeights();
    lambda4.initWeights();
    lambda5.initWeights();

    lambda1.train(trainData, false, "lambda1.txt");
    lambda2.train(trainData, false, "lambda2.txt");
    lambda3.train(trainData, false, "lambda3.txt");
    lambda4.train(trainData, false, "lambda4.txt");
    lambda5.train(trainData, false, "lambda5.txt");

    lambda1.test(trainData, "lambda1-Train.txt", true);
    lambda2.test(trainData, "lambda2-Train.txt", true);
    lambda3.test(trainData, "lambda3-Train.txt", true);
    lambda4.test(trainData, "lambda4-Train.txt", true);
    lambda5.test(trainData, "lambda5-Train.txt", true);

    lambda1.test(testData, "lambda1-Test.txt", false);
    lambda2.test(testData, "lambda2-Test.txt", false);
    lambda3.test(testData, "lambda3-Test.txt", false);
    lambda4.test(testData, "lambda4-Test.txt", false);
    lambda5.test(testData, "lambda5-Test.txt", false);


    // train and test window model
//    model.train(trainData, false); // set to true for gradient checking
//    model.test(trainData, true);
//    model.test(testData, false);
  }
}
