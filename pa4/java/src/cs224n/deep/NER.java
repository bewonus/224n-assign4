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

    // initialize window model (with C, H, alpha, lambda)
    WindowModel model = new WindowModel(5, 50, 0.01, 0.0); //TODO: initially (5, 100, 0.001)
    model.initWeights();

    // train and test window model
    model.train(trainData, false); // set to true for gradient checking
    model.test(trainData, false); // compute training error
    model.test(testData, true); // compute testing error
  }
}
