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

    // this reads in the train and test datasets
    List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
    List<Datum> testData = FeatureFactory.readTestData(args[1]);

    // read the train and test data
    // reads in vocab and word vectors
    FeatureFactory.initializeVocab("../data/vocab.txt");
    SimpleMatrix allVecs = FeatureFactory.readWordVectors("../data/wordVectors.txt");

    // baseline NER
    BaselineModel baseline = new BaselineModel();
    baseline.train(trainData);
    baseline.test(testData);

    // initialize model
    WindowModel model = new WindowModel(2, 100, 0.001); //TODO: change 2 back to 5!
    model.initWeights();

    //TODO: Implement those two functions
    model.train(trainData);
    model.test(testData);
  }
}