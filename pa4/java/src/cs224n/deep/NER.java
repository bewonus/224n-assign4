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

    // adjusting lambda
//    WindowModel model1 = new WindowModel(5, 50, 0.01, 0.01, 20);
//    WindowModel model2 = new WindowModel(5, 50, 0.01, 0.005, 20);
//    WindowModel model3 = new WindowModel(5, 50, 0.01, 0.001, 20);
//    WindowModel model4 = new WindowModel(5, 50, 0.01, 0.0005, 20);
//    WindowModel model5 = new WindowModel(5, 50, 0.01, 0.0001, 20);
//
//    // adjusting hidden size
//    WindowModel model10 = new WindowModel(5, 25, 0.01, 0.001, 20);
////    WindowModel model20 = new WindowModel(5, 50, 0.01, 0.001, 20); // same as model3
//    WindowModel model30 = new WindowModel(5, 75, 0.01, 0.001, 20);
//    WindowModel model40 = new WindowModel(5, 100, 0.01, 0.001, 20);
//    WindowModel model50 = new WindowModel(5, 125, 0.01, 0.001, 20);
//    WindowModel model60 = new WindowModel(5, 150, 0.01, 0.001, 20);
//
//    // adjusting window size
//    WindowModel model100 = new WindowModel(1, 50, 0.01, 0.001, 20);
//    WindowModel model200 = new WindowModel(3, 50, 0.01, 0.001, 20);
////    WindowModel model300 = new WindowModel(3, 50, 0.01, 0.001, 20); // same as model3
//    WindowModel model400 = new WindowModel(7, 50, 0.01, 0.001, 20);
//    WindowModel model500 = new WindowModel(9, 50, 0.01, 0.001, 20);
//
//    model1.initWeights();
//    model2.initWeights();
//    model3.initWeights();
//    model4.initWeights();
//    model5.initWeights();
//
//    model10.initWeights();
//    model30.initWeights();
//    model40.initWeights();
//    model50.initWeights();
//
//    model1.train(trainData, false, "model1.txt");
//    model2.train(trainData, false, "model2.txt");
//    model3.train(trainData, false, "model3.txt");
//    model4.train(trainData, false, "model4.txt");
//    model5.train(trainData, false, "model5.txt");
//
//    model1.test(trainData, "model1-Train.txt", true);
//    model2.test(trainData, "model2-Train.txt", true);
//    model3.test(trainData, "model3-Train.txt", true);
//    model4.test(trainData, "model4-Train.txt", true);
//    model5.test(trainData, "model5-Train.txt", true);
//
//    model1.test(testData, "model1-Test.txt", false);
//    model2.test(testData, "model2-Test.txt", false);
//    model3.test(testData, "model3-Test.txt", false);
//    model4.test(testData, "model4-Test.txt", false);
//    model5.test(testData, "model5-Test.txt", false);


    List<Integer> CValues = new ArrayList<Integer>(Arrays.asList(5, 3, 7));
    List<Integer> HValues = new ArrayList<Integer>(Arrays.asList(100, 50, 125));
    List<Double> lambdaValues = new ArrayList<Double>(Arrays.asList(0.01, 0.001, 0.0001));

    int modelNum = 1;
    for (Integer C : CValues) {
      for (Integer H : HValues) {
        if (C == 7 && H == 125) continue;
        for (Double lambda : lambdaValues) {
          System.out.println("----- NEXT UP: MODEL " + modelNum + " -----");
          WindowModel model = new WindowModel(C, H, 0.01, lambda, 20);
          model.initWeights();
          model.train(trainData, false, "model" + modelNum + "-Diagnostics.txt");
          model.test(trainData, "model" + modelNum + "-Train.txt", true);
          model.test(testData, "model" + modelNum + "-Test.txt", false);
          modelNum++;
        }
      }
    }


    // train and test window model
//    model.train(trainData, false); // set to true for gradient checking
//    model.test(trainData, true);
//    model.test(testData, false);
  }
}
