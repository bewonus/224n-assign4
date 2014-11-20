package cs224n.deep;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.ejml.simple.*;


public class FeatureFactory {


  private FeatureFactory() {

  }


  static List<Datum> trainData;

  /** Do not modify this method **/
  public static List<Datum> readTrainData(String filename) throws IOException {
    if (trainData == null) trainData = read(filename);
    return trainData;
  }

  static List<Datum> testData;

  /** Do not modify this method **/
  public static List<Datum> readTestData(String filename) throws IOException {
    if (testData == null) testData = read(filename);
    return testData;
  }

  private static List<Datum> read(String filename)
    throws FileNotFoundException, IOException {
    // TODO: you'd want to handle sentence boundaries
    List<Datum> data = new ArrayList<Datum>();
    BufferedReader in = new BufferedReader(new FileReader(filename));
    for (String line = in.readLine(); line != null; line = in.readLine()) {
      if (line.trim().length() == 0) {
        continue;
      }
      String[] bits = line.split("\\s+");
      String word = bits[0];
      String label = bits[1];

      Datum datum = new Datum(word, label);
      data.add(datum);
    }

    return data;
  }


  // Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
  // Note: SimpleMatrix is 0 indexed
  static SimpleMatrix allVecs; //access it directly in WindowModel

  public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
    // already initialized, so just return
    if (allVecs != null) return allVecs;

    // read vecFilename
    int col = 0;
    BufferedReader in = new BufferedReader(new FileReader(vecFilename));
    for (String line = in.readLine(); line != null; line = in.readLine()) {
      if (line.trim().length() == 0) {
        continue;
      }

      // split each line into set of n (== 50) doubles
      String[] bits = line.split("\\s+");
      int n = bits.length;

      // initialize allVecs as 0 matrix of dimension n x |V|
      if (col == 0) {
        allVecs = new SimpleMatrix(n, wordToNum.size());
      }

      // set each entry of allVecs according to vecFilename
      for (int row = 0; row < n; row++) {
        allVecs.set(row, col, Double.parseDouble(bits[row]));
      }
      col++;
    }

    return allVecs;
  }

  // might be useful for word to number lookups, just access them directly in WindowModel
  public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>();
  public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

  public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
    // read vocabFilename
    BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
    int index = 0;
    for (String line = in.readLine(); line != null; line = in.readLine()) {
      if (line.trim().length() == 0) {
        continue;
      }
      // store vocab word and index in both hashmaps
      wordToNum.put(line, index);
      numToWord.put(index, line);
      index++;
    }
    return wordToNum;
  }


}
