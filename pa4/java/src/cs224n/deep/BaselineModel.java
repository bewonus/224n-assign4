package cs224n.deep;

import java.io.*;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class BaselineModel {

  private HashMap<Datum, Integer> counts = new HashMap<Datum, Integer>();
  private List<String> labels = new ArrayList<String>(Arrays.asList("O", "LOC", "MISC", "ORG", "PER"));

  public BaselineModel() { }

  /**
   * Trains the baseline model. For each word, stores the frequency of each of label for that word.
   */
  public void train(List<Datum> trainData) {
    for (Datum datum : trainData) {
      if (counts.containsKey(datum)) {
        counts.put(datum, counts.get(datum) + 1);
      } else {
        counts.put(datum, 1);
      }
    }
  }

  /**
   * Predict named-entity labels on the testData based on the most frequent tag for each word.
   */
  public void test(List<Datum> testData) {
    try {
      File file = new File("baseline.txt");
      BufferedWriter output = new BufferedWriter(new FileWriter(file));

      for (Datum datum : testData) {
        boolean labelSeen = false;
//        boolean discard = false;
        int max = 0;
        String predictLabel = "UNK"; // O?
        for (String label : labels) {
          Datum temp = new Datum(datum.word, label);
          if (counts.containsKey(temp)) {
            if (labelSeen) {
              predictLabel = "UNK"; // UNK?
//              discard = true;
              break;
            }else{
              labelSeen = true;
              int count = counts.get(temp);
              if (count > max) {
                max = count;
                predictLabel = label;
              }
            }
          }
        }
//        if (discard) continue;
        output.write(datum.word + "\t" + datum.label + "\t" + predictLabel + "\n");
      }
      output.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
