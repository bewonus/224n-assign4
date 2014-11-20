package cs224n.deep;
import java.util.*;

public class Datum {

  public final String word;
  public final String label;
  
  public Datum(String word, String label) {
    this.word = word;
    this.label = label;
  }

  @Override
  public boolean equals(Object o) {

    if (this == o) return true;
    if (!(o instanceof Datum)) return false;

    final Datum datum = (Datum) o;

    return(word.equals(datum.word) && label.equals(datum.label));
  }

  @Override
  public int hashCode() {
    return word.hashCode() + label.hashCode();
  }
}