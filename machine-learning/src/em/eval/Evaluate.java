package em.eval;

import java.io.FileNotFoundException;
import java.io.IOException;

import semsup.eval.Constants;

public class Evaluate {

  public static void main(String[] args) throws FileNotFoundException, IOException {

    Constants.populate(Constants.definitions);

    for(String phenotype : Constants.phenotypes) {
      EvaluatePhenotype evaluate = new EvaluatePhenotype(phenotype);
      evaluate.start();
    }
  }
}
