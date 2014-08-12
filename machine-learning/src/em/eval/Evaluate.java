package em.eval;

import java.io.FileNotFoundException;
import java.io.IOException;

import semsup.eval.Constants;

public class Evaluate {

  public static void main(String[] args) throws FileNotFoundException, IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of properties file");
    } else {
      System.out.format("%15s %s\n", "properties file", args[0]);
      Constants.populate(args[0], true);  
    }

    for(String phenotype : Constants.phenotypes) {
      EvaluatePhenotype evaluate = new EvaluatePhenotype(phenotype);
      evaluate.start();
    }
  }
}
