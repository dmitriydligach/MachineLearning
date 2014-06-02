package em.eval;

public class Evaluate {

  public static void main(String[] args) {
    
    for(String phenotype : Constants.phenotypes) {
      EvaluatePhenotype evaluate = new EvaluatePhenotype(phenotype);
      evaluate.start();
    }
  }
}
