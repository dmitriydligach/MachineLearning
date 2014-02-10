package em.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Configuration {
  
  public String data;
  public String labels;
  
  public int labeled;
  public int unlabeled;
  public int iterations;
  public boolean normalize;
  
  public Set<String> source; // labels to be remapped
  public String target; // label with which to remap
  
  public Configuration(
      String data,
      String labels,
      int labeled,
      int unlabeled, 
      int iterations, 
      boolean normalize,
      Set<String> source,
      String target) {
    this.data = data;
    this.labels = labels;
    this.labeled = labeled;
    this.unlabeled = unlabeled;
    this.iterations = iterations;
    this.normalize = normalize;
    this.source = source;
    this.target = target;
  }

  /**
   * Generate a list of configurations for a given size of the labeled set.
   */
  public static List<Configuration> generateConfigurations(
      String phenotype, 
      int labeled,
      boolean normalize) throws IllegalArgumentException {

    List<Configuration> configurations = new ArrayList<Configuration>();
    String data;
    String labels;
    Set<String> source;
    String target;

    if(phenotype.equals("cd")) {
      data = Constants.cdData;
      labels = Constants.cdLabels;
      source = null;
      target = null;
    } else if(phenotype.equals("uc")) {
      data = Constants.ucData;
      labels = Constants.ucLabels;
      source = null;
      target = null;
    } else if(phenotype.equals("ms")) {
      data = Constants.msData;
      labels = Constants.msLabels;
      source = Constants.msSourceLabels;
      target = Constants.msTargetLabel;
    } else if(phenotype.equals("t2d")) {
      data = Constants.ucData;
      labels = Constants.t2dLabels;
      source = Constants.t2dSourceLabels;
      target = Constants.t2dTargetLabel;
    } else {
      throw new IllegalArgumentException("Bad phenotype!");
    }

    // make a baseline configuration (labeled data only)
    Configuration configuration0 = new Configuration(
        data,
        labels,
        labeled,
        0,
        0,
        normalize,
        source,
        target);
    configurations.add(configuration0);
    
    // make the rest of configurations
    for(int unlabeled : Constants.unlabeledSizes) {
      Configuration configuration = new Configuration(
          data,
          labels,
          labeled,
          unlabeled,
          Constants.iterations,
          normalize,
          source,
          target);
      configurations.add(configuration);
    }
    
    return configurations;
  }
}

