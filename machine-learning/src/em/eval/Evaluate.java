package em.eval;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class Evaluate {

  public static final String data = Constants.ucData;
  public static final String labels = Constants.ucLabel;
  
  public static void main(String[] args) throws IOException {
    
    File file = new File(Constants.outputFile);
    if(file.exists()) {
      System.out.println(Constants.outputFile + " already exists... deleting...");
      file.delete();
    }
    
    for(int labeled = Constants.step; labeled < Constants.maxLabeled; labeled += Constants.step) {
      List<Configuration> configurations = makeConfigurations(labeled, false);
      StringBuilder output = new StringBuilder();
      output.append(String.format("%-3d ", labeled));
      for(Configuration configuration : configurations) {
        double accuracy = evaluate(configuration);
        output.append(String.format("%.4f ", accuracy));
      }
      output.append("\n");
      Files.append(output, file, Charsets.UTF_8);
    }
  }

  /**
   * Create a list of configurations for a phenotype.
   */
  public static List<Configuration> makeConfigurations(int labeled, boolean normalize) {
    
    List<Configuration> configurations = new ArrayList<Configuration>();
    
    configurations.add(new Configuration(labeled, 0, 0, normalize));
    configurations.add(new Configuration(labeled, 500, 25, normalize));
    configurations.add(new Configuration(labeled, 1000, 25, normalize));
    configurations.add(new Configuration(labeled, 3000, 25, normalize));
    configurations.add(new Configuration(labeled, 5000, 25, normalize));
    
    return configurations;
  }
  
  /**
   * Evaluate a configuration. Return n-fold CV accuracy.
   */
  public static double evaluate(Configuration configuration) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(data, labels);
    dataset.makeAlphabets();
    if(configuration.normalize) {
      dataset.normalize();
    }
    
    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(data, labels, configuration.unlabeled);
    if(configuration.normalize) {
      dataset.normalize();
    }
    
    Split[] splits = dataset.split(Constants.folds);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(configuration.labeled, new Random(100)));
      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          configuration.iterations);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    return cumulativeAccuracy / Constants.folds;
  }
  
  public static class Configuration {
    
    public int labeled;
    public int unlabeled;
    public int iterations;
    public boolean normalize;
    
    // remap all labels in source to target
    public Set<String> sourceLabels;
    public String targetLabel;
    
    public Configuration(int labeled, int unlabeled, int iterations, boolean normalize) {
      this.labeled = labeled;
      this.unlabeled = unlabeled;
      this.iterations = iterations;
      this.normalize = normalize;
    }
    
    Configuration(int labeled, int unlabeled, int iterations, boolean normalize,
                  Set<String> sourceLabels, String targetLabel) {
      this.labeled = labeled;
      this.unlabeled = unlabeled;
      this.iterations = iterations;
      this.normalize = normalize;
      this.sourceLabels = sourceLabels;
      this.targetLabel = targetLabel;
    }
  }
}
