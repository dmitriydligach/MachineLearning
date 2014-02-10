package em.eval;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class Evaluate {

  public static final String phenotype = "cd";
  public static final boolean normalize = true;
  
  public static void main(String[] args) throws IOException {
    
    File file = new File(Constants.outputDir + phenotype + (normalize ? "-normalized" : "") + ".txt");
    if(file.exists()) {
      System.out.println(file.getName() + " already exists... deleting...");
      file.delete();
    }
    
    for(int labeled = Constants.step; labeled < Constants.maxLabeled; labeled += Constants.step) {
      
      // generate experimental configurations for the given number of labeled examples
      List<Configuration> configurations = Configuration.generateConfigurations(phenotype, labeled, normalize);
      
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
   * Evaluate a configuration. Return n-fold CV accuracy.
   */
  public static double evaluate(Configuration configuration) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(configuration.data, configuration.labels);
    dataset.makeAlphabets();
    
    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(configuration.data, configuration.labels, configuration.unlabeled);
    
    if(configuration.normalize) {
      dataset.normalize();
      unlabeled.normalize();
    }
    if(configuration.source != null) {
      dataset.mapLabels(configuration.source, configuration.target);
      unlabeled.mapLabels(configuration.source, configuration.target);
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
}
