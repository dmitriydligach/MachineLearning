package gibbs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.eval.Configuration;
import em.eval.Constants;
import em.implementation.EmModel;

public class Evaluate {

  public static final String phenotype = "cd";
  public static final boolean normalize = false;
  
  public static void main(String[] args) throws IOException {
    
    File file = new File(Constants.outputDir + phenotype + (normalize ? "-normalized" : "") + ".txt");
    if(file.exists()) {
      System.out.println(file.getName() + " already exists... deleting...");
      file.delete();
    }
    
    for(int labeled = Constants.step; labeled < Constants.maxLabeled; labeled += Constants.step) {
      // for a fixed number of labeled examples, create configurations with varying number of unlabeled examples 
      List<Configuration> configurations = Configuration.createConfigurations(phenotype, labeled);
      
      StringBuilder output = new StringBuilder();
      output.append(String.format("%-3d ", labeled));
      
      for(Configuration configuration : configurations) {
        double accuracy = 0.0;
        if(configuration.numUnlabeled == 0) {
          accuracy = evaluateBaseline(configuration);
        } else {
          accuracy = evaluateSampler(configuration);
        }
        output.append(String.format("%.4f ", accuracy));
      }
      
      output.append("\n");      
      Files.append(output, file, Charsets.UTF_8);
    }
  }
  
  /**
   * Evaluate a configuration. Return n-fold CV accuracy.
   */
  public static double evaluateSampler(Configuration configuration) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(configuration.dataPath, configuration.labelPath);

    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(configuration.dataPath, configuration.labelPath, configuration.numUnlabeled);
    
    if(normalize) {
      dataset.normalize();
      unlabeled.normalize();
    }
    if(configuration.sourceLabels != null) {
      // no need to remap unlabeled since there are no labels
      dataset.mapLabels(configuration.sourceLabels, configuration.targetLabel);
    }
    
    // make alphabets now, after labels were potentially remapped 
    dataset.makeAlphabets();

    Split[] splits = dataset.split(Constants.folds);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(configuration.numLabeled, new Random(100)));
      test.hideLabels();
      
      Model sampler = new Model(labeled, unlabeled, test, dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      sampler.run();
      double accuracy = sampler.evaluate();
      
      test.restoreLabels();
      cumulativeAccuracy += accuracy;
    }

    return cumulativeAccuracy / Constants.folds;    
  }
  
  /**
   * Use labeled data only.
   */
  public static double evaluateBaseline(Configuration configuration) throws FileNotFoundException {

    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(configuration.dataPath, configuration.labelPath);
    
    if(normalize) {
      dataset.normalize();
    }
    if(configuration.sourceLabels != null) {
      dataset.mapLabels(configuration.sourceLabels, configuration.targetLabel);
    }
    
    // make alphabets now, after labels were potentially remapped 
    dataset.makeAlphabets();

    Split[] splits = dataset.split(Constants.folds);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(configuration.numLabeled, new Random(100)));

      labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
      labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      labeled.makeVectors();
      
      EmModel classifier = new EmModel(dataset.getLabelAlphabet());
      classifier.train(labeled);
 
      test.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      test.makeVectors();
      double accuracy = classifier.test(test);

      cumulativeAccuracy += accuracy;
    }

    return cumulativeAccuracy / Constants.folds;    
  }  
}
