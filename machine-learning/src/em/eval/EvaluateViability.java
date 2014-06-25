package em.eval;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import semsup.eval.Configuration;
import semsup.eval.Constants;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Alphabet;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EvaluateViability extends Thread {

  public String phenotype;

  public EvaluateViability(String phenotype) {
    this.phenotype = phenotype;
  }

  public void run() {

    File file = new File(Constants.outputDir + phenotype + ".txt");
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
        double accuracy;
        if(configuration.numUnlabeled == 0) {
          accuracy = 0;
        } else {
          accuracy = evaluate(configuration);
        }
        output.append(String.format("%.4f ", accuracy));
      }

      output.append("\n");      
      try {
        Files.append(output, file, Charsets.UTF_8);
      } catch (IOException e) {
        System.err.println("could not write output file!");
      }
    }
  }

  /**
   * Evaluate a configuration. Return n-fold CV accuracy.
   */
  public double evaluate(Configuration configuration) {
    
    final int devFolds = 2;

    I2b2Dataset dataset = new I2b2Dataset();
    I2b2Dataset unlabeled = new I2b2Dataset();
    try {
      dataset.loadCSVFile(configuration.dataPath, configuration.labelPath);
      unlabeled.loadFromCSVFile(configuration.dataPath, configuration.labelPath, configuration.numUnlabeled);
    } catch (FileNotFoundException e) {
      System.err.println("data file not found!");
    }
    if(configuration.sourceLabels != null) {
      // no need to remap unlabeled since there are no labels
      dataset.mapLabels(configuration.sourceLabels, configuration.targetLabel);
    }
    // make alphabets now, after labels were potentially remapped 
    dataset.makeAlphabets();

    Split[] splits = dataset.split(Constants.folds);

    double cumulativeImprovement = 0;
    for(int fold = 0; fold < Constants.folds; fold++) {
    
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      labeled.add(nontest.popRandom(configuration.numLabeled, new Random(Constants.rndSeed)));

      Split[] devSplits = labeled.split(devFolds);

      double cumulativeDevImprovement = 0;
      for(int devFold = 0; devFold < devFolds; devFold++) {
        
        double baselineAccuracy = EmAlgorithm.runAndEvaluate(
            devSplits[devFold].getPoolSet(), 
            new Dataset(),
            devSplits[devFold].getTestSet(), 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            0,
            Constants.defaultLambda);
        
        double semSupAccuracy = EmAlgorithm.runAndEvaluate(
            devSplits[devFold].getPoolSet(), 
            unlabeled,
            devSplits[devFold].getTestSet(), 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            10,
            Constants.defaultLambda);
        
        double improvement = semSupAccuracy - baselineAccuracy;
        cumulativeDevImprovement = cumulativeDevImprovement + improvement;
      }
     
      double averageDevImprovement = cumulativeDevImprovement / devFolds;
      cumulativeImprovement = cumulativeImprovement + averageDevImprovement;
    }

    return cumulativeImprovement / Constants.folds;
  }
  
  /**
   * Search for best parameters using labeled training data.
   */
  public void tune(Dataset labeled, Dataset unlabeled, Alphabet labelAlphabet, Alphabet featureAlphabet) {
    
    final int folds = 2;
    final int iterations = 10;
    
    Split[] splits = labeled.split(folds);

    double cumulativeDevImprovement = 0;
    for(int fold = 0; fold < folds; fold++) {
      
      double baselineAccuracy = EmAlgorithm.runAndEvaluate(
          splits[fold].getPoolSet(), 
          new Dataset(),
          splits[fold].getTestSet(), 
          labelAlphabet, 
          featureAlphabet,
          0,
          Constants.defaultLambda);
      
      double semSupAccuracy = EmAlgorithm.runAndEvaluate(
          splits[fold].getPoolSet(), 
          unlabeled,
          splits[fold].getTestSet(), 
          labelAlphabet,
          featureAlphabet,
          iterations,
          Constants.defaultLambda);
      
      double improvement = semSupAccuracy - baselineAccuracy;
      cumulativeDevImprovement = cumulativeDevImprovement + improvement;
    }
   
    double averageDevImprovement = cumulativeDevImprovement / folds;
  }
}
