package em.eval;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import semsup.eval.Configuration;
import semsup.eval.Constants;
import semsup.eval.Evaluation;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Alphabet;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EvaluatePhenotype extends Thread {

  public String phenotype;

  public EvaluatePhenotype(String phenotype) {
    this.phenotype = phenotype;
  }

  public void run() {

    File file = new File(Constants.outputDir + phenotype + ".txt");
    if(file.exists()) {
      System.out.println(file.getName() + " already exists... deleting...");
      file.delete();
    }

    for(int labeled = Constants.step; labeled < Constants.maxLabeled; labeled += Constants.step) {
      // configurations with varying number of unlabeled examples for a fixed number of labeled examples 
      List<Configuration> configurations = Configuration.createConfigurations(phenotype, labeled);
      StringBuilder output = new StringBuilder();
      output.append(String.format("%-3d ", labeled));
      for(Configuration configuration : configurations) {
        double accuracy;
        if(configuration.numUnlabeled == 0) {
          accuracy = Evaluation.evaluateBaseline(configuration);
        } else {
          accuracy = evaluate(configuration);
        }
        output.append(String.format("%.4f ", accuracy));
      }
      try {
        Files.append(output + "\n", file, Charsets.UTF_8);
      } catch (IOException e) {
        System.err.println("could not write output file!");
      }
    }
  }

  /**
   * Evaluate a configuration. Return n-fold CV accuracy.
   */
  public double evaluate(Configuration configuration) {

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
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      labeled.add(nontest.popRandom(configuration.numLabeled, new Random(Constants.rndSeed)));
      
      double lambda;
      if(Constants.gridSearch) {
        lambda = findBestLambda(labeled, unlabeled, dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      } else {
        lambda = Constants.defaultLambda;
      }
      
      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          configuration.numIterations,
          lambda);
      
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    return cumulativeAccuracy / Constants.folds;
  }
  
  /**
   * Search for best lambda using labeled training data.
   * Begin with lambda=0 (i.e. unlabeled data has zero weight). 
   * New value of lambda is returned if its performance is higher than a threshold.
   */
  public double findBestLambda(Dataset labeled, Dataset unlabeled, Alphabet labelAlphabet, Alphabet featureAlphabet) {

    // unlabeled data has no effect by default
    double bestLambda = 0; 
    double bestAccuracy = evaluateLambda(labeled, unlabeled, labelAlphabet, featureAlphabet, bestLambda);
    
    // now try the other values and see if they differ enough
    for(double lambda : Constants.lambdas) {
      double accuracy = evaluateLambda(labeled, unlabeled, labelAlphabet, featureAlphabet, lambda);
      if((accuracy - bestAccuracy) > Constants.delta) { 
        bestAccuracy = accuracy;
        bestLambda = lambda;
      }
    }

    return bestLambda;
  }
  
  /**
   * Evaluate a specific value of lambda using n-fold CV.
   */
  public double evaluateLambda(Dataset labeled, Dataset unlabeled, Alphabet labelAlphabet, Alphabet featureAlphabet, double lambda) {
    
    Split[] splits = labeled.split(Constants.devFolds);
    double cumulativeAccuracy = 0;
    
    for(int fold = 0; fold < Constants.devFolds; fold++) {      
      double accuracy = EmAlgorithm.runAndEvaluate(
          splits[fold].getPoolSet(), 
          unlabeled,
          splits[fold].getTestSet(), 
          labelAlphabet,
          featureAlphabet,
          Constants.devIterations,
          lambda);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }
   
    return cumulativeAccuracy / Constants.devFolds;
  }
}
