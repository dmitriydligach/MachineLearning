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
          accuracy = Evaluation.evaluateBaseline(configuration);
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

      Split[] devSplits = labeled.split(2);
      double cumulativeDevImprovement = 0;
      
      for(int devFold = 0; devFold < 2; devFold++) {
        
        double baselineAccuracy = EmAlgorithm.runAndEvaluate(
            devSplits[devFold].getPoolSet(), 
            new Dataset(),
            devSplits[devFold].getTestSet(), 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            0);
        
        double semSupAccuracy = EmAlgorithm.runAndEvaluate(
            devSplits[devFold].getPoolSet(), 
            unlabeled,
            devSplits[devFold].getTestSet(), 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            configuration.numIterations);
        
        double improvement = semSupAccuracy - baselineAccuracy;
        cumulativeDevImprovement = cumulativeDevImprovement + improvement;
      }
      
      double averageDevImprovement = cumulativeDevImprovement / 2;
      cumulativeImprovement = cumulativeImprovement + averageDevImprovement;
    }

    return cumulativeImprovement / Constants.folds;
  }
}
