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

public class EvaluatePhenotype extends Thread {

  public String phenotype;
  public final boolean normalize = false;

  public EvaluatePhenotype(String phenotype) {
    this.phenotype = phenotype;
  }

  public void run() {

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
        double accuracy = evaluate(configuration);
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

      labeled.add(nontest.popRandom(configuration.numLabeled, Constants.rndSource));
      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          configuration.numIterations);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    return cumulativeAccuracy / Constants.folds;
  }
}
