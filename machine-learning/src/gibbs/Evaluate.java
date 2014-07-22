package gibbs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.StatUtils;

import semsup.eval.Configuration;
import semsup.eval.Constants;
import semsup.eval.Evaluation;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;

public class Evaluate {

  public static final String phenotype = "t2d";
  
  public static void main(String[] args) throws IOException {
    
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
        double accuracy = 0.0;
        if(configuration.numUnlabeled == 0) {
          double[] foldAccuracy = Evaluation.evaluateBaseline(configuration);
          accuracy = StatUtils.mean(foldAccuracy);
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
}
