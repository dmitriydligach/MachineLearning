package semsup.eval;

import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class Evaluation {

  /**
   * Baseline evaluation for semi-supervised learning. Train a model
   * using labeled data only and returned the performance for each fold. 
   */
  public static double[] evaluateBaseline(Configuration configuration) {

    I2b2Dataset dataset = new I2b2Dataset();
    try {
      dataset.loadCSVFile(configuration.dataPath, configuration.labelPath);
    } catch (FileNotFoundException e) {
      System.err.println("could not load data: " + configuration.dataPath);
    }
    if(configuration.sourceLabels != null) {
      dataset.mapLabels(configuration.sourceLabels, configuration.targetLabel);
    }
    // make alphabets now, after labels were potentially remapped 
    dataset.makeAlphabets();

    Split[] splits = dataset.split(Constants.folds);
    double[] foldAccuracy = new double[Constants.folds];
    
    for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(configuration.numLabeled, new Random(Constants.rndSeed)));

      labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
      labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      labeled.makeVectors();
      
      EmModel classifier = new EmModel(dataset.getLabelAlphabet(), Constants.defaultLambda);
      classifier.train(labeled);
 
      test.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      test.makeVectors();
      foldAccuracy[fold] = classifier.test(test);
    }

    return foldAccuracy;    
  }  
}
