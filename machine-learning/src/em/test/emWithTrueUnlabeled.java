package em.test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class emWithTrueUnlabeled {

  public static void main(String[] args) throws IOException {
    
    for(int numLabeled = Constants.STEP; numLabeled < Constants.MAXLABELED; numLabeled += Constants.STEP) {
      double labeledOnlyAccuracy = testEm(numLabeled, 0);
      double labeledAndUnlabeledAccuracy = testEm(numLabeled, Constants.ITERATIONS);
      System.out.format("%d %.4f %.4f\n", numLabeled, labeledOnlyAccuracy, labeledAndUnlabeledAccuracy);
    }
  }
  
  public static double testEm(int numberOfLabeledExamples, int iterations) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(Constants.DATAFILE, Constants.LABELFILE);
    dataset.makeAlphabets();
    
    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(Constants.DATAFILE, Constants.LABELFILE, Constants.UNLABELED);
    
    Split[] splits = dataset.split(Constants.FOLDS);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < Constants.FOLDS; fold++) {

      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(numberOfLabeledExamples, new Random(100)));

      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          iterations);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    double averageAccuracy = cumulativeAccuracy / Constants.FOLDS;
    
    return averageAccuracy;
  }
}
