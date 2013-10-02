package em.test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class TestEmAlgorithm {

  final static int FOLDS = 10; 
  final static int EMITERATIONS = 50;

  public static void main(String[] args) throws IOException {
    
    for(int numberOfLabeledExamples = 5; numberOfLabeledExamples < 100; numberOfLabeledExamples += 5) {
      double labeledOnlyAccuracy = testEm(numberOfLabeledExamples, 0);
      double labeledAndUnlabeledAccuracy = testEm(numberOfLabeledExamples, EMITERATIONS);
      System.out.format("%d: %.3f vs %.3f\n", numberOfLabeledExamples, labeledOnlyAccuracy, labeledAndUnlabeledAccuracy);
    }
  }
  
  public static double testEm(int numberOfLabeledExamples, int iterations) throws FileNotFoundException {

    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < FOLDS; fold++) {

      Dataset labeled = new Dataset();
      Dataset unlabeled = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(unlabeled.popRandom(numberOfLabeledExamples, new Random(100)));
      labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));

      labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      labeled.makeVectors();

      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled, 
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          iterations);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    double averageAccuracy = cumulativeAccuracy / FOLDS;
    
    return averageAccuracy;
  }
}
