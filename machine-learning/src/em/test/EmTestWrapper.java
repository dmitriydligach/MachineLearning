package em.test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EmTestWrapper {

  final static int NUMBEROFFOLDS = 10; 

  public static void main(String[] args) throws IOException {
    
    double labeledOnlyAccuracy = testEm(10, 0);
    double labeledAndUnlabeledAccuracy = testEm(10, 50);
    
    System.out.println(labeledOnlyAccuracy + " vs. " + labeledAndUnlabeledAccuracy);
  }
  
  public static double testEm(int numberOfLabeledExamples, int iterations) throws FileNotFoundException {

    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
    dataset.makeAlphabets();

    Split[] splits = dataset.split(NUMBEROFFOLDS);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < NUMBEROFFOLDS; fold++) {

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

    double averageAccuracy = cumulativeAccuracy / NUMBEROFFOLDS;
    
    return averageAccuracy;
  }
}
