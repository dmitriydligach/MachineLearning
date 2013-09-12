package em.test;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class TestEmAlgorithm {

  public static void main(String[] args) throws IOException {

    final int FOLDS = 5; // number of folds
    final int LABELED = 5; // number of labeled examples
    final int ITERATIONS = 25; // number of iterations
        
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS, new Random(100));
    double cumulativeAccuracy = 0;
    
    for(int fold = 0; fold < FOLDS; fold++) {

      Dataset labeled = new Dataset();
      Dataset unlabeled = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      
      labeled.add(unlabeled.popRandom(LABELED, new Random(100)));
      labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
      
      labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      labeled.makeVectors();

      double accuracy = EmAlgorithm.em(labeled, 
                                 unlabeled, 
                                 test, 
                                 dataset.getLabelAlphabet(), 
                                 dataset.getFeatureAlphabet(),
                                 ITERATIONS);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }
    
    double accuracy = cumulativeAccuracy / FOLDS;
    System.out.println("average accuracy: " + accuracy);
  }
}
