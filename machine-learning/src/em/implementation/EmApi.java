package em.implementation;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Alphabet;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;

public class EmApi {

  public static final int ITERATIONS = 10; // number of iterations
  
  public static double em(Dataset labeled, Dataset unlabeled, Dataset test, Alphabet labelAlphabet, Alphabet featureAlphabet) {
    
    labeled.setAlphabets(labelAlphabet, featureAlphabet);
    labeled.makeVectors();

    EmModel em = new EmModel(labelAlphabet);
    em.train(labeled);
    
    for(int iteration = 0; iteration < ITERATIONS; iteration++) {
      
      // E-step
      unlabeled.setAlphabets(labelAlphabet, featureAlphabet);
      unlabeled.makeVectors();
      em.label(unlabeled);

      // M-step
      Dataset labeledPlusUnlabeled = new Dataset(labeled.getInstances(), unlabeled.getInstances());
      labeledPlusUnlabeled.setAlphabets(labelAlphabet, featureAlphabet);
      labeledPlusUnlabeled.makeVectors();
      em.train(labeledPlusUnlabeled);
    }
    
    test.setAlphabets(labelAlphabet, featureAlphabet);
    test.makeVectors();
    double accuracy = em.test(test);

    return accuracy;
  }
  
  public static void main(String[] args) throws IOException {

    final int FOLDS = 5; // number of folds
    final int LABELED = 20; // number of labeled examples
    
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

      double accuracy = EmApi.em(labeled, unlabeled, test, dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }
    
    double accuracy = cumulativeAccuracy / FOLDS;
    System.out.println("average accuracy: " + accuracy);
  }
}
