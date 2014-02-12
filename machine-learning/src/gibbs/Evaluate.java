package gibbs;

import java.io.FileNotFoundException;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;

public class Evaluate {

  public static final String dataPath = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/data.txt";
  public static final String labelPath = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/labels-cd.txt";
  public static final int numUnlabeled = 1000;
  public static final int numLabeled = 20;
  public static final int numFolds = 10;
  
  public static void main(String[] args) throws FileNotFoundException {

    double averageAccuracy = evaluate();
    System.out.println("average accuracy: " + averageAccuracy);
  }

  public static double evaluate() throws FileNotFoundException {
    
    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(dataPath, labelPath);
    
    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(dataPath, labelPath, numUnlabeled);
    
    Split[] splits = dataset.split(numFolds);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < numFolds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(numLabeled, new Random(100)));

      Model sampler = new Model(labeled, unlabeled, test);
      sampler.run();
      
      // double accuracy = sampler.evaluate();
      // cumulativeAccuracy += accuracy;
    }

    return cumulativeAccuracy / numFolds;    
  }
}
