package gibbs;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class EvaluateOld {

  public static final String dataPath = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/data.txt";
  public static final String labelPath = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/labels-cd.txt";
  public static final String outfile = "/Users/Dima/Boston/Output/gibbs.txt";
  public static final int numUnlabeled = 1000;
  public static final int numFolds = 10;
  public static final int step = 5;
  public static final int maxLabeled = 300;

  public static void main(String[] args) throws IOException {

    File file = new File(outfile);
    if(file.exists()) {
      System.out.println(outfile + " already exists... deleting...");
      file.delete();
    }

    for(int numLabeled = step; numLabeled < maxLabeled; numLabeled += step) {
      double labeledOnlyAccuracy = baseline(numLabeled);
      double labeledAndUnlabeledAccuracy = evaluate(numLabeled);
      String out = String.format("%d %.4f %.4f\n", numLabeled, labeledOnlyAccuracy, labeledAndUnlabeledAccuracy);
      Files.append(out, file, Charsets.UTF_8);
    }
  }

  public static double evaluate(int numLabeled) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(dataPath, labelPath);
    dataset.makeAlphabets();

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
      test.hideLabels();
      
      Model sampler = new Model(labeled, unlabeled, test, dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      sampler.run();
      double accuracy = sampler.evaluate();
      
      test.restoreLabels();
      cumulativeAccuracy += accuracy;
    }

    return cumulativeAccuracy / numFolds;    
  }

  /**
   * Use labeled data only.
   */
  public static double baseline(int numLabeled) throws FileNotFoundException {

    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(dataPath, labelPath);
    dataset.makeAlphabets();

    Split[] splits = dataset.split(numFolds);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < numFolds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(numLabeled, new Random(100)));

      labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
      labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      labeled.makeVectors();
      
      EmModel classifier = new EmModel(dataset.getLabelAlphabet());
      classifier.train(labeled);
 
      test.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
      test.makeVectors();
      double accuracy = classifier.test(test);

      cumulativeAccuracy += accuracy;
    }

    return cumulativeAccuracy / numFolds;    
  }  
}
