package em.eval;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class Evaluate {

  public final static int FOLDS = 10;
  public final static int MAXLABELED = 500;
  public final static int STEP = 5;
  public final static int ITERATIONS = 25;
  
  public static final String DATAFILE = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/data.txt";
  public static final String LABELFILE = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/labels-cd.txt";
  public static final String OUTFILE = "/Users/Dima/Boston/Output/em.txt";
  
  public static final String LABELREMAP = "";
  
  public static void main(String[] args) throws IOException {
    
    File file = new File(OUTFILE);
    if(file.exists()) {
      System.out.println(OUTFILE + " already exists... deleting...");
      file.delete();
    }
    
    for(int labeled = STEP; labeled < MAXLABELED; labeled += STEP) {
      List<Configuration> configurations = new ArrayList<Configuration>();
      configurations.add(new Configuration(labeled, 0, 0, false));
      configurations.add(new Configuration(labeled, 500, ITERATIONS, false));
      configurations.add(new Configuration(labeled, 1000, ITERATIONS, false));
      configurations.add(new Configuration(labeled, 3000, ITERATIONS, false));

      StringBuilder output = new StringBuilder();
      output.append(String.format("%-3d ", labeled));
      for(Configuration configuration : configurations) {
        double accuracy = 0.123456789; // evaluate(configuration);
        output.append(String.format("%.4f ", accuracy));
      }
      output.append("\n");
      Files.append(output, file, Charsets.UTF_8);
    }
  }

  public static class Configuration {
    
    public int labeled;
    public int unlabeled;
    public int iterations;
    public boolean normalize;
    
    Configuration(int labeled, int unlabeled, int iterations, boolean normalize) {
      this.labeled = labeled;
      this.unlabeled = unlabeled;
      this.iterations = iterations;
      this.normalize = normalize;
    }
  }

  public static double evaluate(Configuration configuration) throws FileNotFoundException {

    // load labeled data
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(DATAFILE, LABELFILE);
    dataset.makeAlphabets();
    if(configuration.normalize) {
      dataset.normalize();
    }
    
    // load unlabeled data
    I2b2Dataset unlabeled = new I2b2Dataset();
    unlabeled.loadFromCSVFile(DATAFILE, LABELFILE, configuration.unlabeled);
    if(configuration.normalize) {
      dataset.normalize();
    }
    
    Split[] splits = dataset.split(FOLDS);
    double cumulativeAccuracy = 0;

    for(int fold = 0; fold < FOLDS; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();

      labeled.add(nontest.popRandom(configuration.labeled, new Random(100)));
      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          configuration.iterations);
      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    }

    double averageAccuracy = cumulativeAccuracy / FOLDS;
    return averageAccuracy;
  }
}
