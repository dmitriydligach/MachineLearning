package em.features;

import java.io.IOException;
import java.util.Random;

import semsup.eval.Constants;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EmFeatureEval {
	
  public static final int NUMLABELED = 50;
  public static final int NUMUNLABELED = 1000;
  
	public static void main(String[] args) throws IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of the properties file");
    } else {
      Constants.populate(args[0], false);  
    }
	  
		I2b2Dataset dataset = new I2b2Dataset();
		I2b2Dataset unlabeled = new I2b2Dataset();
		dataset.loadCSVFile(Constants.cdData, Constants.cdLabels);
		unlabeled.loadFromCSVFile(Constants.cdData, Constants.cdLabels, NUMUNLABELED);
    // dataset.mapLabels(Constants.msSourceLabels, Constants.msTargetLabel);
		dataset.makeAlphabets();
		
		Split[] splits = dataset.split(Constants.folds);
		double cumulAcc = 0;
		
		for(int fold = 0; fold < Constants.folds; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      labeled.add(nontest.popRandom(NUMLABELED, new Random(Constants.rndSeed)));

      double accuracy = EmAlgorithm.runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          25,
          1);

      cumulAcc = cumulAcc + accuracy;
		}
		
    System.out.format("%d-fold cv accuracy: %.4f\n", Constants.folds, cumulAcc / Constants.folds);
	}
}
