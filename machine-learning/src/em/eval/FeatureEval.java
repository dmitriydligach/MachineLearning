package em.eval;

import java.io.IOException;
import java.util.HashSet;

import semsup.eval.Constants;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class FeatureEval {
	
	public static void main(String[] args) throws IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of the properties file");
    } else {
      Constants.populate(args[0], false);  
    }
	  
		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile(Constants.cdData, Constants.cdLabels);
		dataset.makeAlphabets();
		
		Split[] splits = dataset.split(Constants.folds);
		
		for(int fold = 0; fold < 1; fold++) {
			Dataset trainSet = splits[fold].getPoolSet();
			Dataset testSet = splits[fold].getTestSet();
			
			trainSet.makeAlphabets();
			trainSet.makeVectors();
			testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
			testSet.makeVectors();

			trainSet.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
			EmModel classifier = new EmModel(dataset.getLabelAlphabet(), 1.0);
			classifier.train(trainSet);
			double accuracy = classifier.test(testSet);
			System.out.format("accuracy: %.4f\n", accuracy);
			
			classifier.computeFeatureWeights(dataset.getFeatureAlphabet());
		}
	}
}
