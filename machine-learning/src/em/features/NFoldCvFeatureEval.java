package em.features;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import semsup.eval.Constants;
import utils.CuiLookup;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.Ordering;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class NFoldCvFeatureEval {
	
	public static void main(String[] args) throws IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of the properties file");
    } else {
      Constants.populate(args[0], false);  
    }
	  
		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile(Constants.msData, Constants.msLabels);
    dataset.mapLabels(Constants.msSourceLabels, Constants.msTargetLabel);
		dataset.makeAlphabets();
		
		Split[] splits = dataset.split(Constants.folds);
		double cumulAcc = 0;
		
		// run just one fold and look at feature weights
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
			cumulAcc = cumulAcc + accuracy;
			
			Map<String, Double> featureWeights = classifier.computeFeatureWeights(dataset.getFeatureAlphabet());
	    List<String> featureNamesSortedByWeight = new ArrayList<String>(featureWeights.keySet());
	    Function<String, Double> getValue = Functions.forMap(featureWeights);
	    Collections.sort(featureNamesSortedByWeight, Ordering.natural().reverse().onResultOf(getValue));
			
	    CuiLookup mapper = null;
	    mapper = new CuiLookup("resources/snomed-only-uniq-codes.txt");
	    for(String feature : featureNamesSortedByWeight) {
	      String capitalizedNegationRemoved = feature.replace("c", "C").replace("-", "");
	      String text = mapper.getTerm(capitalizedNegationRemoved);
	      if(text == null) {
	        text = "n/a";
	      }
	      System.out.format("%12s %-60s %.4f\n", feature, text, featureWeights.get(feature));
	    }
		}
		
    // System.out.format("%d-fold cv accuracy: %.4f\n", Constants.folds, cumulAcc / Constants.folds);
	}
}
