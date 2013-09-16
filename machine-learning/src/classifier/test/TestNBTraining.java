package classifier.test;

import java.io.IOException;

import classifier.NaiveBayesClassifier;
import data.Dataset;

public class TestNBTraining {
	
	public static void main(String[] args) throws IOException {

		Dataset dataset = new Dataset();
		dataset.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/cui-features/features.txt");
		dataset.makeAlphabets();
		dataset.makeVectors();
		
		NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier(dataset.getLabelAlphabet());
		naiveBayesClassifier.train(dataset);
		naiveBayesClassifier.printModel(dataset.getFeatureAlphabet());
	}
}
