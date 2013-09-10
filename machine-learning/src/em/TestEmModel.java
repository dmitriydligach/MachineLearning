package em;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import data.I2b2Dataset;
import data.Instance;
import data.Split;

public class TestEmModel {
	
	public static void main(String[] args) throws IOException {

		final int N = 5; // number of folds
		
		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
		dataset.makeAlphabets(); // need label alphabet to init NB classifier
		
		Split[] splits = dataset.split(N, new Random(100));
		System.out.println("done splitting");
		
		double cumulativeAccuracy = 0;
		for(int fold = 0; fold < N; fold++) {
			Dataset trainSet = splits[fold].getPoolSet();
			Dataset testSet = splits[fold].getTestSet();
			
			trainSet.makeAlphabets();
			trainSet.makeVectors();

			testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
			testSet.makeVectors();
			
			for(Instance instance : trainSet.getInstances()) {
				instance.setClassProbabilities(new HashSet<String>(trainSet.getLabelAlphabet().getStrings()));
			}
			
			EmModel classifier = new EmModel(dataset.getLabelAlphabet());
			
			classifier.train(trainSet);
			double accuracy = classifier.test(testSet);
			
			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / N;
		System.out.println("average accuracy:\t" + accuracy);
	}
}
