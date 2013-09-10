package i2b2active;

import java.io.IOException;
import java.util.Random;

import classifier.NaiveBayesClassifier;
import cv.LearningCurve;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;

public class PassiveCurve {
	
	public static void main(String[] args) throws IOException {

		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile(Constants.dataFile, Constants.labelFile);
		dataset.makeAlphabets();
		
		// splits should be the same for random sampling and active learning
		Split[] splits = dataset.split(Constants.totalFolds, new Random(Constants.rndSeedForSplitting));
		LearningCurve learningCurve = new LearningCurve();
		
		for(int fold = 0; fold < Constants.totalFolds; fold++) {
			learningCurve.startNewFold();
			
			Dataset pool = splits[fold].getPoolSet();
			Dataset test = splits[fold].getTestSet();
			Dataset train = new Dataset();

			// seed (should be the same for active learning and random sampling)
			train.add(pool.popRandom(Constants.seedSize, new Random(Constants.rndSeedForSeeding)));
			
			while(true) {
				NaiveBayesClassifier classifier = new NaiveBayesClassifier(dataset.getLabelAlphabet());
				
				// train a model
				train.makeAlphabets();
				train.makeVectors();
				classifier.train(train);
				
				// get the model's performance on the test set
				test.setAlphabets(train.getLabelAlphabet(), train.getFeatureAlphabet());
				test.makeVectors();
				double accuracy = classifier.test(test);
			
				// record performance at this size of training set
				learningCurve.add(train.size(), (float)accuracy);
				
				// was last pool instance added to train set during previous iteration?
				if(pool.size() == 0) {
					break;
				}
				
				// add a randoml example
				train.add(pool.popRandom(1, new Random())); 
			}
		}
		
		learningCurve.average();
		learningCurve.saveAveragedCurve(Constants.outputFileRandom);
		System.out.println("random sampling done!");
	}
}
