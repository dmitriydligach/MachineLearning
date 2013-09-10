package em;

import java.util.HashMap;
import java.util.Map;

import utils.Misc;
import data.Alphabet;
import data.Dataset;
import data.Instance;

/**
 * Implements a multinomial naive bayes classifier
 * as described in Nigam and Ghani (2000).
 * 
 * @author dmitriy dligach
 *
 */
public class EmModel {

  // number of classes
  protected int numClasses;
  // number of words (features)
  protected int numWords;
  // number of examples in training data
  protected int numInstances;
  
  // total word (feature) count in each class
  private double[] totalClassWords;

  // p(w|c) for all classes 
  protected double[][] theta;
  // p(c) for all classes
  protected double[] priors;
	
  // map labels to ints and ints to labels
  Alphabet labelAlphabet;
  
  /** 
   * Set the label alphabet here. This cannot be done in initialize()
   * using the dataset's alphabet because sometimes the training data
   * may not have all the labels that exist in the test data.
   */
	public EmModel(Alphabet labelAlphabet) {

	  this.labelAlphabet = labelAlphabet;
	}
	
	 /**
   * Initialize various counts needed for training.
   * Assumes alphabet and vectors are generated for the dataset.
   */
  protected void initialize(Dataset dataset) {
    
    numClasses = labelAlphabet.size();
    numWords = dataset.getNumberOfDimensions();
    numInstances = dataset.size();
    
    totalClassWords = new double[numClasses];
    
    priors = new double[numClasses];
    theta = new double[numClasses][numWords];
  }	
	
  
  /**
   * Train a model ...
   */
  public void train(Dataset dataset) {
    
    initialize(dataset);
    computeTotalClassWords(dataset); 
    computeTheta(dataset);
    computePriors(dataset);
  }
  
  /**
   * Classify instances in a dataset. Return accuracy.
   */
  public double test(Dataset dataset) {
    
    int correct = 0;
    int total = 0;
    
    for(Instance instance : dataset.getInstances()) {
      int prediction = classify(instance);
      if(prediction == labelAlphabet.getIndex(instance.getLabel())) {
        correct++;
      }
      total++;
    }

    return (double) correct / total;
  }
  
	/**
   * Classify instances in a dataset. Also set probability distribution
   * over classes for each instance. Return accuracy.
   */
  public double label(Dataset dataset) {
    
    int correct = 0;
    int total = 0;
    
    for(Instance instance : dataset.getInstances()) {

      double[] logSum = getUnnormalizedClassLogProbs(instance);
      double[] p = logToProb(logSum[0], logSum[1]); 
            
      Map<String, Float> classDistribution = new HashMap<String, Float>();
      for(int label = 0; label < numClasses; label++) {
        classDistribution.put(labelAlphabet.getString(label), (float)p[label]);
      }
      instance.setClassProbabilities(classDistribution); 
       
      int prediction = Misc.getIndexOfLargestElement(p);
      if(prediction == labelAlphabet.getIndex(instance.getLabel())) {
        correct++;
      }
      total++;
    }

    return (double) correct / total;
  }

	
	/**
	 * Sum from the numerator from equation 1.
	 */
	public double getWordCountInClass(Dataset dataset, int wordIndex, int classIndex) {
		
		double sum = 0;
		for(Instance instance : dataset.getInstances()) {
			String label = labelAlphabet.getString(classIndex);
			Float wordCount = instance.getDimensionValue(wordIndex);
			if(wordCount != null) {
			  float classProbability = instance.getClassProbability(label);
				sum += wordCount * classProbability; // TODO: sometimes class Probability is not a number
			}
		}
		
		return sum;
	}
	
	/**
	 * Sum from the denominator from equation 1 
	 */
	public double getTotalWordCountInClass(Dataset dataset, int classIndex) {
		
		double sum = 0;
		for(int word = 0; word < numWords; word++) {
		  double wordCount = getWordCountInClass(dataset, word, classIndex);
		  sum += wordCount;
		}
		
		return sum;
	}
	
	/**
	 * 
	 */
	public void computeTotalClassWords(Dataset dataset) {

		for(int label = 0; label < numClasses; label++) {
			totalClassWords[label] = getTotalWordCountInClass(dataset, label);
		}
	}
	
	/**
	 * Equation 1.
	 */
	public void computeTheta(Dataset dataset) {
		
		for(int label = 0; label < numClasses; label++) {
			for(int word = 0; word < numWords; word++) {
				theta[label][word] = 
						(1 + getWordCountInClass(dataset, word, label)) / 
						(numWords + totalClassWords[label]);
			}
		}
	}
	
	/**
	 * Equation 2.
	 */
	public void computePriors(Dataset dataset) {
		
		for(int classIndex = 0; classIndex < numClasses; classIndex++) {
		  double sum = 0;
			for(Instance instance : dataset.getInstances()) {
				String label = labelAlphabet.getString(classIndex);
				sum += instance.getClassProbability(label);
			}
			
			priors[classIndex] = (1 + sum) / (numClasses + numInstances);
		}
	}
	
	 /**
   * Calculate for each class:
   * 
   * p(c|w_0, ..., w_n-1) ~ p(c)p(w_0|c)p(w_1|c)...p(w_n-1|c)
   *   where w_0, ..., w_n-1 are word tokens in the document
   *   
   * Calculations are done in log space. I.e. we need to calculate:
   * log(p(c)p(w_0|c)...p(w_n-1|c)) = log(p(c)) + log(p(w_0|c)) + ... + log(p(w_n-1|c))
   * 
   * OOV words (i.e. words not seen during training) are currently ignored.
   * 
   * TODO: For the inner loop, iterate over words that exist in this vector.
   * For some of them (OOV words), p(w|c) will be unknown (ignore them).
   */
  public double[] getUnnormalizedClassLogProbs(Instance instance) {
    
    double[] logSum = new double[numClasses];

    for(int label = 0; label < numClasses; label++) {
      logSum[label] = Math.log10(priors[label]); 

      // iterate over words that were seen during training
      for(int word = 0; word < numWords; word++) {
        Float wordCount = instance.getDimensionValue(word);
        if(wordCount == null) {
          // this instance does not contain the current word; just go on to
          // the next word (i.e. pretend the wordCount for this word is zero)
          continue;
        }
        
        double wordProb = theta[label][word]; // TODO: sometimes theta not a number
        logSum[label] += wordCount * Math.log10(wordProb);
        
//        System.out.println("prob=" + logSum[label] + ", wordProb=" + wordProb + ", wordCount=" + wordCount);
      }
    }
    
    return logSum;
  }
	
  /**
   * Convert two unnormalized log probs to probabilities.
   */
  public double[] logToProb(double logP0, double logP1) {

    double[] p = new double[2];
    
    // odds0 is p0/p1 or p0 / (1 - p0)
    double odds0 = Math.pow(10, logP0 - logP1);

    p[0] = odds0 / (1 + odds0);
    p[1] = 1 / (1 + odds0);
    
    return p;
  }
  
  /**
   * Classify a document using a multinomial naive bayes model. 
   */
  public int classify(Instance instance) {
    double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
    return Misc.getIndexOfLargestElement(classLogProbs);
  }
}
