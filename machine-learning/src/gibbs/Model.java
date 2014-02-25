package gibbs;

import java.util.HashSet;

import cc.mallet.types.Dirichlet;
import data.Alphabet;
import data.Dataset;
import data.Instance;
import em.implementation.EmModel;

/**
 * Implements a naive bayes model for a gibbs sampler.
 * 
 * @author dmitriy dligach
 */
public class Model {
  
  public static final int numSamples = 10;
  
  // hyperparameters of beta distribution
  public static final double[] betaParams = {1, 1};  
  // hyperparameters of dirichlet distribution
  public static final double[] dirichletParams = {1, 1};

	// number of classes
	protected int numClasses;
	// number of words (features)
	protected int numWords;
	// number of examples in training data
	protected int numInstances;
	
	// word (feature) counts for each class
	protected double[][] wordCounts;
	// number of examples for each class
	protected int[] labelCounts;
	// total word (feature) count in each class
	private int[] totalClassWords;

	// p(w|c) for all classes 
	protected double[][] theta;
	// p(c) for all classes
	protected double[] priors;

	// map labels to ints and ints to labels
	Alphabet labelAlphabet;
	// word to ints mapping
	Alphabet featureAlphabet;

	// may not need a class member
	public Dataset labeled;
	public Dataset unlabeled;
	public Dataset test;
	public Dataset sampled; 
	
	// constructor
	public Model(Dataset labeled, Dataset unlabeled, Dataset test, Alphabet labelAlphabet, Alphabet featureAlphabet) {

	  test.hideLabels();
	  
	  this.labeled = labeled;
	  this.unlabeled = unlabeled;
	  this.test = test;
	  
	  this.labelAlphabet = labelAlphabet;
	  this.featureAlphabet = featureAlphabet;

	  numClasses = labelAlphabet.size();
	  numWords = featureAlphabet.size(); 
	  numInstances = labeled.size() + unlabeled.size() + test.size();
	  
	  labelCounts = new int[numClasses];
	  wordCounts = new double[numClasses][numWords];
	  totalClassWords = new int[numClasses];

	  priors = new double[numClasses];
	  theta = new double[numClasses][numWords];
	}
	
	public void initialize() {
	  
	  // label unlabeled examples
	  labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(labelAlphabet.getStrings()));
	  labeled.setAlphabets(labelAlphabet, featureAlphabet);
	  labeled.makeVectors();
	  
	  EmModel classifier = new EmModel(labelAlphabet);
	  classifier.train(labeled);
	    
	  unlabeled.setAlphabets(labelAlphabet, featureAlphabet);
	  unlabeled.makeVectors();
	  
	  test.setAlphabets(labelAlphabet, featureAlphabet);
	  test.makeVectors();
	  
	  classifier.label2(unlabeled);
	  classifier.label2(test);
	  
	  Dataset all = new Dataset(labeled.getInstances(), unlabeled.getInstances(), test.getInstances());
	  all.setAlphabets(labelAlphabet, featureAlphabet);
	  all.makeVectors();
	  
	  // compute counts
	  computeLabelCounts(all);
	  computeWordCounts(all);
	  computeTotalClassWords(all);
	  
	  computeTheta();
	  
	  // instances that need to be sampled
	  sampled = new Dataset(unlabeled.getInstances(), test.getInstances());
	}
	
	public void sample() {

	  for(Instance instance : sampled.getInstances()) {

	    // subtract this instance's word counts and label counts
	    int oldLabel = labelAlphabet.getIndex(instance.getLabel());
	    labelCounts[oldLabel]--;
	    for(int word = 0; word < numWords; word++) {
	      Float wordCount = instance.getDimensionValue(word);
	      if(wordCount == null) {
	        continue;
	      }
	      wordCounts[oldLabel][word] -= wordCount;
	    }
	    
	    double[] logSum = getUnnormalizedLogProbForClasses(instance); 
	    double[] p = logToProb(logSum[0], logSum[1]);
	    int newLabel = Math.random() < p[0] ? 0 : 1;
	    instance.setLabel(labelAlphabet.getString(newLabel));
	    
	    // keep track of lables for the test set
	    if(instance.getMisc() != null) {
	      instance.labelList.add(newLabel);
	    }
	   
	    // add counts back
	    labelCounts[newLabel]++;
	    for(int word = 0; word < numWords; word++) {
	      Float wordCount = instance.getDimensionValue(word);
	      if(wordCount == null) {
	        continue;
	      }
	      wordCounts[newLabel][word] += wordCount;
	    }

	    // sample thetas
	    double[][] dirParams = new double[numClasses][numWords];
	    for(int label = 0; label < numClasses; label++) {
	      for(int word = 0; word < numWords; word++) {
	        dirParams[label][word] = wordCounts[label][word] + dirichletParams[label];
	      }
	      Dirichlet dir = new Dirichlet(dirParams[label]);
	      theta[label] = dir.nextDistribution();
	    }
	  }
	  
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
	
  private double[] getUnnormalizedLogProbForClasses(Instance instance){
    // equation 49 from "Gibbs Sampling for the Uninitiated" by Resnik and Hardisty (June 2010 version)
    
    double[] logSum = new double[numClasses];
    
    for(int label = 0; label < numClasses; label++) {
      
      double outer = (labelCounts[label] + betaParams[label] - 1) / 
          (double)(numInstances + betaParams[0] + betaParams[1] - 1);

      logSum[label] = Math.log10(outer);
      
      for(int word = 0; word < numWords; word++) {
        double inner = wordCounts[label][word] * Math.log10(theta[label][word]);
        logSum[label] += inner;
      }
    }
    
    return logSum;
  }
  
	/**
	 * Compute p(w|c) parameters.
	 * 
	 * Smoothing is needed because feature alphabet is computed based
	 * on the entire dataset (i.e. all classes). If no smoothing is done,
	 * for some words, class probability p(w|c) will be zero because some 
	 * words are only seen in the instances of one class. For classification,
	 * we need to ensure that no p(w|c) in the model is zero.
	 * 
	 * During classification, there will be OOV words (i.e. words that were not
	 * seen during training). These words will be just ignored.
	 */
	public void computeTheta() {
		
		for(int label = 0; label < numClasses; label++) {
			for(int word = 0; word < numWords; word++) {
				theta[label][word] = (wordCounts[label][word] + 1) / (totalClassWords[label] + numWords);
			}
		}
	}
	
	/**
	 * Compute feature frequency n x m matrix, where
	 * n: number of classes, m: number of dimensions
	 * 
	 * TODO: Make more efficient by changing the inner loop
	 * to iterate over feature names (later converted to dimensions)
	 * instead of all dimensions.
	 */
	public void computeWordCounts(Dataset dataset) {
		
		for(Instance instance : dataset.getInstances()) {
			for(int index = 0; index < numWords; index++) {
				int label = labelAlphabet.getIndex(instance.getLabel());
				Float value = instance.getDimensionValue(index); 
				if(value != null) {
					wordCounts[label][index] += value;
				}
			}
		}
	}
	
	/**
	 * For each label, compute the number of examples that have that label.
	 */
	public void computeLabelCounts(Dataset dataset) {
		
		for(Instance instance : dataset.getInstances()) {
			int label = labelAlphabet.getIndex(instance.getLabel());
			labelCounts[label]++;
		}
	}
	
	/**
	 * Compute total number of words in each class.
	 */
	public void computeTotalClassWords(Dataset dataset) {

		for(Instance instance : dataset.getInstances()) {
			int label = labelAlphabet.getIndex(instance.getLabel());
			totalClassWords[label] += instance.getTotalMass();
		}
	}
	
	/**
	 * Run sampler.
	 */
	public void run() {

	  initialize();
	  for(int sample = 0; sample < numSamples; sample++) {
	    sample();
	  }
	}
	
	public double evaluate() {

	  int correct = 0;
	  
	  for(Instance instance : sampled.getInstances()) {
	    if(instance.getMisc() != null) {
	      
	      double cumulative = 0;
	      for(int label : instance.labelList) {
	        cumulative += label;
	      }
	      
	      int prediction;
	      if((cumulative / numSamples) < 0.5) {
	        prediction = 0;
	      } else {
	        prediction = 1;
	      }
	      
	      if(labelAlphabet.getString(prediction).equals(instance.getMisc())) {
	        correct++;
	      }
	    }
	  }

	  return (double) correct / test.size();
	}
}