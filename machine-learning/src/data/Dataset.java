package data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import com.google.common.base.Joiner;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;

import cv.CrossValidation;

public class Dataset {

	// list of examples
	protected List<Instance> instances;
	
	// string->int and int->string mappings
	protected Alphabet featureAlphabet;
	protected Alphabet labelAlphabet;
	
	// document frequencies for features
	protected Multiset<String> dfs; 

	/**
	 * Constructor.
	 */
	public Dataset() {
		instances = new ArrayList<Instance>();
		featureAlphabet = new Alphabet();
		labelAlphabet = new Alphabet();
		dfs = HashMultiset.create();
	}
	
	/**
	 * Create a dataset from a list of instances
	 */
	public Dataset(List<Instance> sourceInstances) {

		instances = new ArrayList<Instance>();
		featureAlphabet = new Alphabet();
		labelAlphabet = new Alphabet();
		dfs = HashMultiset.create();

		for(Instance sourceInstance : sourceInstances) {
			instances.add(new Instance(sourceInstance));
		}
	}
	
	/** 
	 * Create a dataset from two lists of instances
	 */
	public Dataset(List<Instance> sourceInstances1, List<Instance> sourceInstances2) {
	  
	  instances = new ArrayList<Instance>();
	  featureAlphabet = new Alphabet();
	  labelAlphabet = new Alphabet();
	  dfs = HashMultiset.create();

	  for(Instance sourceInstance : sourceInstances1) {
	    instances.add(new Instance(sourceInstance));
	  }
	  for(Instance sourceInstance : sourceInstances2) {
      instances.add(new Instance(sourceInstance));
    }
	}

  /** 
   * Create a dataset from three lists of instances
   */
  public Dataset(List<Instance> sourceInstances1, List<Instance> sourceInstances2, List<Instance> sourceInstances3) {
    
    instances = new ArrayList<Instance>();
    featureAlphabet = new Alphabet();
    labelAlphabet = new Alphabet();
    dfs = HashMultiset.create();

    for(Instance sourceInstance : sourceInstances1) {
      instances.add(new Instance(sourceInstance));
    }
    for(Instance sourceInstance : sourceInstances2) {
      instances.add(new Instance(sourceInstance));
    }
    for(Instance sourceInstance : sourceInstances3) {
      instances.add(new Instance(sourceInstance));
    }
  }
	
	/**
	 * Load instances from comma-separated file. 
	 * The file must conform to this format:
	 * 
	 * <label>,<feature>:<value>,...
	 */
	public void loadCSVFile(String inputFile) throws FileNotFoundException {
    File file = new File(inputFile);
    Scanner scan = new Scanner(file);

    while(scan.hasNextLine()) {
      String line = scan.nextLine();
      String[] elements = line.split(","); // e.g. <label>,<feat>:<value>,...
      
      Instance instance = new Instance();
      instance.setLabel(elements[0]);

      // iterate over feature name-value pairs
      for(int i = 1; i < elements.length; i++) {
      	String[] pair = elements[i].split(":");
      	instance.addFeature(pair[0], Float.parseFloat(pair[1]));
      }
  
      instances.add(instance);
    }
	}
	
	/**
	 * Load instances from a space-separated Mallet-style formatted file.
	 * The file must look like this:
	 * 
	 * <feature1>:<value1> <feature2>:<value2> ... <label>
	 */
	public void loadMalletFile(String inputFile) throws FileNotFoundException {
    File file = new File(inputFile);
    Scanner scan = new Scanner(file);

    while(scan.hasNextLine()) {
      String line = scan.nextLine();
      String[] elements = line.split(" "); 
      
      Instance instance = new Instance();
      instance.setLabel(elements[elements.length - 1]);

      // iterate over feature name-value pairs
      for(int i = 0; i < elements.length - 1; i++) {
      	String[] pair = elements[i].split(":");
      	instance.addFeature(pair[0], Float.parseFloat(pair[1]));
      }
  
      instances.add(instance);
    }
	}
	
	/**
	 * Remove n random instances and return them.
	 * Specify the source of randomness.
	 */
	public List<Instance> popRandom(int n, Random random) {

		Collections.shuffle(instances, random);
		List<Instance> removedInstances = new ArrayList<Instance>();
	
		for(int i = 0; i < n; i++) {
			if(instances.size() < 1) {
				System.out.println("pool size too small: " + instances.size());
				break;
			}

			Instance removedInstance = instances.remove(0);
			removedInstances.add(removedInstance);
		}
				
		return removedInstances;
	}
	
	/**
	 * Add instances to the @Dataset.
	 */
	public void add(List<Instance> sourceInstances) {
		
		for(Instance instance : sourceInstances) {
			instances.add(new Instance(instance));
		}
	}
	
	/**
	 * Add a single instance.
	 */
	public void add(Instance instance) {
		instances.add(instance);
	}
	
	/**
	 * Generate alphabets based on the instances contained in this object.
	 * TODO: need to generate doc frequencies here too?
	 */
	public void makeAlphabets() {
		
		// alphabets may already exist, so create new ones
		featureAlphabet = new Alphabet();
		labelAlphabet = new Alphabet();
		
		for(Instance instance : instances) {
			labelAlphabet.add(instance.getLabel());
			for(String feature : instance.getFeatures().keySet()) {
				featureAlphabet.add(feature);
			}
		}
	}

	/**
	 * Generate only label alphabet. Useful for the cases when
	 * feature alphabet is not needed (i.e. when sparse vectors 
	 * are available directly).
	 */
	public void makeLabelAlphabet() {
		
		// alphabet may already exist, so create a new one
		labelAlphabet = new Alphabet();
		
		for(Instance instance : instances) {
			labelAlphabet.add(instance.getLabel());
		}
	}
	
	/**
	 * Set alphabets. Useful when using alphabets generated 
	 * from another data set (e.g. during training).
	 */
	public void setAlphabets(Alphabet labelAlphabet, Alphabet featureAlphabet) {
		this.labelAlphabet = labelAlphabet;
		this.featureAlphabet = featureAlphabet;
	}
	
	/**
	 * Set label alphabet. E.g. When the training set is small, it is useful to set
	 * test set alphabet to the one obtained from the entire data set.
	 */
	public void setLabelAlphabet(Alphabet labelAlphabet) {
		this.labelAlphabet = labelAlphabet;
	}
	
	/**
	 * Set feature alphabet. In some cases (e.g. above) may need to set 
	 * alphabets separately rather than both at the same time.
	 */
	public void setFeatureAlphabet(Alphabet featureAlphabet) {
		this.featureAlphabet = featureAlphabet;
	}
	
	/**
	 * Convert each instance's label into a probability distribution
	 * over all possible labels. Basically, if the label for this instance
	 * is "yes", than the probability of "yes" should be 1 and the 
	 * probability of all other labels should be 0. This is useful when
	 * a probability distribution over labels is needed rather than
	 * a single gold labels for each instance (e.g. for EM algorithm).
	 */
	public void setInstanceClassProbabilityDistribution(Set<String> uniqueLabels) {
	  
	  for(Instance instance : instances) {
	    instance.setClassProbabilities(uniqueLabels);
	  }
	}
	
	/**
	 * Label alphabet (labels->int and int->labels)
	 */
	public Alphabet getLabelAlphabet() {
		return labelAlphabet;
	}
	
	/**
	 * Feature alphabet (features->int and int->features)
	 */
	public Alphabet getFeatureAlphabet() {
		return featureAlphabet;
	}
	
	/**
	 * Make sparse vector representations for each instance.
	 */
	public void makeVectors() {
		
		for(Instance instance : instances) {
			
			// vector may already exist
			instance.resetVector();
			
			for(int index = Alphabet.startIndex; index < Alphabet.startIndex + featureAlphabet.size(); index++) {
				String featureName = featureAlphabet.getString(index); 
				Float featureValue = instance.getFeatureValue(featureName);
				if(featureValue != null) {
					instance.setVector(index, featureValue);
				}
			}
		}
	}
	
	/**
   * Discard features below and above the specified 
   * min and max document frequencies. 
   * 
   * Alphabets are not affected and may need to be regenerated.
	 */
	public void discardFeatures(int min, int max) {
		
		// create a document frequency histogram
		for(Instance instance : instances) {
			for(String feature : instance.getFeatures().keySet()) {
				dfs.add(feature);
			}
		}
		
		// discard low and high-frequency features
		for(Instance instance : instances) {
			Iterator<Map.Entry<String, Float>> iterator = instance.getFeatures().entrySet().iterator();
			while(iterator.hasNext()) {
				Map.Entry<String, Float> entry = iterator.next();
				String feature = entry.getKey();
				if(dfs.count(feature) < min || dfs.count(feature) > max) {
					iterator.remove();
				}
			}
		}
	}
	
	/**
	 * Split into n parts for n-fold cross-validation.
	 * Specify the source of randomness. 
	 */
	public Split[] split(int n) {
		
		Split[] splits = new Split[n];
		int[] foldAssignment = CrossValidation.assignToFolds(instances.size(), n);
		
		for(int fold = 0; fold < n; fold++) {
			List<Instance> trainInstances = new ArrayList<Instance>();
			List<Instance> testInstances = new ArrayList<Instance>();
			
			for(int instanceIndex : CrossValidation.getPoolDocuments(foldAssignment, fold)) {
				trainInstances.add(instances.get(instanceIndex));
			}
			
			for(int instanceIndex : CrossValidation.getTestDocuments(foldAssignment, fold)) {
				testInstances.add(instances.get(instanceIndex));
			}
			
			splits[fold] = new Split(trainInstances, testInstances);
		}
		
		return splits;
	}
	
	/**
	 * Get two subsets of instances of size1 and size2. 
	 * Wrap each subset into a Dataset object.
	 * Specify source of randomness.
	 */
	public Dataset[] split(int size1, int size2, Random random) {
	  
	  List<List<Instance>> twoParts = CrossValidation.split(instances, size1, size2, random);
	  Dataset[] result = new Dataset[2];
	  
	  result[0] = new Dataset(twoParts.get(0));
	  result[1] = new Dataset(twoParts.get(1));
	  
	  return result;
	}
	
	/**
	 * Write sparse vectors to file in libsvm format.
	 * Assumes sparse vectors are populated.
	 */
	public void writeLibsvmFile(String outputFile) throws IOException {

		BufferedWriter bufferedWriter = getWriter(outputFile);
		
		for(Instance instance : instances) {
			int label = labelAlphabet.getIndex(instance.getLabel());
			bufferedWriter.write(Integer.toString(label) + " " + instance.getVectorAsString(":", " ") + "\n");
		}
		
		bufferedWriter.close();
	}
	
	/**
	 * Write sparse vectors to file in weka arff format.
	 * Populate sparse vectors before running this method.
	 */
	public void writeWekaFile(String outputFile) throws IOException {
		
		BufferedWriter bufferedWriter = getWriter(outputFile);
		bufferedWriter.write("@relation mydata\n\n");
		
		// write features (attribute)
		for(String feature : featureAlphabet.getStrings()) {
			bufferedWriter.write("@attribute " + feature.replaceAll("[^A-Za-z0-9]", "_nonalpha_") + " numeric\n");
		}
		
		// write labels
		List<String> labels = labelAlphabet.getStrings();
		Joiner joiner = Joiner.on(",");
		bufferedWriter.write("@attribute label {" + joiner.join(labels) + "}\n");
		
		// write feature vectors
		bufferedWriter.write("\n@data\n");
		for(Instance instance : instances) {
			String label = String.format("%d %s", featureAlphabet.size(), instance.getLabel());
			bufferedWriter.write("{" + instance.getVectorAsString(" ", ",") + "," + label + "}\n");
		}
		
		bufferedWriter.close();
	}
	
	/**
	 * Get buffered writer object for writing to file.
	 */
	public static BufferedWriter getWriter(String filePath) {
		
		BufferedWriter bufferedWriter = null;
    try {
    	FileWriter fileWriter = new FileWriter(filePath);
    	bufferedWriter = new BufferedWriter(fileWriter);
    } catch (IOException e) {
	    e.printStackTrace();
    }
    
    return bufferedWriter;
	}

	/**
	 * Get total number of instances.
	 */
	public int size() {
		return instances.size();
	}
	
	/**
	 * Get total number of unique classes in the data.
	 */
	public int getNumberOfClasses() {
		return labelAlphabet.size();
	}
	
	/**
	 * Get total number of unique features in the data.
	 */
	public int getNumberOfDimensions() {
		return featureAlphabet.size();
	}
	
	/**
	 * Getter for the list of instances.
	 */
	public List<Instance> getInstances() {
		return instances;
	}
	
	/** 
	 * Get the instance at the specified index.
	 */
	public Instance getInstance(int index) {
		return instances.get(index);
	}
	
	/** 
	 * Remove the instance at the specified index.
	 * Return the removed instance.
	 */
	public Instance removeInstance(int index) {
		return instances.remove(index);
	}
	
	/**
	 * Normalize feature values.
	 */
	public void normalize() {
	  
	  for(Instance instance: instances) {
	    instance.normalize();
	  }
	}
	
	/**
	 * Remove label from 'label' field and copy it to 'temp' field.
	 * Can be useful for evaluations / sanity checking.
	 * WARNING: This method modifies the dataset.
	 */
	public void hideLabels() {
	  
	  for(Instance instance : instances) {
	    instance.setTemp(instance.getLabel());
	    instance.setLabel(null);
	  }
	}
	
	/**
   * Remove label from 'temp' field and copy it to 'label' field.
   * This method should be run after hideLabels() to restore the original state.
   * WARNING: This method modifies the dataset.
   */
  public void restoreLabels() {
    
    for(Instance instance : instances) {
      instance.setLabel(instance.getTemp());
      instance.setTemp(null);
    }
  }
}
