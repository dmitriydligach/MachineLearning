package classifier;

import java.util.Map;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;
import data.Dataset;
import data.Instance;

public class LibSvmClassifier {

	public svm_model model;
	
	// need this to supress training output
  private static svm_print_interface svm_print_null = new svm_print_interface()
  {
          public void print(String s) {}
  };

	/**
	 * Train an SVM model
	 */
	public void train(Dataset dataset) {
		
		svm_problem svmProblem = new svm_problem();
		svmProblem.x = new svm_node[dataset.size()][];
		svmProblem.y = new double[dataset.size()];
		svmProblem.l = dataset.size();
		int instanceCounter = 0;
		
		for(Instance instance : dataset.getInstances()) {
			Map<Integer, Float> vector = instance.getVector();
			svm_node[] nodes = new svm_node[vector.size()]; // libsvm vector representation
			int nodeCounter = 0;
			
			for(int dimension : vector.keySet()) {
				svm_node node = new svm_node(); // libsvm single dimension representation				
				node.index = dimension;
				node.value = instance.getDimensionValue(dimension);
				nodes[nodeCounter++] = node;
			}
			
			svmProblem.x[instanceCounter] = nodes;
			svmProblem.y[instanceCounter] = dataset.getLabelAlphabet().getIndex(instance.getLabel());
			instanceCounter++;
		}
		
		svm_parameter svmParameters = new svm_parameter();
		
    // default values (taken from svm_train.java)                                                                                   
    svmParameters.svm_type = svm_parameter.C_SVC;
    svmParameters.kernel_type = svm_parameter.LINEAR;
    svmParameters.degree = 1;
    svmParameters.gamma = 0;                                           
    svmParameters.coef0 = 0;
    svmParameters.nu = 0.5;
    svmParameters.cache_size = 100;
    svmParameters.C = 1;
    svmParameters.eps = 1e-3;
    svmParameters.p = 0.1;
    svmParameters.shrinking = 1;
    svmParameters.probability = 0;
    svmParameters.nr_weight = 0;
    svmParameters.weight_label = new int[0];
    svmParameters.weight = new double[0];
		
		// check parameters
    String error = svm.svm_check_parameter(svmProblem,svmParameters);
    if(error != null) {
    	System.out.println("parameter error: " + error);
    }
		
    svm.svm_set_print_string_function(svm_print_null);
		model = svm.svm_train(svmProblem, svmParameters);
	}
	
	/**
	 * Classify test set. Return accuracy.
	 */
	public double test(Dataset dataset) {
		
		int correct = 0;
		
		for(Instance instance : dataset.getInstances()) {
			Map<Integer, Float> vector = instance.getVector();
			svm_node[] nodes = new svm_node[vector.size()]; // libsvm vector representation
			int nodeCounter = 0;
			
			for(int dimension : vector.keySet()) {
				svm_node node = new svm_node(); // libsvm single dimension representation				
				node.index = dimension;
				node.value = instance.getDimensionValue(dimension);
				nodes[nodeCounter++] = node;
			}
			
			Double libSvmPrediction = svm.svm_predict(model, nodes); 
			String predictedLabel = dataset.getLabelAlphabet().getString(libSvmPrediction.intValue());
			
			if(predictedLabel.equals(instance.getLabel())) {
				correct++;
			}
		}
		
		return (double) correct / dataset.size();
	}
}
