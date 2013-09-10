package misc;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class LibSvmTest {

	public static void main(String[] args) {
		
		svm_node dim0 = new svm_node(); // single dimension
		dim0.index = 0; 
		dim0.value = 1;
		svm_node dim1 = new svm_node(); // single dimension
		dim1.index = 5;
		dim1.value = 2;
		svm_node[] example = {dim0, dim1}; // single example
		
		svm_problem problem = new svm_problem();

		problem.x = new svm_node[2][]; // two examples
		problem.y = new double[2];
		
		problem.l = 2;
		
		problem.x[0] = example;
		problem.x[1] = example;

		problem.y[0] = 1;
		problem.y[1] = 0;
		
		svm_parameter parameter = new svm_parameter();
		parameter.svm_type = svm_parameter.C_SVC;
		parameter.kernel_type = svm_parameter.LINEAR;
		parameter.C = 1;
		
		svm_model model = svm.svm_train(problem, parameter);
	}
}
