package cv;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Utilities related to performing various splits and 
 * data subset selection related to n-fold cross-validation.
 * 
 * @author dmitriy dligach
 *
 */
public class CrossValidation {

	/**
	 * Assign each document from a collection of size = numOfDocs to a fold.
	 * Specify the source of randomness, e.g. Random(0). Use random assignment.
	 */
	public static int[] assignToFoldsRand(int numOfDocs, int numOfFolds, Random random) {

		int[] foldAssignment = new int[numOfDocs];
		
		for(int i = 0; i < numOfDocs; i++) {
			foldAssignment[i] = random.nextInt(numOfFolds); // random int between 0 (inclusive) and numOfFolds (exclusive)
		}
		
		return foldAssignment;
	}

	/**
	 * Assign to folds using deterministic fold assignment. The returned
	 * array contains the fold assignment. I.e. the instance i's fold is
	 * stored in the array's ith element.
	 */
	public static int[] assignToFolds(int numOfDocs, int numOfFolds) {
		
		int[] foldAssignment = new int[numOfDocs];
		int currentFold = 0; // folds numbered starting from zero
		
		for(int i = 0; i < numOfDocs; i++) {
			foldAssignment[i] = currentFold;
			currentFold++;
			if(currentFold == numOfFolds) {
				currentFold = 0;
			}
		}
		
		return foldAssignment;
	}
	
	/**
	 * Return document indices that are in test set
	 */
	public static ArrayList<Integer> getTestDocuments(int[] foldAssignments, int testFold) {
		
		ArrayList<Integer> testDocuments = new ArrayList<Integer>();
		
		for(int docIndex = 0; docIndex < foldAssignments.length; docIndex++) {
			if(foldAssignments[docIndex] == testFold) {
				testDocuments.add(docIndex);
			}
		}
		
		return testDocuments;
	}
	
	/**
	 * Return indicies of the documents that are not in the test set
	 */
public static ArrayList<Integer> getPoolDocuments(int[] foldAssignments, int testFold) {
		
		ArrayList<Integer> poolDocuments = new ArrayList<Integer>();
		
		for(int docIndex = 0; docIndex < foldAssignments.length; docIndex++) {
			if(foldAssignments[docIndex] != testFold) {
				poolDocuments.add(docIndex);
			}
		}
		
		return poolDocuments;
	}
	
	
	/**
	 * Return n randomly selected document indices that are in not in test set
	 */
	public static ArrayList<Integer> getPoolSample(int[] foldAssignments, int testFold, int selectionSize) {
		
		ArrayList<Integer> poolSample = new ArrayList<Integer>();
		
		ArrayList<Integer> poolDocuments = getPoolDocuments(foldAssignments, testFold);
		Collections.shuffle(poolDocuments);
		
		for(int i = 0; i < selectionSize; i++) {
			poolSample.add(poolDocuments.get(i));
		}
		
		return poolSample;
	}
	
	
	/**
	 * Get two samples of specified sizes from a list
	 */
	public static <TYPE> List<List<TYPE>> split(List<TYPE> list, int size1, int size2) {
		
		Collections.shuffle(list);
		
		List<TYPE> list1 = new ArrayList<TYPE>(list.subList(0, size1)); 
		List<TYPE> list2 = new ArrayList<TYPE>(list.subList(size1, size1 + size2));
		
		List<List<TYPE>> result = new ArrayList<List<TYPE>>(); 
		result.add(list1);
		result.add(list2);
		
		return result;
	}
	
	public static void main(String[] args) {
		
		List<List<Integer>> result = split(new ArrayList<Integer>(Arrays.asList(1,2,3,4,5,6,7,8,9,10)), 5, 5);
		System.out.println(result.get(0));
		System.out.println(result.get(1));
	}
}
