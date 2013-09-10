package utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Misc {

	public static int getIndexOfLargestElement(double[] array) {
		
		int index = 0;
		double largest = array[0];

		for(int i = 0; i < array.length; i++) {
			if(array[i] > largest) {
				largest = array[i];
				index = i;
			}
		}
		
		return index;
	}
	
	/**
	 * Get buffered writer object for writing to file.
	 * If file already exsits, remove it first.
	 */
	public static BufferedWriter getWriter(String filePath) {
		
		File file = new File(filePath);
		if(file.exists()) {
			System.out.println(filePath + " already exists... deleting...");
			file.delete();
		}
		
		BufferedWriter bufferedWriter = null;
    try {
    	FileWriter fileWriter = new FileWriter(filePath);
    	bufferedWriter = new BufferedWriter(fileWriter);
    } catch (IOException e) {
	    e.printStackTrace();
    }
    
    return bufferedWriter;
	}
}
