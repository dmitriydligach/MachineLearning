package data;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;

public class I2b2Dataset extends Dataset {

	/**
	 * Load instances from comma-separated file. Map patient_num(s) to labels.
	 * The file is formated using i2b2 phenotype format. Each patient_num is 
	 * read only once (some datasets have multiple vectors for the same patient). 
	 */
	public void loadCSVFile(String vectorFile, String labelFile) throws FileNotFoundException {
    
		HashSet<String> patientNumsProcessed = new HashSet<String>();
		Map<String, String> patientNumToLabel = loadLabels(labelFile);
		
		File file = new File(vectorFile);
    Scanner scan = new Scanner(file);
    
    String[] columns = null;
    
    while(scan.hasNextLine()) {
    	String line = scan.nextLine();

    	// skip comments and empty lines
    	if(line.startsWith("#") || line.length() == 0) {
    		continue;
    	}
    	
    	// read the line containing column names
    	if(line.startsWith("patient_num")) {
    		columns = line.split("\\|");
    		continue;
    	}
    	
    	// now read the data
      String[] elements = line.split(",");
      String patientNum = elements[0];
      
      // we'll only need the vectors for which there are labels
      if(patientNumToLabel.containsKey(patientNum)) {
      	
      	// only store each patient once
      	if(patientNumsProcessed.contains(patientNum)) {
      		continue;
      	} else {
      		patientNumsProcessed.add(patientNum);
      	}
      	
      	Instance instance = new Instance();
      	instance.setLabel(patientNumToLabel.get(patientNum));

      	// iterate over vector elements
      	for(int i = 1; i < elements.length; i++) {
      		String feature = columns[i];
      		Float value = Float.parseFloat(elements[i]);
      		if(value != 0.0) {
      			instance.addFeature(feature, value);
      		}
      	}

      	instances.add(instance);
      }
    }
	}
	
	/**
	 * Load labels from a comma-separated file: <patient_num>,<label>.
	 */
	private static Map<String, String> loadLabels(String inputFile) throws FileNotFoundException {
		
		Map<String, String> patientNumToLabel = new HashMap<String, String>();
		
		File file = new File(inputFile);
		Scanner scan = new Scanner(file);
		
		while(scan.hasNextLine()) {
			String line = scan.nextLine();
			String[] elements = line.split(",");
			patientNumToLabel.put(elements[0], elements[1]);
		}
		
		return patientNumToLabel;
	}
}
