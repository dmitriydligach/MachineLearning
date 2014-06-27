package semsup.eval;

import java.io.FileInputStream;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Set;

public class Constants {
  
  public static int folds;
  public static int devFolds;
  public static int maxLabeled;
  public static int step;
  public static int iterations;
  public static int devIterations;
  public static int rndSeed;
  public static double defaultLambda;
   
  public static List<String> phenotypes = new LinkedList<String>();
  public static List<Integer> unlabeledSizes = new LinkedList<Integer>();
	
  public static String cdData;
	public static String cdLabels;
	public static String ucData;
  public static String ucLabels;
  public static String msData;
  public static String msLabels;
  public static String t2dData;
  public static String t2dLabels;
  
  public static Set<String> msSourceLabels = new HashSet<String>(Arrays.asList("2", "3", "4", "5"));
  public static String msTargetLabel = "2";
  public static Set<String> t2dSourceLabels = new HashSet<String>(Arrays.asList("\"possible\""));
  public static String t2dTargetLabel = "\"no\"";
  
	public static String outputDir;
	
	/**
	 * Read constants from a properties file.
	 */
	public static void populate(String propertiesFile) {
	  
    Properties properties = new Properties();
    try {
      properties.load(new FileInputStream(propertiesFile));
    } catch (Exception e) {
      System.err.println("couldn't open properties file: " + propertiesFile);
    }
    
    folds = Integer.parseInt((String) properties.get("folds"));
    devFolds = Integer.parseInt((String) properties.get("devFolds"));
    maxLabeled = Integer.parseInt((String) properties.get("maxLabeled"));
    step = Integer.parseInt((String) properties.get("step"));
    iterations = Integer.parseInt((String) properties.get("iterations"));
    devIterations = Integer.parseInt((String) properties.get("devIterations"));
    rndSeed = Integer.parseInt((String) properties.get("rndSeed"));
    defaultLambda = Double.parseDouble((String) properties.get("defaultLambda"));
    
    for(String phenotype : ((String) properties.get("phenotypes")).split(",")) {
      phenotypes.add(phenotype);  
    }
    for(String unlabeledSize : ((String) properties.get("unlabeledSizes")).split(",")) {
      unlabeledSizes.add(Integer.parseInt(unlabeledSize));
    }
    
    cdData = (String) properties.get("cdData");
    cdLabels = (String) properties.get("cdLabels");
    ucData = (String) properties.get("ucData");
    ucLabels = (String) properties.get("ucLabels");
    msData = (String) properties.get("msData");
    msLabels = (String) properties.get("msLabels");
    t2dData = (String) properties.get("t2dData");
    t2dLabels = (String) properties.get("t2dLabels");
    
    outputDir = (String) properties.get("outputDir");
    print();
	}
	
	/**
	 * Print certain constants.
	 */
	public static void print() {
	  
	  System.out.format("%20s %d\n", "folds", folds);
	  System.out.format("%20s %d\n", "dev folds", devFolds);
	  System.out.format("%20s %d\n", "step", step);
	  System.out.format("%20s %d\n", "iterations", iterations);
	  System.out.format("%20s %d\n", "dev iterations", devIterations);
	  System.out.format("%20s %s\n", "phenotypes", phenotypes);
	  System.out.format("%20s %s\n", "unlabeled sizes", unlabeledSizes);
	}
}
