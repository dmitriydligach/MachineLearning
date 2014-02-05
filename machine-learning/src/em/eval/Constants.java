package em.eval;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Constants {
  
  public final static int folds = 10;
  public final static int maxLabeled = 600;
  public final static int step = 5;
	
  public static final String cdData = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/data.txt";
	public static final String cdLabel = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/labels-cd.txt";
	
	public static final String ucData = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/data.txt";
  public static final String ucLabel = "/Users/Dima/Boston/Data/Phenotype/IBD/Data/labels-uc.txt";
  
  public static final String msData = "/Users/Dima/Boston/Data/Phenotype/MS/Data/data.txt";
  public static final String msLabel = "/Users/Dima/Boston/Data/Phenotype/MS/Data/labels.txt";
  public static final Set<String> msSourceLabels = new HashSet<String>(Arrays.asList("2", "3", "4", "5"));
  public static final String msTargetLabel = "2";
  
  public static final String t2dData = "/Users/Dima/Boston/Data/Phenotype/T2D/Data/data.txt";
  public static final String t2dLabel = "/Users/Dima/Boston/Data/Phenotype/T2D/Data/labels.txt";
  public static final Set<String> t2dSourceLabels = new HashSet<String>(Arrays.asList("\"possible\""));
  public static final String t2dTargetLabel = "\"no\"";
  
	public static final String outputFile = "/Users/Dima/Boston/Output/em.txt";
}
