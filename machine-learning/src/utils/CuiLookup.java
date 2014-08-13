package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tim Miller's code to lookup UMLS preferred name by CUI.
 * Originally called TopicCuis2Words.java.
 */
public class CuiLookup {

  HashMap<String,String> cui2term = null;
  HashMap<String,String> cui2tui = null;
  
  public CuiLookup(String cuiFile) throws IOException{
    cui2term = new HashMap<String,String>();
    cui2tui = new HashMap<String, String>();
    
    BufferedReader reader = new BufferedReader(new FileReader(cuiFile));
    String line;
    
    while((line = reader.readLine()) != null){
      String[] fields = line.split("\\|");
      cui2term.put(fields[0], fields[2]);
      cui2tui.put(fields[0], fields[5]);
    }
  }
  
  public String getTerm(String cui){
    return cui2term.get(cui);
  }
  
  public String getTui(String cui){
    return cui2tui.get(cui);
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {
    String defaultCui = "C0373675";
    CuiLookup mapper = null;
    try {
      mapper = new CuiLookup("resources/snomed-only-uniq-codes.txt");
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
      System.err.println("Cannot read file of cuis!");
      System.exit(-1);
    }
    
    Scanner scanner = new Scanner(System.in);
    Matcher m;
    Pattern cuiPatt = Pattern.compile("-?([Cc]\\d{7})(.*)");
    
    while(scanner.hasNextLine()){
      String line = scanner.nextLine().trim();
      String[] cuis = line.split("\\s+");
      for(String cui : cuis){
        m = cuiPatt.matcher(cui);
        if(m.matches()){
          cui = m.group(1).toUpperCase();
          String term = mapper.getTerm(cui);
          System.out.printf("%s%s\n", term, m.group(2));
        }
      }
    }
//    System.out.println(mapper.getTerm(cui));
    
  }

}
